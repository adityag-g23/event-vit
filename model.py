"""Event-biased ViT-B/16 for object detection on event+RGB data.

Architecture:
  - Pretrained ViT-B/16 backbone (all weights frozen)
  - Last 4 transformer blocks: Attention replaced by EventBiasAttention
      scores = Q@K.T / sqrt(d) + alpha * event_density_j
    where alpha is a per-layer learned scalar (init 0), event_density_j
    is a (B, num_patches) tensor set externally before each forward pass.
  - DetectionHead: 2 linear layers on patch tokens -> (objectness, cx, cy, w, h)
  - Trainable params: 4 alpha scalars + DetectionHead only
"""

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention


class EventBiasAttention(Attention):
    """Attention with an additive event-density bias injected before softmax."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.event_bias: torch.Tensor | None = None  # (B, num_patches); set externally
        self.fused_attn = False  # must be False so we can intercept the logits

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, heads, N, N)

        if self.event_bias is not None:
            # Prepend zero for CLS position, broadcast over heads and query axis
            cls_zero = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
            bias = torch.cat([cls_zero, self.event_bias], dim=1)  # (B, N)
            attn = attn + self.alpha * bias.unsqueeze(1).unsqueeze(2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Store for passive capture in visualize.py
        self._attn_weights = attn.detach()

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DetectionHead(nn.Module):
    """Two-layer MLP on patch tokens predicting objectness + box per patch."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes + 4),  # obj_score(s) + (cx,cy,w,h)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, num_patches, embed_dim)
        return self.net(patch_tokens)  # (B, num_patches, num_classes+4)


def _copy_attn_weights(old_attn: Attention, new_attn: EventBiasAttention) -> None:
    """Copy all matching parameters from old attention to new, skip alpha."""
    old_sd = old_attn.state_dict()
    new_attn.load_state_dict(old_sd, strict=False)  # alpha key absent -> stays zeros


def build_model(
    num_classes: int = 1,
    pretrained: bool = True,
    patch_size: int = 16,
) -> tuple[nn.Module, DetectionHead]:
    """Load ViT-B/16, replace last 4 attention modules, freeze, return (vit, head).

    Trainable parameters after this call:
      - 4 alpha scalars (one per replaced block)
      - All DetectionHead parameters
    """
    vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
    embed_dim = vit.embed_dim
    num_heads = vit.blocks[0].attn.num_heads

    # Replace last 4 attention modules with EventBiasAttention
    for block in vit.blocks[-4:]:
        old_attn = block.attn
        qkv_bias = old_attn.qkv.bias is not None
        qk_norm  = not isinstance(old_attn.q_norm, nn.Identity)
        new_attn = EventBiasAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=old_attn.attn_drop.p,
            proj_drop=old_attn.proj_drop.p,
        )
        _copy_attn_weights(old_attn, new_attn)
        block.attn = new_attn

    # Freeze all backbone parameters
    for param in vit.parameters():
        param.requires_grad_(False)

    # Unfreeze only the 4 alpha scalars
    for block in vit.blocks[-4:]:
        block.attn.alpha.requires_grad_(True)

    head = DetectionHead(embed_dim=embed_dim, hidden_dim=256, num_classes=num_classes)
    return vit, head


def set_event_bias(vit: nn.Module, density: torch.Tensor) -> None:
    """Set event_bias on all EventBiasAttention modules before a forward pass.

    Args:
        vit: the ViT backbone returned by build_model
        density: (B, num_patches) event density tensor (or zeros for RGB-only)
    """
    for block in vit.blocks[-4:]:
        if isinstance(block.attn, EventBiasAttention):
            block.attn.event_bias = density


def forward_pass(
    vit: nn.Module,
    head: DetectionHead,
    images: torch.Tensor,
    density: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: set bias, run backbone, run head.

    Returns:
        (B, num_patches, num_classes+4) prediction tensor
    """
    set_event_bias(vit, density)
    features = vit.forward_features(images)  # (B, N+prefix, D)
    num_prefix = getattr(vit, "num_prefix_tokens", 1)
    patch_tokens = features[:, num_prefix:, :]  # (B, num_patches, D)
    return head(patch_tokens)
