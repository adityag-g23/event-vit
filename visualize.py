"""Attention map visualization: side-by-side RGB | events | attn (no events) | attn (with events).

Hooks are used *passively* in this file to capture attention weights stored by EventBiasAttention.
The backbone forward is not modified here in any way.
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as T

from model import build_model, set_event_bias, forward_pass
from event_utils import batch_events_to_patch_density

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

IMG_SIZE   = 224
PATCH_SIZE = 16


def _load_image(path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Return (normalized tensor (1,3,H,W), uint8 HWC numpy)."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    t = T.ToTensor()(img).unsqueeze(0)             # (1, 3, H, W) in [0,1]
    t_norm = (t - _IMG_MEAN) / _IMG_STD
    return t_norm, arr


def _register_attn_hook(vit) -> tuple[dict, "handle"]:
    """Register a passive forward hook on the last block to capture attn weights."""
    captured = {}

    def hook_fn(module, inp, out):
        # EventBiasAttention stores _attn_weights after softmax
        captured["weights"] = module._attn_weights  # (B, heads, N, N)

    handle = vit.blocks[-1].attn.register_forward_hook(hook_fn)
    return captured, handle


def _attn_to_heatmap(captured: dict, patch_h: int, patch_w: int) -> np.ndarray:
    """Extract CLS→patch attention, reshape to (patch_h, patch_w), return float32."""
    weights = captured.get("weights")
    if weights is None:
        return np.zeros((patch_h, patch_w), dtype=np.float32)
    # Average over heads; CLS is query 0, patches are keys 1..
    cls_attn = weights[0].mean(dim=0)[0, 1:].cpu().numpy()  # (num_patches,)
    return cls_attn.reshape(patch_h, patch_w)


def _overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Resize heatmap to image resolution and blend as a colormap overlay."""
    h, w = rgb.shape[:2]
    hmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    hmap_big = np.array(
        Image.fromarray((hmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    colored = (cm.inferno(hmap_big)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * rgb).astype(np.uint8)


def _density_to_image(density: torch.Tensor, patch_h: int, patch_w: int) -> np.ndarray:
    """Render (num_patches,) density as a colored image."""
    grid = density.reshape(patch_h, patch_w).cpu().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    big = np.array(
        Image.fromarray((grid * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    ) / 255.0
    return (cm.plasma(big)[:, :, :3] * 255).astype(np.uint8)


def visualize(
    vit,
    head,
    image_path: str,
    events_x: np.ndarray,
    events_y: np.ndarray,
    device: torch.device,
    out_path: str = "attention_comparison.png",
) -> None:
    """Generate and save the 4-panel attention comparison figure."""
    img_tensor, img_rgb = _load_image(image_path)
    img_tensor = img_tensor.to(device)

    density_batch = batch_events_to_patch_density(
        [{"x": events_x, "y": events_y}], IMG_SIZE, IMG_SIZE, PATCH_SIZE
    ).to(device)  # (1, num_patches)
    zero_density = torch.zeros_like(density_batch)

    pH = pW = IMG_SIZE // PATCH_SIZE

    # --- Pass 1: RGB-only (zero events) ---
    captured_rgb, handle = _register_attn_hook(vit)
    with torch.no_grad():
        forward_pass(vit, head, img_tensor, zero_density)
    handle.remove()
    attn_rgb = _attn_to_heatmap(captured_rgb, pH, pW)

    # --- Pass 2: RGB + Events ---
    captured_ev, handle = _register_attn_hook(vit)
    with torch.no_grad():
        forward_pass(vit, head, img_tensor, density_batch)
    handle.remove()
    attn_ev = _attn_to_heatmap(captured_ev, pH, pW)

    # --- Build 4 panels ---
    panel_rgb    = img_rgb
    panel_events = _density_to_image(density_batch[0], pH, pW)
    panel_attn_base = _overlay_heatmap(img_rgb, attn_rgb)
    panel_attn_ev   = _overlay_heatmap(img_rgb, attn_ev)

    panels = [panel_rgb, panel_events, panel_attn_base, panel_attn_ev]
    titles = ["RGB image", "Event density", "Attention (RGB only)", "Attention (RGB+Event)"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True, help="Path to RGB image")
    parser.add_argument("--events_npy",  required=True,
                        help="Path to .npy file with shape (N,2): columns x,y")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--out",         default="attention_comparison.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit, head = build_model(num_classes=args.num_classes, pretrained=(args.checkpoint is None))
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        head.load_state_dict(ckpt["head"])
        for i, block in enumerate(vit.blocks[-4:]):
            key = f"block_{i}"
            if key in ckpt.get("vit_alpha", {}):
                block.attn.alpha.data.fill_(ckpt["vit_alpha"][key])
    vit, head = vit.to(device), head.to(device)
    vit.eval(); head.eval()

    ev = np.load(args.events_npy)  # (N, 2)
    visualize(vit, head, args.image, ev[:, 0], ev[:, 1], device, out_path=args.out)


if __name__ == "__main__":
    main()
