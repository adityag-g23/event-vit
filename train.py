"""Training loop for the event-biased ViT detection model."""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PEODDataset, collate_fn
from model import build_model, forward_pass


# ---------------------------------------------------------------------------
# Target assignment
# ---------------------------------------------------------------------------

def assign_targets(
    boxes: torch.Tensor,      # (B, max_boxes, 4) normalized xyxy
    labels: torch.Tensor,     # (B, max_boxes)  -1 = padding
    img_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign GT boxes to patches by center containment.

    Returns:
        obj_targets: (B, num_patches) 0/1 float
        box_targets: (B, num_patches, 4) normalized cxcywh (0 for negatives)
    """
    B, max_boxes, _ = boxes.shape
    pH = pW = img_size // patch_size
    num_patches = pH * pW
    device = boxes.device

    obj_targets = torch.zeros(B, num_patches, device=device)
    box_targets  = torch.zeros(B, num_patches, 4, device=device)

    for b in range(B):
        for n in range(max_boxes):
            if labels[b, n] < 0:
                break
            x1, y1, x2, y2 = boxes[b, n].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w,  h  = x2 - x1,       y2 - y1
            pi = min(int(cy * pH), pH - 1)
            pj = min(int(cx * pW), pW - 1)
            patch_idx = pi * pW + pj
            obj_targets[b, patch_idx] = 1.0
            box_targets[b, patch_idx] = torch.tensor([cx, cy, w, h], device=device)

    return obj_targets, box_targets


def detection_loss(
    preds: torch.Tensor,       # (B, num_patches, 1+4)
    obj_targets: torch.Tensor, # (B, num_patches)
    box_targets: torch.Tensor, # (B, num_patches, 4)
    box_weight: float = 5.0,
) -> torch.Tensor:
    obj_pred  = preds[..., 0]           # (B, num_patches)
    box_pred  = preds[..., 1:]          # (B, num_patches, 4)

    loss_obj = F.binary_cross_entropy_with_logits(obj_pred, obj_targets)

    pos_mask = obj_targets.bool()
    if pos_mask.any():
        loss_box = F.smooth_l1_loss(box_pred[pos_mask], box_targets[pos_mask])
    else:
        loss_box = box_pred.sum() * 0.0

    return loss_obj + box_weight * loss_box


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    vit, head, loader, optimizer, device, img_size, patch_size
) -> float:
    vit.train(False)   # backbone always eval (frozen)
    head.train(True)

    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images   = batch["image"].to(device)
        density  = batch["density"].to(device)
        boxes    = batch["boxes"].to(device)
        labels   = batch["labels"].to(device)

        preds = forward_pass(vit, head, images, density)

        obj_t, box_t = assign_targets(boxes, labels, img_size, patch_size)
        loss = detection_loss(preds, obj_t, box_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--num_classes",type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = PEODDataset(args.data_root, args.img_size, args.patch_size, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    vit, head = build_model(num_classes=args.num_classes, pretrained=True)
    vit, head = vit.to(device), head.to(device)

    trainable = [p for p in list(vit.parameters()) + list(head.parameters())
                 if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            vit, head, train_dl, optimizer, device, args.img_size, args.patch_size
        )
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}")

        if epoch % 5 == 0:
            ckpt = {
                "epoch":    epoch,
                "vit_alpha": {f"block_{i}": vit.blocks[-(4-i)].attn.alpha.item()
                               for i in range(4)},
                "head":     head.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch{epoch:03d}.pt"))


if __name__ == "__main__":
    main()
