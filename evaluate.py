"""Compute mAP@0.5 per lighting condition on the PEOD validation split."""

import argparse
import collections
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PEODDataset, collate_fn, CONDITIONS
from model import build_model, forward_pass


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def compute_ap(scores: list, matches: list, n_gt: int, iou_thresh: float = 0.5) -> float:
    """Compute Average Precision from sorted predictions."""
    if n_gt == 0:
        return float("nan")
    order   = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = [matches[i] for i in order]
    fp = [1 - m for m in tp]
    tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    recall    = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    # Append sentinel
    recall    = torch.cat([torch.zeros(1), recall,    torch.ones(1)])
    precision = torch.cat([torch.ones(1),  precision, torch.zeros(1)])
    # Area under curve (trapezoidal)
    return float(torch.trapz(precision, recall))


# ---------------------------------------------------------------------------
# Inference + collection
# ---------------------------------------------------------------------------

IMG_SIZE   = 224
PATCH_SIZE = 16
IOU_THRESH = 0.5
OBJ_THRESH = 0.3  # sigmoid threshold for patch objectness


def decode_preds(preds: torch.Tensor, img_size: int, patch_size: int):
    """Convert (B, num_patches, 5) logits to per-image lists of (score, box xyxy)."""
    B = preds.shape[0]
    pH = pW = img_size // patch_size
    results = []
    for b in range(B):
        scores_raw = torch.sigmoid(preds[b, :, 0])  # (num_patches,)
        boxes_raw  = preds[b, :, 1:]                # (num_patches, 4)  cxcywh

        keep = scores_raw > OBJ_THRESH
        s = scores_raw[keep]
        bx = boxes_raw[keep]
        bx_xyxy = cxcywh_to_xyxy(bx).clamp(0, 1)
        results.append((s.cpu(), bx_xyxy.cpu()))
    return results


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(vit, head, loader, device) -> dict[str, float]:
    """Return mAP@0.5 per condition."""
    vit.eval(); head.eval()

    # per condition: list of (score, is_tp)  and total GT count
    pred_store: dict[str, list] = {c: [] for c in CONDITIONS + ("all",)}
    gt_counts:  dict[str, int]  = {c: 0  for c in CONDITIONS + ("all",)}

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images    = batch["image"].to(device)
            density   = batch["density"].to(device)
            boxes_gt  = batch["boxes"]    # (B, max_boxes, 4) xyxy norm, cpu
            labels_gt = batch["labels"]   # (B, max_boxes)

            preds    = forward_pass(vit, head, images, density)
            decoded  = decode_preds(preds.cpu(), IMG_SIZE, PATCH_SIZE)
            conds    = batch["condition"]

            for b, (scores, pred_boxes) in enumerate(decoded):
                cond = conds[b]
                valid = labels_gt[b] >= 0
                gt_b  = cxcywh_to_xyxy(boxes_gt[b][valid]) if valid.any() else torch.zeros(0, 4)
                n_gt  = gt_b.shape[0]
                gt_counts[cond] += n_gt
                gt_counts["all"]  += n_gt

                matched_gt = set()
                for si, (sc, pb) in enumerate(zip(scores.tolist(), pred_boxes)):
                    pb = pb.unsqueeze(0)
                    is_tp = 0
                    if n_gt > 0:
                        iou = box_iou(pb, gt_b)[0]
                        best_i = int(iou.argmax())
                        if float(iou[best_i]) >= IOU_THRESH and best_i not in matched_gt:
                            is_tp = 1
                            matched_gt.add(best_i)
                    pred_store[cond].append((sc, is_tp))
                    pred_store["all"].append((sc, is_tp))

    results: dict[str, float] = {}
    for cond in list(CONDITIONS) + ["all"]:
        pairs = pred_store[cond]
        if pairs:
            sc_list = [p[0] for p in pairs]
            tp_list = [p[1] for p in pairs]
        else:
            sc_list, tp_list = [], []
        results[cond] = compute_ap(sc_list, tp_list, gt_counts[cond])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = PEODDataset(args.data_root, IMG_SIZE, PATCH_SIZE, split="val")
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    vit, head = build_model(num_classes=args.num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    head.load_state_dict(ckpt["head"])
    # Restore alpha values
    for i, block in enumerate(vit.blocks[-4:]):
        key = f"block_{i}"
        if key in ckpt.get("vit_alpha", {}):
            block.attn.alpha.data.fill_(ckpt["vit_alpha"][key])

    vit, head = vit.to(device), head.to(device)
    results = evaluate(vit, head, val_dl, device)

    print(f"\nmAP@{IOU_THRESH} by condition:")
    for cond, ap in results.items():
        tag = f"  {cond:<14s}"
        print(f"{tag}  {ap:.4f}" if not (ap != ap) else f"{tag}  N/A (no GT)")


if __name__ == "__main__":
    main()
