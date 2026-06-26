"""End-to-end data + model verification. Run this before training.

Usage (on server):
    cd ~/Event-ViT
    python project/verify.py --data_root dataset/

Checks, in order:
  1. .dat binary format  -- x/y in sensor range, events not empty
  2. COCO JSON loading   -- box coords in [0,1], label range
  3. Timestamp CSV       -- correct row count vs frame count
  4. Dataset samples     -- shapes, dtype, density nonzero
  5. DataLoader batch    -- collate, no crash
  6+7. Forward + backward -- prediction shape, loss, gradients
"""

import argparse, os, sys, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

SENSOR_H, SENSOR_W = 720, 1280
IMG_SIZE    = 224
PATCH_SIZE  = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
NUM_CLASSES = 6
OK  = "\033[32mv\033[0m"
ERR = "\033[31mX\033[0m"

def check(cond, msg):
    print(f"  {'OK' if cond else 'FAIL'}  {msg}")
    if not cond:
        sys.exit(1)

# ── 1. DAT FORMAT ──────────────────────────────────────────────────
def verify_dat(root):
    print("\n── 1. DAT FORMAT ──")
    dats = sorted(glob.glob(os.path.join(root, "train", "*.dat")))
    check(len(dats) > 0, f"Found {len(dats)} .dat files in train/")
    from dataset import load_dat_events
    path = dats[0]
    print(f"  File: {path}  ({os.path.getsize(path):,} bytes)")
    with open(path, "rb") as f:
        first = f.read(64)
    print(f"  Has ASCII header: {first[:1] == b'%'}")
    print(f"  First 32 bytes hex: {first[:32].hex()}")
    ev_x, ev_y, ev_t = load_dat_events(path)
    check(len(ev_x) > 0, f"Loaded {len(ev_x):,} events (must be > 0)")
    x_min, x_max = int(ev_x.min()), int(ev_x.max())
    y_min, y_max = int(ev_y.min()), int(ev_y.max())
    t_min, t_max = int(ev_t.min()), int(ev_t.max())
    print(f"  x range: [{x_min}, {x_max}]  (sensor W={SENSOR_W})")
    print(f"  y range: [{y_min}, {y_max}]  (sensor H={SENSOR_H})")
    print(f"  t range: [{t_min}, {t_max}] us")
    if x_max > SENSOR_W or y_max > SENSOR_H:
        print("  *** DAT DECODER WRONG — coords exceed sensor size ***")
        print("  Fix load_dat_events() bit-unpacking in dataset.py then re-run.")
        sys.exit(1)
    check(x_max <= SENSOR_W, f"x_max {x_max} <= {SENSOR_W}")
    check(y_max <= SENSOR_H, f"y_max {y_max} <= {SENSOR_H}")

# ── 2. COCO JSON ───────────────────────────────────────────────────
def verify_coco(root):
    print("\n── 2. COCO JSON ──")
    jsons = sorted(glob.glob(
        os.path.join(root, "coco_0111_last", "train", "**", "*.json"), recursive=True))
    check(len(jsons) > 0, f"Found {len(jsons)} annotation JSONs")
    from dataset import load_coco_json
    fname_anns = load_coco_json(jsons[0])
    n_frames = len(fname_anns)
    n_boxes  = sum(v["boxes"].shape[0] for v in fname_anns.values())
    check(n_frames > 0, f"{jsons[0].split('/')[-1]}: {n_frames} frames, {n_boxes} boxes")
    sample_boxes  = next(iter(fname_anns.values()))["boxes"]
    sample_labels = next(iter(fname_anns.values()))["labels"]
    check(sample_boxes.max() <= 1.0 + 1e-4,
          f"Boxes normalized <= 1.0 (max={sample_boxes.max():.4f})")
    check(sample_boxes.min() >= 0.0 - 1e-4,
          f"Boxes >= 0.0 (min={sample_boxes.min():.4f})")
    check(sample_labels.max() < NUM_CLASSES,
          f"Labels < {NUM_CLASSES} (max={sample_labels.max()})")

# ── 3. TIMESTAMPS ──────────────────────────────────────────────────
def verify_timestamps(root):
    print("\n── 3. TIMESTAMPS ──")
    from dataset import load_timestamps
    csvs = sorted(glob.glob(os.path.join(root, "timestamp", "train", "*.csv")))
    check(len(csvs) > 0, f"Found {len(csvs)} timestamp CSVs")
    ts  = load_timestamps(csvs[0])
    seq = os.path.splitext(os.path.basename(csvs[0]))[0]
    n_pngs = len(glob.glob(os.path.join(root, "train", seq, "*.png")))
    check(len(ts) > 0, f"{seq}.csv: {len(ts)} timestamps, {n_pngs} PNGs")
    check(abs(len(ts) - n_pngs) <= 2, f"Timestamp count ~= PNG count ({len(ts)} vs {n_pngs})")
    print(f"  First 3 timestamps: {ts[:3]} us")

# ── 4. DATASET SAMPLES ─────────────────────────────────────────────
def verify_dataset(root):
    print("\n── 4. DATASET SAMPLES ──")
    from dataset import PEODDataset
    ds = PEODDataset(root, img_size=IMG_SIZE, patch_size=PATCH_SIZE, split="train")
    check(len(ds) > 0, f"Train dataset: {len(ds):,} samples")
    for i in [0, len(ds)//2, len(ds)-1]:
        s = ds[i]
        check(s["image"].shape   == (3, IMG_SIZE, IMG_SIZE),
              f"[{i}] image shape {tuple(s['image'].shape)}")
        check(s["density"].shape == (NUM_PATCHES,),
              f"[{i}] density shape {tuple(s['density'].shape)}")
        check(s["density"].min() >= 0,
              f"[{i}] density >= 0")
        check(s["boxes"].ndim == 2 and s["boxes"].shape[1] == 4,
              f"[{i}] boxes shape {tuple(s['boxes'].shape)}")
        nonzero = (s["density"] > 0).float().mean().item() * 100
        print(f"  [{i}] density nonzero={nonzero:.1f}%  "
              f"condition={s['condition']}  n_boxes={s['boxes'].shape[0]}")
    return ds

# ── 5. DATALOADER ──────────────────────────────────────────────────
def verify_dataloader(ds):
    print("\n── 5. DATALOADER BATCH ──")
    from dataset import collate_fn
    dl = DataLoader(ds, batch_size=4, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)
    batch = next(iter(dl))
    check(batch["image"].shape   == (4, 3, IMG_SIZE, IMG_SIZE),
          f"image   {tuple(batch['image'].shape)}")
    check(batch["density"].shape == (4, NUM_PATCHES),
          f"density {tuple(batch['density'].shape)}")
    check(batch["boxes"].ndim    == 3,
          f"boxes   {tuple(batch['boxes'].shape)}")
    print(f"  Conditions: {batch['condition']}")

# ── 6+7. FORWARD + BACKWARD ────────────────────────────────────────
def verify_model(root):
    print("\n── 6+7. FORWARD + BACKWARD ──")
    from dataset import PEODDataset, collate_fn
    from model import build_model, forward_pass
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    ds = PEODDataset(root, IMG_SIZE, PATCH_SIZE, split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(dl))
    images  = batch["image"].to(device)
    density = batch["density"].to(device)

    vit, head = build_model(num_classes=NUM_CLASSES, pretrained=True)
    vit, head = vit.to(device), head.to(device)

    # Trainable param audit
    trainable = [(n,p) for n,p in
                 list(vit.named_parameters()) + list(head.named_parameters())
                 if p.requires_grad]
    alpha_params = [n for n,_ in trainable if "alpha" in n]
    head_params  = [n for n,_ in trainable if "alpha" not in n]
    total = sum(p.numel() for _,p in trainable)
    check(len(alpha_params) == 4,
          f"Trainable alpha params: {len(alpha_params)} (expected 4)")
    check(len(head_params)  > 0,
          f"Trainable head params:  {len(head_params)}")
    print(f"  Total trainable params: {total:,}")

    # Forward
    preds = forward_pass(vit, head, images, density)
    expected = (4, NUM_PATCHES, NUM_CLASSES + 4)
    check(tuple(preds.shape) == expected,
          f"Predictions shape {tuple(preds.shape)} == {expected}")

    # Loss
    dummy_obj = torch.zeros(4, NUM_PATCHES, device=device)
    loss = F.binary_cross_entropy_with_logits(preds[..., 0], dummy_obj)
    check(torch.isfinite(loss), f"Loss is finite: {loss.item():.4f}")

    # Backward
    loss.backward()
    for n, p in trainable:
        check(p.grad is not None, f"Gradient exists for {n}")
    frozen_with_grad = [n for n,p in
                        list(vit.named_parameters()) + list(head.named_parameters())
                        if not p.requires_grad and p.grad is not None]
    check(len(frozen_with_grad) == 0,
          f"Frozen params have no gradients ({len(frozen_with_grad)} violations)")

    print(f"  Alpha init values (all must be 0.0):")
    for i, block in enumerate(vit.blocks[-4:]):
        print(f"    block-{4-i}: alpha = {block.attn.alpha.item():.6f}")

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    root = os.path.expanduser(args.data_root)
    print(f"\nPEOD verification  --  data_root: {root}")
    verify_dat(root)
    verify_coco(root)
    verify_timestamps(root)
    ds = verify_dataset(root)
    verify_dataloader(ds)
    verify_model(root)
    print(f"\n{'='*55}")
    print("All checks passed -- safe to start training.")
    print(f"\n  python project/train.py \\")
    print(f"    --data_root {root} \\")
    print(f"    --epochs 20 --batch_size 16 --num_workers 4")

if __name__ == "__main__":
    main()
