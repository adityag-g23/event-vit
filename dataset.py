"""PEODDataset — actual layout on disk:

  dataset/
    train/
      sequence_001/             ← PNG frames
        sequence_001_0001.png
        ...
      sequence_001.dat          ← binary event stream (alongside the folder)
      sequence_002/
      sequence_002.dat
      ...
    test/
      normal/
        sequence_XXX/
        sequence_XXX.dat
        ...
      challenge/
        ...
    coco_0111_last/
      train/
        normal/   sequence_XXX.json   ← COCO JSON, one per sequence
        motion_blur/
        challenge/
      test/
        normal/
        challenge/
    timestamp/
      train/  sequence_XXX.csv   ← rows: (flag, timestamp_us) at 30 Hz
      test/

COCO bbox format: [x, y, w, h] in pixels (origin top-left).
Condition label comes from which annotation subfolder the JSON lives in.
"""

import os
import json
import glob
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from event_utils import batch_events_to_patch_density

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]

SENSOR_H, SENSOR_W = 720, 1280
CLASSES    = ("car", "person", "bus", "truck", "2-wheeler", "3-wheeler")
NUM_CLASSES = len(CLASSES)
CONDITIONS  = ("normal", "motion_blur", "challenge")


# ---------------------------------------------------------------------------
# Event loader  (Prophesee RAW .dat format — update if verify.py shows otherwise)
# ---------------------------------------------------------------------------

def load_dat_events(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Prophesee-style binary .dat file.

    Returns:
        x: (N,) uint16   pixel x-coord
        y: (N,) uint16   pixel y-coord
        t: (N,) uint32   timestamp in microseconds
    """
    empty = np.zeros(0, np.uint16), np.zeros(0, np.uint16), np.zeros(0, np.uint32)
    if not os.path.exists(path):
        return empty
    try:
        with open(path, "rb") as f:
            # Skip ASCII header lines (begin with b'%')
            header_bytes = 0
            while True:
                line = f.readline()
                if not line or not line.startswith(b"%"):
                    break
                header_bytes += len(line)
            f.seek(header_bytes)
            raw = f.read()

        if len(raw) == 0:
            return empty

        # Prophesee EVT2 compact format: 8 bytes per event
        #   word0 (u32): timestamp in µs
        #   word1 (u32): encoded address — bits[0:10]=x, bits[11:20]=y, bit[21]=p
        dtype = np.dtype([("t", "<u4"), ("addr", "<u4")])
        if len(raw) % 8 != 0:
            raw = raw[: (len(raw) // 8) * 8]   # trim incomplete last record
        data = np.frombuffer(raw, dtype=dtype)

        t = data["t"]
        addr = data["addr"]
        x = (addr & 0x7FF).astype(np.uint16)
        y = ((addr >> 11) & 0x3FF).astype(np.uint16)
        return x, y, t

    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Timestamp loader
# ---------------------------------------------------------------------------

def load_timestamps(csv_path: str) -> np.ndarray:
    """Return array of frame timestamps (µs) from a two-column CSV (flag, ts)."""
    if not os.path.exists(csv_path):
        return np.array([], dtype=np.float64)
    ts = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                ts.append(float(row[1]))
    return np.array(ts, dtype=np.float64)


# ---------------------------------------------------------------------------
# COCO annotation loader
# ---------------------------------------------------------------------------

def load_coco_json(json_path: str) -> dict[str, dict]:
    """Parse a COCO JSON and return a dict keyed by image file_name.

    Each value: {"boxes": (N,4) float32 normalized xyxy, "labels": (N,) int64}
    """
    with open(json_path) as f:
        data = json.load(f)

    id_to_fname: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    fname_anns: dict[str, dict] = {}

    for ann in data["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        x, y, w, h = ann["bbox"]          # COCO: xywh pixels
        box = [x / SENSOR_W,              # normalize to [0,1]
               y / SENSOR_H,
               (x + w) / SENSOR_W,
               (y + h) / SENSOR_H]
        if fname not in fname_anns:
            fname_anns[fname] = {"boxes": [], "labels": []}
        fname_anns[fname]["boxes"].append(box)
        fname_anns[fname]["labels"].append(int(ann["category_id"]))

    # Convert to arrays
    for v in fname_anns.values():
        v["boxes"]  = np.array(v["boxes"],  dtype=np.float32)
        v["labels"] = np.array(v["labels"], dtype=np.int64)

    return fname_anns


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PEODDataset(Dataset):
    """Pixel-aligned Event-RGB Object Detection dataset.

    Args:
        root:       path to the top-level dataset folder
                    (contains train/, test/, coco_0111_last/, timestamp/)
        img_size:   square resize target for ViT (224)
        patch_size: ViT patch size (16)
        split:      'train' | 'test_normal' | 'test_challenge'
    """

    _SPLITS = {
        "train":          ("train",        ["normal", "motion_blur", "challenge"], "train"),
        "test_normal":    ("test/normal",  ["normal"],                             "test"),
        "test_challenge": ("test/challenge",["challenge"],                         "test"),
    }

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        patch_size: int = 16,
        split: str = "train",
    ) -> None:
        assert split in self._SPLITS, f"split must be one of {list(self._SPLITS)}"
        self.img_size   = img_size
        self.patch_size = patch_size
        self.transform  = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(_IMG_MEAN, _IMG_STD),
        ])

        rgb_subdir, ann_conditions, ts_subdir = self._SPLITS[split]
        rgb_root = os.path.join(root, rgb_subdir)
        ts_root  = os.path.join(root, "timestamp", ts_subdir)

        # Discover all annotation JSONs for this split across all condition folders
        json_paths: list[tuple[str, str]] = []  # (json_path, condition)
        for cond in ann_conditions:
            pattern = os.path.join(root, "coco_0111_last", ts_subdir, cond, "*.json")
            for p in sorted(glob.glob(pattern)):
                json_paths.append((p, cond))

        if not json_paths:
            raise FileNotFoundError(
                f"No annotation JSONs found for split='{split}' under {root}/coco_0111_last/"
            )

        # Build flat sample list: one entry per annotated frame
        self.samples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = []
        # (png_path, events_x, events_y, boxes_xyxy_norm, labels, condition) —
        # but events are per-sequence so we store references

        # Temporary per-sequence cache so we don't reload events for every frame
        _ev_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for json_path, cond in json_paths:
            seq = os.path.splitext(os.path.basename(json_path))[0]  # e.g. "sequence_066"
            fname_anns = load_coco_json(json_path)

            # Events: alongside the RGB folder
            dat_path = os.path.join(rgb_root, seq + ".dat")
            if seq not in _ev_cache:
                _ev_cache[seq] = load_dat_events(dat_path)
            ev_x, ev_y, ev_t = _ev_cache[seq]

            # Timestamps for temporal slicing (optional — used if events are available)
            ts_path = os.path.join(ts_root, seq + ".csv")
            timestamps = load_timestamps(ts_path)

            for fname, anns in sorted(fname_anns.items()):
                png_path = os.path.join(rgb_root, seq, fname)
                if not os.path.exists(png_path):
                    continue

                # Determine which events belong to this frame via timestamps
                # filename: sequence_066_0001.png → frame index 1 (1-based)
                try:
                    frame_idx = int(fname.rsplit("_", 1)[-1].split(".")[0]) - 1
                except ValueError:
                    frame_idx = -1

                # Temporal slice: events in [t_prev, t_cur)
                if len(timestamps) > 0 and len(ev_t) > 0 and frame_idx >= 0:
                    t_lo = timestamps[frame_idx - 1] if frame_idx > 0 else 0.0
                    t_hi = timestamps[frame_idx] if frame_idx < len(timestamps) else timestamps[-1]
                    mask = (ev_t >= t_lo) & (ev_t < t_hi)
                    fx, fy = ev_x[mask], ev_y[mask]
                else:
                    # Fallback: use all events (coarser but still informative)
                    fx, fy = ev_x, ev_y

                self.samples.append((
                    png_path,
                    fx.astype(np.float32),
                    fy.astype(np.float32),
                    anns["boxes"],
                    anns["labels"],
                    cond,
                ))

        if not self.samples:
            raise RuntimeError(
                f"No valid samples (PNG + annotation) found for split='{split}'. "
                f"Check that rgb_root={rgb_root} contains sequence folders."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        png_path, ev_x, ev_y, boxes, labels, condition = self.samples[idx]

        image = self.transform(Image.open(png_path).convert("RGB"))  # (3,H,W)

        # Scale event coords from sensor resolution (1280×720) → img_size×img_size
        # so the density map aligns with the resized RGB image patches.
        sx = self.img_size / SENSOR_W
        sy = self.img_size / SENSOR_H
        density = batch_events_to_patch_density(
            [{"x": ev_x * sx, "y": ev_y * sy}], self.img_size, self.img_size, self.patch_size
        ).squeeze(0)   # (num_patches,)

        return {
            "image":     image,
            "density":   density,
            "boxes":     torch.from_numpy(boxes),
            "labels":    torch.from_numpy(labels),
            "condition": condition,
        }


def collate_fn(batch: list[dict]) -> dict:
    images     = torch.stack([b["image"]   for b in batch])
    densities  = torch.stack([b["density"] for b in batch])
    conditions = [b["condition"] for b in batch]
    max_boxes  = max(b["boxes"].shape[0] for b in batch) if batch else 1
    padded_boxes  = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["boxes"].shape[0]
        padded_boxes[i,  :n] = b["boxes"]
        padded_labels[i, :n] = b["labels"]
    return {
        "image":     images,
        "density":   densities,
        "boxes":     padded_boxes,
        "labels":    padded_labels,
        "condition": conditions,
    }
