"""PEODDataset: loads synchronized RGB frames, raw events, and annotations.

Actual PEOD layout on disk:
  rgb/
    train/  sequence_001_0001.png  sequence_001_0002.png ...
    test/
      normal/
      challenge/
  event/
    train/  sequence_001.dat  ...       (binary event stream per sequence)
    test/
      normal/
      challenge/
  timestamp/
    train/  sequence_001.npy  ...       (frame timestamps, shape (N,))
  annotations/
    train/  sequence_001.npy  ...       (shape (N_boxes, 6): frame_id x1 y1 x2 y2 cls)

Resolution: 1280×720.  Resized to img_size×img_size for ViT input.
Conditions: 'normal' (test/normal/) or 'challenge' (test/challenge/).
            All train samples labelled 'normal'.

NOTE: load_dat_events() implements Prophesee RAW .dat format (most likely).
      If inspect_data.py reveals a different binary layout, only that function
      needs to change — nothing else in this file.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from event_utils import batch_events_to_patch_density

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]

# 6 PEOD classes (index 0-5)
CLASSES = ("car", "bus", "truck", "two-wheeler", "three-wheeler", "person")
CONDITIONS = ("normal", "challenge")

# Native sensor resolution
_SENSOR_H, _SENSOR_W = 720, 1280


# ---------------------------------------------------------------------------
# Event file reader — update dtype/header here once inspect_data.py confirms
# ---------------------------------------------------------------------------

# Prophesee RAW .dat dtype: each event = 8 bytes (t u32, addr u32 encoding x,y,p)
# Fallback plain dtype if header-less: (t u32, x u16, y u16) with polarity in MSB
_EVT_DTYPE = np.dtype([("t", "<u4"), ("_addr", "<u4")])


def load_dat_events(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Prophesee-style .dat event file.

    Returns:
        x: (N,) uint16 x-coordinates
        y: (N,) uint16 y-coordinates
    """
    with open(path, "rb") as f:
        # Skip ASCII header lines beginning with '%'
        header_bytes = 0
        while True:
            line = f.readline()
            if not line.startswith(b"%"):
                break
            header_bytes += len(line)
        f.seek(header_bytes)
        data = np.frombuffer(f.read(), dtype=_EVT_DTYPE)

    if len(data) == 0:
        return np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16)

    # Decode x, y from packed address word: bits[0:10]=x, bits[11:21]=y
    addr = data["_addr"]
    x = (addr & 0x7FF).astype(np.uint16)
    y = ((addr >> 11) & 0x7FF).astype(np.uint16)
    return x, y


# ---------------------------------------------------------------------------
# Annotation reader
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-sequence annotation .npy file.

    Expected shape: (N_boxes, 6) — [frame_id, x1, y1, x2, y2, class_id]
    x1..y2 are pixel coordinates in the original 1280×720 resolution.

    Returns:
        frame_ids: (N,) int
        boxes:     (N, 4) float32 normalized [x1,y1,x2,y2] in [0,1]
        labels:    (N,)  int
    """
    ann = np.load(path, allow_pickle=True)
    if ann.ndim == 1:          # allow_pickle may return object array
        ann = np.stack(ann)
    frame_ids = ann[:, 0].astype(np.int64)
    boxes_px  = ann[:, 1:5].astype(np.float32)
    labels    = ann[:, 5].astype(np.int64)
    # Normalize to [0,1]
    boxes_px[:, [0, 2]] /= _SENSOR_W
    boxes_px[:, [1, 3]] /= _SENSOR_H
    return frame_ids, boxes_px, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PEODDataset(Dataset):
    """Pixel-aligned Event-RGB Object Detection dataset.

    Args:
        root:       path to the top-level PEOD folder
        img_size:   square size to resize images to (224 for ViT-B/16)
        patch_size: ViT patch size (16)
        split:      'train', 'test_normal', or 'test_challenge'
    """

    _SPLIT_DIRS = {
        "train":          ("rgb/train",         "event/train",         "annotations/train",   "timestamp/train"),
        "test_normal":    ("rgb/test/normal",    "event/test/normal",   "annotations/test/normal",   "timestamp/test/normal"),
        "test_challenge": ("rgb/test/challenge", "event/test/challenge","annotations/test/challenge","timestamp/test/challenge"),
    }

    def __init__(self, root: str, img_size: int = 224, patch_size: int = 16, split: str = "train"):
        assert split in self._SPLIT_DIRS, f"split must be one of {list(self._SPLIT_DIRS)}"
        self.img_size   = img_size
        self.patch_size = patch_size
        self.condition  = "normal" if "normal" in split or split == "train" else "challenge"
        self.transform  = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(_IMG_MEAN, _IMG_STD),
        ])

        rgb_dir, ev_dir, ann_dir, ts_dir = [
            os.path.join(root, d) for d in self._SPLIT_DIRS[split]
        ]

        # Discover sequences by annotation files (one .npy per sequence)
        ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.npy")))
        if not ann_files:
            raise FileNotFoundError(f"No annotation .npy files found in {ann_dir}")

        # Build flat sample index: (rgb_path, ev_path, boxes_for_frame, labels_for_frame)
        self.samples: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for ann_path in ann_files:
            seq = os.path.splitext(os.path.basename(ann_path))[0]  # e.g. 'sequence_001'
            ev_path = os.path.join(ev_dir, seq + ".dat")
            if not os.path.exists(ev_path):
                continue

            frame_ids, boxes, labels = load_annotations(ann_path)
            # Pre-load all events for this sequence (fast; events are small)
            ex, ey = load_dat_events(ev_path)

            # Get sorted unique frame ids that have PNG files
            unique_frames = sorted(set(frame_ids.tolist()))
            for fid in unique_frames:
                png = os.path.join(rgb_dir, f"{seq}_{fid:04d}.png")
                if not os.path.exists(png):
                    continue
                mask = frame_ids == fid
                self.samples.append((png, (ex, ey), boxes[mask], labels[mask]))

        if not self.samples:
            raise RuntimeError(f"No valid (PNG, DAT, annotation) triples found for split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        png_path, (ex, ey), boxes, labels = self.samples[idx]

        image = self.transform(Image.open(png_path).convert("RGB"))  # (3, H, W)

        density = batch_events_to_patch_density(
            [{"x": ex, "y": ey}], self.img_size, self.img_size, self.patch_size
        ).squeeze(0)  # (num_patches,)

        return {
            "image":     image,
            "density":   density,
            "boxes":     torch.from_numpy(boxes),
            "labels":    torch.from_numpy(labels),
            "condition": self.condition,
        }


def collate_fn(batch: list[dict]) -> dict:
    images    = torch.stack([b["image"]   for b in batch])
    densities = torch.stack([b["density"] for b in batch])
    conditions = [b["condition"] for b in batch]
    max_boxes = max(b["boxes"].shape[0] for b in batch) if batch else 1
    padded_boxes  = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["boxes"].shape[0]
        padded_boxes[i, :n]  = b["boxes"]
        padded_labels[i, :n] = b["labels"]
    return {"image": images, "density": densities, "boxes": padded_boxes,
            "labels": padded_labels, "condition": conditions}
