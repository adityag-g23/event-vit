"""PEODDataset: loads synchronized RGB frames, raw events, and box annotations from HDF5.

Assumed HDF5 layout (one file per sequence):
  /images/frame_{i}          uint8 (H, W, 3)
  /events/frame_{i}/x        float32 array
  /events/frame_{i}/y        float32 array
  /events/frame_{i}/p        int8 array  (polarity, unused in density)
  /labels/frame_{i}/boxes    float32 (N, 4)  normalized [x1,y1,x2,y2]
  /labels/frame_{i}/labels   int32   (N,)
  /metadata/condition        scalar string or bytes per frame, or single attr
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from event_utils import batch_events_to_patch_density

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]

_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(_IMG_MEAN, _IMG_STD),
])

CONDITIONS = ("normal", "low_light", "motion_blur")


class PEODDataset(Dataset):
    """Pixel-aligned Event-RGB Object Detection dataset.

    Args:
        root: directory containing *.h5 sequence files
        img_size: target image size (square); images are resized if needed
        patch_size: ViT patch size for density pooling
        split: 'train' or 'val'
    """

    def __init__(self, root: str, img_size: int = 224, patch_size: int = 16, split: str = "train"):
        self.img_size = img_size
        self.patch_size = patch_size
        self.resize = T.Resize((img_size, img_size))

        pattern = os.path.join(root, split, "*.h5")
        h5_files = sorted(glob.glob(pattern))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found at {pattern}")

        # Build flat index: (filepath, frame_index)
        self.samples: list[tuple[str, int]] = []
        for path in h5_files:
            with h5py.File(path, "r") as f:
                n = len(f["images"].keys())
                self.samples.extend((path, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, frame_i = self.samples[idx]
        key = f"frame_{frame_i}"

        with h5py.File(path, "r") as f:
            # --- RGB ---
            rgb = f["images"][key][:]          # (H, W, 3) uint8
            from PIL import Image
            img = Image.fromarray(rgb)
            img = self.resize(img)
            image = _normalize(img)            # (3, H, W) float32

            # --- Events ---
            ev_grp = f["events"][key]
            events = {
                "x": ev_grp["x"][:].astype(np.float32),
                "y": ev_grp["y"][:].astype(np.float32),
            }

            # --- Boxes & labels ---
            lbl_grp = f["labels"][key]
            boxes  = torch.from_numpy(lbl_grp["boxes"][:].astype(np.float32))   # (N,4)
            labels = torch.from_numpy(lbl_grp["labels"][:].astype(np.int64))    # (N,)

            # --- Condition ---
            condition = "normal"
            if "metadata" in f:
                md = f["metadata"]
                raw = md["condition"][frame_i] if "condition" in md else b"normal"
                condition = raw.decode() if isinstance(raw, bytes) else str(raw)

        # Compute patch density in this call (single-sample)
        density = batch_events_to_patch_density(
            [events], self.img_size, self.img_size, self.patch_size
        ).squeeze(0)  # (num_patches,)

        return {
            "image":     image,          # (3, H, W)
            "density":   density,        # (num_patches,)
            "boxes":     boxes,          # (N, 4) normalized xyxy
            "labels":    labels,         # (N,)
            "condition": condition,      # str
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that pads variable-length box lists."""
    images    = torch.stack([b["image"]   for b in batch])
    densities = torch.stack([b["density"] for b in batch])
    conditions = [b["condition"] for b in batch]

    max_boxes = max(b["boxes"].shape[0] for b in batch)
    padded_boxes  = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["boxes"].shape[0]
        padded_boxes[i, :n]  = b["boxes"]
        padded_labels[i, :n] = b["labels"]

    return {
        "image":     images,
        "density":   densities,
        "boxes":     padded_boxes,
        "labels":    padded_labels,
        "condition": conditions,
    }
