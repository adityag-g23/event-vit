"""Convert raw event camera data (x, y, t, p) to per-patch density scalars."""

import torch
import numpy as np


def events_to_density_map(
    events_x: torch.Tensor,
    events_y: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Bin events into a 2D spatial grid and apply log(1+count) normalization.

    Args:
        events_x: (N,) x-coordinates of events (int or float)
        events_y: (N,) y-coordinates of events
        height: image height in pixels
        width: image width in pixels

    Returns:
        density_map: (H, W) float tensor, log-normalized event counts
    """
    device = events_x.device if torch.is_tensor(events_x) else torch.device("cpu")
    flat = torch.zeros(height * width, dtype=torch.float32, device=device)

    if len(events_x) == 0:
        return flat.reshape(height, width)

    x = torch.as_tensor(events_x, dtype=torch.long, device=device).clamp(0, width - 1)
    y = torch.as_tensor(events_y, dtype=torch.long, device=device).clamp(0, height - 1)
    idx = y * width + x

    ones = torch.ones(idx.shape[0], dtype=torch.float32, device=device)
    flat.scatter_add_(0, idx, ones)
    return torch.log1p(flat).reshape(height, width)


def density_map_to_patch_density(
    density_map: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """Average-pool a 2D density map onto the ViT patch grid.

    Args:
        density_map: (H, W) float tensor
        patch_size: side length of each ViT patch in pixels

    Returns:
        patch_density: (num_patches,) one scalar per patch
    """
    H, W = density_map.shape
    pH, pW = H // patch_size, W // patch_size
    cropped = density_map[: pH * patch_size, : pW * patch_size]
    # (pH, patch_size, pW, patch_size) -> mean over spatial dims inside patch
    grid = cropped.reshape(pH, patch_size, pW, patch_size).mean(dim=(1, 3))
    return grid.reshape(-1)  # (num_patches,)


def batch_events_to_patch_density(
    batch_events: list[dict],
    height: int,
    width: int,
    patch_size: int = 16,
) -> torch.Tensor:
    """Process a list of per-sample event dicts into a batched density tensor.

    Args:
        batch_events: list of dicts, each with keys 'x' and 'y' (array-like)
        height, width: image dimensions in pixels
        patch_size: ViT patch size

    Returns:
        (B, num_patches) float tensor
    """
    densities = []
    for ev in batch_events:
        x = torch.as_tensor(ev["x"], dtype=torch.float32)
        y = torch.as_tensor(ev["y"], dtype=torch.float32)
        dm = events_to_density_map(x, y, height, width)
        densities.append(density_map_to_patch_density(dm, patch_size))
    return torch.stack(densities, dim=0)
