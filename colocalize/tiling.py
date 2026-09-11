"""Non-overlapping tile generation and optional mask stitching."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def generate_tiles(
    image: np.ndarray,
    tile_size: tuple[int, int],
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield (tile_cyx, y0, x0) for non-overlapping tiles covering the full CYX image.

    Edge tiles are zero-padded to tile_size when the image is not evenly divisible.
    """
    _, H, W = image.shape
    th, tw = tile_size
    n_rows = math.ceil(H / th)
    n_cols = math.ceil(W / tw)

    for row in range(n_rows):
        for col in range(n_cols):
            y0, x0 = row * th, col * tw
            tile = image[:, y0 : y0 + th, x0 : x0 + tw]
            if tile.shape[1] < th or tile.shape[2] < tw:
                padded = np.zeros((image.shape[0], th, tw), dtype=image.dtype)
                padded[:, : tile.shape[1], : tile.shape[2]] = tile
                tile = padded
            yield tile, y0, x0


def stitch_masks(
    image_shape: tuple[int, int],
    tile_results: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Combine per-tile mask arrays into a single full-image label array.

    Labels are offset per tile so every cell retains a unique ID across the stitched image.
    Padded zero regions beyond the original image boundary are clipped.
    """
    stitched = np.zeros(image_shape, dtype=np.int32)
    label_offset = 0
    for y0, x0, tile_masks in tile_results:
        H, W = image_shape
        y1 = min(y0 + tile_masks.shape[0], H)
        x1 = min(x0 + tile_masks.shape[1], W)
        region = tile_masks[: y1 - y0, : x1 - x0]
        stitched[y0:y1, x0:x1] = np.where(region > 0, region + label_offset, 0)
        if region.size:
            label_offset += int(region.max())
    return stitched
