"""Non-overlapping tile generation and optional mask stitching."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def generate_tiles(
    image: np.ndarray,
    tile_size: tuple[int, int],
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield non-overlapping tiles that together cover the full CYX image.

    Tiles are produced in row-major (top-to-bottom, left-to-right) order.
    When an image dimension is not evenly divisible by the tile size the
    rightmost or bottom-most tiles are zero-padded to exactly ``tile_size``.

    Args:
        image: CYX array to tile.  The channel axis (C) is preserved in every
            tile.
        tile_size: ``(tile_height, tile_width)`` in pixels.

    Yields:
        Three-tuples of ``(tile_cyx, y0, x0)`` where ``tile_cyx`` is the
        cropped and zero-padded tile array, ``y0`` is the top-left row index in
        the original image, and ``x0`` is the top-left column index.
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

    Cell IDs are made globally unique by offsetting each tile's labels by the
    maximum label seen in all preceding tiles.  Zero-padded edge regions that
    extend beyond the original image boundary are clipped away.

    Args:
        image_shape: ``(height, width)`` of the original (un-padded) image.
        tile_results: List of ``(y0, x0, masks)`` tuples, one per tile, in any
            order.  ``y0`` and ``x0`` are the tile origins and ``masks`` is the
            integer label array produced by segmentation for that tile.

    Returns:
        Integer label array of shape ``image_shape``.  Every cell has a unique
        ID across the entire stitched image.
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
