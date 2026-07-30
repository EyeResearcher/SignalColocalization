"""Compact quality-control plots for notebook use."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


def show_segmentation(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    title: str | None = None,
):
    """Display reference, signal, masks, and a false-color channel overlay."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    normalized_reference = _scale(reference)
    axes[0, 0].imshow(normalized_reference, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Reference (1st–99th percentile)")

    if signal is not None:
        normalized_signal = _scale(signal)
        axes[0, 1].imshow(normalized_signal, cmap="magma", vmin=0, vmax=1)
        axes[0, 1].set_title("Signal (1st–99th percentile)")
    else:
        normalized_signal = np.zeros_like(normalized_reference)
        axes[0, 1].set_title("Signal (not provided)")

    mask_overlay = np.stack([normalized_reference] * 3, axis=-1)
    mask_overlay[find_boundaries(masks)] = (1, 0.1, 0.1)
    axes[1, 0].imshow(mask_overlay)
    axes[1, 0].set_title(f"Masks (red boundaries, n={int(np.max(masks))})")

    channel_overlay = np.zeros((*normalized_reference.shape, 3), dtype=float)
    channel_overlay[..., 0] = normalized_signal
    channel_overlay[..., 1] = normalized_reference
    channel_overlay[..., 2] = normalized_signal
    channel_overlay[find_boundaries(masks)] = (1, 1, 0)
    axes[1, 1].imshow(channel_overlay)
    axes[1, 1].set_title(
        "Overlay (reference=green, signal=magenta, masks=yellow)"
    )

    for axis in axes.flat:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def _scale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)
