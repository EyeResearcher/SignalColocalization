"""Compact quality-control plots for notebook use."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from .colocalize import signal_threshold
from .datasets import SignalChannel


def show_segmentation(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    signal_spec: SignalChannel | None = None,
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
    mask_boundaries = find_boundaries(masks)
    channel_overlay[mask_boundaries] = (1, 1, 0)

    positive_labels = _positive_mask_labels(signal, masks, signal_spec)
    if positive_labels.size:
        positive_masks = np.where(np.isin(masks, positive_labels), masks, 0)
        positive_boundaries = find_boundaries(positive_masks)
        channel_overlay[positive_boundaries] = (0, 1, 1)
    axes[1, 1].imshow(channel_overlay)
    axes[1, 1].set_title(
        "Overlay (reference=green, signal=magenta)\n"
        f"masks=yellow, colocalized=cyan (n={positive_labels.size})"
    )

    for axis in axes.flat:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def _positive_mask_labels(
    signal: np.ndarray | None,
    masks: np.ndarray,
    signal_spec: SignalChannel | None,
) -> np.ndarray:
    """Return labels whose positive-signal fraction meets the configured cutoff."""
    if signal is None or signal_spec is None:
        return np.array([], dtype=np.asarray(masks).dtype)

    values = np.asarray(signal, dtype=float)
    labels = np.asarray(masks)
    threshold = signal_threshold(values, signal_spec)
    positive = values > threshold
    return np.asarray(
        [
            label
            for label in np.unique(labels)
            if label != 0
            and np.mean(positive[labels == label])
            >= signal_spec.positive_fraction_cutoff
        ],
        dtype=labels.dtype,
    )


def _scale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)
