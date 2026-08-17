"""Compact quality-control plots for notebook use."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
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
    panels = _segmentation_panels(reference, masks, signal, signal_spec)
    _draw_segmentation_panels(figure, axes.flat, panels, title)
    return figure


def _draw_segmentation_panels(figure, axes, panels, title: str | None) -> None:
    """Draw prepared segmentation panels onto a figure's axes."""
    for axis, (_, image, cmap, panel_title) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(panel_title)
        axis.axis("off")

    if title:
        figure.suptitle(title)
    figure.tight_layout()


def save_segmentation_views(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    output_dir: str | Path,
    name: str,
    signal_spec: SignalChannel | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> list[Path]:
    """Save the four-panel grid and each constituent panel as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    panels = _segmentation_panels(reference, masks, signal, signal_spec)
    grid = Figure(figsize=(10, 10))
    FigureCanvasAgg(grid)
    _draw_segmentation_panels(grid, grid.subplots(2, 2).flat, panels, title)
    grid_path = output_dir / f"{name}__grid.png"
    grid.savefig(grid_path, dpi=dpi, bbox_inches="tight")
    saved.append(grid_path)

    for panel_name, image, cmap, panel_title in panels:
        figure = Figure(figsize=(6, 6))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(panel_title)
        axis.axis("off")
        if title:
            figure.suptitle(title)
        figure.tight_layout()
        destination = output_dir / f"{name}__{panel_name}.png"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        saved.append(destination)
    return saved


def _segmentation_panels(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None,
    signal_spec: SignalChannel | None,
) -> list[tuple[str, np.ndarray, str | None, str]]:
    """Build the images, color maps, and labels used by the QC grid."""
    normalized_reference = _scale(reference)

    if signal is not None:
        normalized_signal = _scale(signal)
        signal_title = "Signal (1st–99th percentile)"
    else:
        normalized_signal = np.zeros_like(normalized_reference)
        signal_title = "Signal (not provided)"

    mask_overlay = np.stack([normalized_reference] * 3, axis=-1)
    mask_overlay[find_boundaries(masks)] = (1, 0.1, 0.1)

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
    return [
        (
            "reference",
            normalized_reference,
            "gray",
            "Reference (1st–99th percentile)",
        ),
        ("signal", normalized_signal, "magma", signal_title),
        (
            "masks",
            mask_overlay,
            None,
            f"Masks (red boundaries, n={int(np.max(masks))})",
        ),
        (
            "overlay",
            channel_overlay,
            None,
            "Overlay (reference=green, signal=magenta)\n"
            f"masks=yellow, colocalized=cyan (n={positive_labels.size})",
        ),
    ]


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
