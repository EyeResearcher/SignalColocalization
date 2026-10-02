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
from .distributions import _cutoffs, _signal_values

import pandas as pd
def show_segmentation(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    signal_spec: SignalChannel | None = None,
    title: str | None = None,
    z_index: int | None = None,
):
    """Display reference, signal, masks, and a false-color channel overlay."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = _segmentation_panels(reference,
                                  masks,
                                  signal,
                                  signal_spec,
                                  z_index=z_index,
                                )
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


def make_segmentation_views(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    signal_spec: SignalChannel | None = None,
    title: str | None = None,
    z_index: int | None = None,
) -> dict[str, Figure]:
    """Build the grid and individual panel figures without saving."""
    panels = _segmentation_panels(reference, masks, signal, signal_spec, z_index=z_index)
    figures: dict[str, Figure] = {}

    grid = Figure(figsize=(10, 10))
    FigureCanvasAgg(grid)
    _draw_segmentation_panels(grid, grid.subplots(2, 2).flat, panels, title)
    figures["grid"] = grid

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
        figures[panel_name] = figure

    return figures
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def _segmentation_panels(
    reference: np.ndarray,
    masks: np.ndarray,
    signal: np.ndarray | None,
    signal_spec: SignalChannel | None,
    *,
    z_index: int | None = None,
) -> list[tuple[str, np.ndarray, str | None, str]]:
    """Build the images, color maps, and labels used by the QC grid."""
    reference = np.asarray(reference)
    masks = np.asarray(masks)
    signal = None if signal is None else np.asarray(signal)

    if reference.ndim not in (2, 3):
        raise ValueError("Expected a YX image or ZYX volume.")

    if masks.shape != reference.shape:
        raise ValueError("Reference and masks must have matching shapes.")

    if signal is not None and signal.shape != reference.shape:
        raise ValueError("Signal must match the reference shape.")

    # Classify cells using every voxel, before selecting a display plane.
    positive_labels = _positive_mask_labels(signal, masks, signal_spec)

    slice_label = ""
    if reference.ndim == 3:
        depth = reference.shape[0]
        if z_index is None:
            z_index = depth // 2

        if not isinstance(z_index, (int, np.integer)):
            raise TypeError("z_index must be an integer.")
        if not 0 <= z_index < depth:
            raise ValueError(
                f"z_index must be between 0 and {depth - 1}."
            )

        slice_label = f" | z={z_index} (zero-based)"
        reference = reference[z_index]
        masks = masks[z_index]
        signal = None if signal is None else signal[z_index]

    elif z_index is not None:
        raise ValueError("z_index only applies to 3D input.")

    # Only count positive cells that appear on the displayed plane.
    visible_labels = np.unique(masks)
    visible_labels = visible_labels[visible_labels != 0]
    visible_positive_labels = np.intersect1d(
        positive_labels, visible_labels
    )
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

    
    if visible_positive_labels.size:
        positive_masks = np.where(
            np.isin(masks, visible_positive_labels), masks, 0
        )
        positive_boundaries = find_boundaries(positive_masks)
        channel_overlay[positive_boundaries] = (0, 1, 1)
    panels = [
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
           f"Masks (red boundaries, visible n={visible_labels.size})",
        ),
        (
            "overlay",
            channel_overlay,
            None,
            "Overlay (reference=green, signal=magenta)\n"
            f"masks=yellow, positive=cyan "
            f"(visible n={visible_positive_labels.size})"
        ),
    ]
    return [(name, image, cmap, panel_title + slice_label)
         for name, image, cmap, panel_title in panels]


def _positive_mask_labels(
    signal: np.ndarray | None,
    masks: np.ndarray,
    signal_spec: SignalChannel | None,
) -> np.ndarray:
    """Return labels whose positive-signal fraction meets the configured cutoff."""
    if signal is None or signal_spec is None:
        return np.array([], dtype=np.asarray(masks).dtype)

    from scipy import ndimage as ndi  # pylint: disable=import-outside-toplevel

    values = np.asarray(signal, dtype=float)
    labels = np.asarray(masks)
    threshold = signal_threshold(values, signal_spec)
    positive = (values > threshold).astype(np.float32)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    if unique_labels.size == 0:
        return np.array([], dtype=labels.dtype)
    # one vectorized pass over the image instead of one boolean mask per label
    mean_positive = ndi.mean(positive, labels, index=unique_labels)
    return unique_labels[
        np.asarray(mean_positive) >= signal_spec.positive_fraction_cutoff
    ]


def _scale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def plot_signal_ecdf(cells : pd.DataFrame, signal: str, *, metric: str = "corrected_median", group: str = "source", thresholds=(), normalize_by_noise: bool = False):
    """Plot ECDFs with optional cutoffs matching summarize_signal_ecdf.

    Noise normalization is available for raw-unit corrected mean/median.
    The fraction above a cutoff is 1 minus the CDF at that cutoff. Nonfinite
    values (and invalid sigma when normalizing) are excluded in both helpers.
    Filter to one reference set before plotting multiple images together.
    """
    column = f"signal_{signal}_{metric}"
    if column not in cells or group not in cells:
        raise KeyError(f"Expected columns {column!r} and {group!r}.")
    thresholds = _cutoffs(thresholds)
    _signal_values(cells, signal, metric, normalize_by_noise)
    series = [(label, np.sort(_signal_values(table, signal, metric, normalize_by_noise)))
              for label, table in cells.groupby(group, sort=False, dropna=False, observed=True)]
    bounds = np.concatenate([values for _, values in series] + [np.asarray(thresholds)])
    if bounds.size:
        low, high = bounds.min(), bounds.max()
        padding = max(1, high - low) * .05
        low, high = low - padding, high + padding
    figure, axis = plt.subplots()
    for label, values in series:
        if values.size:
            unique, counts = np.unique(values, return_counts=True)
            axis.step(np.r_[low, unique, high], np.r_[0, counts.cumsum() / values.size, 1], where="post", label=str(label))
            for threshold in thresholds:
                axis.plot(threshold, np.count_nonzero(values <= threshold) / values.size,
                          marker="o", color=axis.lines[-1].get_color(), linestyle="none")
    for threshold in thresholds:
        axis.axvline(threshold, color="gray", linestyle="--", linewidth=.8,
                     label=f"Cutoff {threshold:g}")
    xlabel = f"{column} / background sigma" if normalize_by_noise else column
    axis.set(xlabel=xlabel, ylabel="Cell fraction", title=f"{signal} cell-level eCDF", ylim=(0, 1))
    #if axis.lines:
        #axis.legend(title=group)
    return figure


def plot_cell_mean_histograms(
    cells: pd.DataFrame,
    signal: str,
    *,
    metric: str = "corrected_mean",
    group: str = "source",
    bins: int | str | np.ndarray = 30,
):
    """Plot per-cell signal means as comparable relative-frequency histograms.

    All groups use the same bin edges, and each group's bars sum to one. This
    makes image sets with unequal numbers of segmented cells directly
    comparable. The default signed ``corrected_mean`` deliberately retains
    values below zero: they indicate cells below the estimated background.

    Args:
        cells: Per-cell output table from :func:`measure_masks`.
        signal: Signal-channel name, for example ``"GD"``.
        metric: Signal metric suffix to plot. The default selects
            ``signal_<signal>_corrected_mean``.
        group: Column defining image sets, normally ``"source"``.
        bins: Shared NumPy histogram bin specification.

    Returns:
        Matplotlib figure containing one outlined relative-frequency histogram
        per group.
    """
    column = f"signal_{signal}_{metric}"
    if column not in cells or group not in cells:
        raise KeyError(f"Expected columns {column!r} and {group!r}.")

    if cells[group].notna().sum() == 0:
        raise ValueError(
            f"Grouping column {group!r} contains no non-missing values. "
            "Check how its labels were parsed or populated."
        )

    groups = [
        (str(label), table[column].dropna().to_numpy(dtype=float))
        for label, table in cells.groupby(group, sort=False)
    ]
    groups = [(label, values[np.isfinite(values)]) for label, values in groups]
    groups = [(label, values) for label, values in groups if values.size]
    if not groups:
        raise ValueError(f"No finite values found in {column!r}.")

    edges = np.histogram_bin_edges(np.concatenate([values for _, values in groups]), bins=bins)
    figure, axis = plt.subplots()
    for label, values in groups:
        axis.hist(
            values,
            bins=edges,
            weights=np.full(values.size, 1 / values.size),
            histtype="step",
            linewidth=1.5,
            label=label,
        )
    axis.set(
        xlabel=column,
        ylabel="Relative cell frequency",
        title=f"{signal} per-cell mean intensity distribution",
    )
    axis.set_ylim(0, .2)
    #axis.legend(title=group)
    return figure


def plot_quantile_profiles(cells, signal: str, *, group: str = "source"):
    """Plot the mean within-cell intensity quantile profile for each group."""
    columns = sorted((column for column in cells if column.startswith(f"signal_{signal}_q")), key=lambda column: int(column.rsplit("q", 1)[1]))
    if not columns or group not in cells:
        raise KeyError(f"No quantile columns for {signal!r}, or missing group {group!r}.")
    figure, axis = plt.subplots()
    percentiles = [int(column.rsplit("q", 1)[1]) for column in columns]
    for label, table in cells.groupby(group, sort=False):
        axis.plot(percentiles, table[columns].mean(axis=0), marker="o", label=str(label))
    axis.set(xlabel="Pixel-intensity percentile", ylabel="Mean per-cell intensity", title=f"{signal} quantile profiles")
    axis.legend(title=group)
    # Autoscale across every group, including negative corrected intensities.
    axis.margins(y=0.05)
    return figure


def plot_cell_distribution_heatmap(cells, signal: str, *, sort_by: str = "q50"):
    """Plot cells by distributional quantile bin, sorted by one selected bin."""
    columns = sorted((column for column in cells if column.startswith(f"signal_{signal}_q")), key=lambda column: int(column.rsplit("q", 1)[1]))
    key = f"signal_{signal}_{sort_by}"
    if not columns or key not in cells:
        raise KeyError(f"No quantile columns for {signal!r}, or missing {key!r}.")
    values = cells.sort_values(key)[columns].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, max(3, values.shape[0] / 30)))
    image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set(xlabel="Quantile", ylabel="Cell (sorted)", xticks=np.arange(len(columns)), xticklabels=[column.rsplit("q", 1)[1] for column in columns])
    figure.colorbar(image, ax=axis, label="Intensity")
    return figure


def random_crop(reference: np.ndarray, masks: np.ndarray, signal: np.ndarray | None = None, *, size: int = 1024, seed: int | None = None):
    """Return aligned, reproducible random square crops for segmentation QC."""
    reference, masks = np.asarray(reference), np.asarray(masks)
    if reference.shape != masks.shape:
        raise ValueError("reference and masks must have matching shapes.")
    if reference.ndim not in (2, 3):
        raise ValueError("Expected a YX image or ZYX volume.")

    if signal is not None and np.shape(signal) != reference.shape:
        raise ValueError("Signal must match the reference shape.")

    height, width = reference.shape[-2:]
    crop_h, crop_w = min(size, height), min(size, width)
    rng = np.random.default_rng(seed)
    y = int(rng.integers(height - crop_h + 1))
    x = int(rng.integers(width - crop_w + 1))
    crop = (
            Ellipsis,
            slice(y, y + crop_h),
            slice(x, x + crop_w),
        )
    result = (reference[crop], masks[crop])
    return (*result, None if signal is None else np.asarray(signal)[crop])
