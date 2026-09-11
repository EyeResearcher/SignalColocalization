"""Per-mask morphology, intensity, and colocalization measurements."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from .datasets import SignalChannel


def expand_masks(masks: np.ndarray, expansion_radius: int = 1) -> np.ndarray:
    """Expand labeled regions outward without merging neighboring objects.

    Each background pixel within ``expansion_radius`` of a label is assigned to
    the nearest labeled region.  Adjacent objects never merge because
    equidistant background pixels preserve the existing boundary.

    Args:
        masks: Integer label array where 0 is background and positive integers
            are distinct objects.
        expansion_radius: Maximum distance in pixels to grow each label.
            A value of 0 or less returns a copy of the input unchanged.

    Returns:
        Label array of the same dtype and shape as ``masks`` with expanded
        object boundaries.
    """
    labels = np.asarray(masks)
    if expansion_radius <= 0 or not np.any(labels):
        return labels.copy()
    background = labels == 0
    distances, indices = ndi.distance_transform_edt(background, return_indices=True)
    expanded = labels.copy()
    fill = background & (distances <= expansion_radius)
    nearest = labels[tuple(axis[fill] for axis in indices)]
    expanded[fill] = nearest
    return expanded


def signal_threshold(image: np.ndarray, spec: SignalChannel) -> float:
    """Calculate the image-level positivity threshold for a signal channel.

    The threshold is derived from the full image (not per-cell) and is used to
    classify pixels as signal-positive or signal-negative inside each mask.

    Args:
        image: 2-D intensity array for the signal channel.
        spec: Channel configuration specifying the thresholding method
            (``'otsu'``, ``'percentile'``, ``'absolute'``, or ``'none'``) and
            any associated numeric value.

    Returns:
        Scalar threshold float.  Pixels strictly above this value are positive.
        Returns ``np.nan`` if the image contains no finite values.

    Raises:
        ValueError: If ``spec.threshold_method`` is ``'absolute'`` and
            ``spec.threshold_value`` is ``None``, or if the method name is
            unrecognised.
    """
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    method = spec.threshold_method.casefold()
    if method == "otsu":
        if np.all(finite == finite[0]):
            return float(finite[0])
        return float(threshold_otsu(finite))
    if method == "percentile":
        percentile = 99.0 if spec.threshold_value is None else spec.threshold_value
        return float(np.percentile(finite, percentile))
    if method == "absolute":
        if spec.threshold_value is None:
            raise ValueError(f"Signal {spec.name!r} needs an absolute threshold_value.")
        return float(spec.threshold_value)
    if method == "none":
        return -np.inf
    raise ValueError(
        f"Unknown threshold method {spec.threshold_method!r}; "
        "choose 'otsu', 'percentile', 'absolute', or 'none'."
    )


def measure_masks(
    *,
    reference_image: np.ndarray,
    masks: np.ndarray,
    signal_data: dict[str, tuple[np.ndarray, SignalChannel]],
    tile_yx: tuple[int, int] = (0, 0),
) -> pd.DataFrame:
    """Return one row of morphology and signal descriptors per labeled mask.

    Args:
        reference_image: 2-D array for the reference (segmentation) channel.
        masks: Integer label array aligned to ``reference_image``; each
            positive integer identifies one segmented object.
        signal_data: Mapping of signal name to ``(image array, SignalChannel
            spec)`` for every channel to be measured inside the masks.
        tile_yx: ``(y0, x0)`` pixel offset of this tile within the full image.
            Centroids are reported in full-image coordinates.  Defaults to
            ``(0, 0)`` for non-tiled runs.

    Returns:
        DataFrame with one row per labeled mask.  Columns include tile
        coordinates, cell morphology, reference-channel intensity stats, and
        per-signal colocalization metrics (threshold, Pearson r, Manders
        coefficients, Jaccard index, etc.).  The ``source`` and
        ``reference_set`` identity columns are not included; the caller should
        prepend them with ``DataFrame.insert``.
    """
    reference = np.asarray(reference_image, dtype=float)
    labels = np.asarray(masks)
    reference_threshold = _otsu_or_constant(reference)
    thresholds = {name: signal_threshold(img, spec) for name, (img, spec) in signal_data.items()}
    signal_suffixes = [
        "threshold", "mean", "median", "max", "integrated",
        "positive_area_px", "positive_fraction", "positive_cell",
        "pearson_r", "positive_jaccard", "manders_m1_reference", "manders_m2_signal",
    ]
    columns = [
        "tile_y", "tile_x", "cell_id", "centroid_y", "centroid_x",
        "area_px", "perimeter_px", "equivalent_diameter_px",
        "eccentricity", "solidity",
        "reference_mean", "reference_median", "reference_integrated",
    ]
    columns.extend(f"signal_{name}_{suffix}" for name in signal_data for suffix in signal_suffixes)
    rows = [
        _measure_region(region, reference, reference_threshold, signal_data, thresholds, tile_yx)
        for region in regionprops(labels)
    ]
    return pd.DataFrame(rows, columns=columns)


def _measure_region(  # pylint: disable=too-many-arguments
    region,
    reference: np.ndarray,
    reference_threshold: float,
    signal_data: dict[str, tuple[np.ndarray, SignalChannel]],
    thresholds: dict[str, float],
    tile_yx: tuple[int, int],
) -> dict:
    """Assemble the full measurement row for one labeled region."""
    coords = region.coords
    ref_values = reference[coords[:, 0], coords[:, 1]]
    reference_positive = ref_values > reference_threshold
    ref_weights = _positive_weights(ref_values, reference_threshold)
    row: dict[str, float | int | bool] = {
        "tile_y": tile_yx[0],
        "tile_x": tile_yx[1],
        "cell_id": int(region.label),
        "centroid_y": float(region.centroid[0]) + tile_yx[0],
        "centroid_x": float(region.centroid[1]) + tile_yx[1],
        "area_px": int(region.area),
        "perimeter_px": float(region.perimeter),
        "equivalent_diameter_px": float(region.equivalent_diameter_area),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "reference_mean": _nan_stat(np.mean, ref_values),
        "reference_median": _nan_stat(np.median, ref_values),
        "reference_integrated": _nan_stat(np.sum, ref_values),
    }
    for name, (image, spec) in signal_data.items():
        row.update(
            _measure_signal(
                coords, image, spec, thresholds[name], ref_values, reference_positive, ref_weights
            )
        )
    return row


def _measure_signal(  # pylint: disable=too-many-arguments
    coords: np.ndarray,
    image: np.ndarray,
    spec: SignalChannel,
    threshold: float,
    ref_values: np.ndarray,
    reference_positive: np.ndarray,
    ref_weights: np.ndarray,
) -> dict:
    """Compute all signal-vs-reference colocalization metrics for one mask region."""
    values = np.asarray(image, dtype=float)[coords[:, 0], coords[:, 1]]
    positive = values > threshold
    signal_weights = _positive_weights(values, threshold)
    prefix = f"signal_{spec.name}_"
    return {
        f"{prefix}threshold": threshold,
        f"{prefix}mean": _nan_stat(np.mean, values),
        f"{prefix}median": _nan_stat(np.median, values),
        f"{prefix}max": _nan_stat(np.max, values),
        f"{prefix}integrated": _nan_stat(np.sum, values),
        f"{prefix}positive_area_px": int(np.count_nonzero(positive)),
        f"{prefix}positive_fraction": float(np.mean(positive)),
        f"{prefix}positive_cell": bool(np.mean(positive) >= spec.positive_fraction_cutoff),
        f"{prefix}pearson_r": _pearson(ref_values, values),
        f"{prefix}positive_jaccard": _safe_ratio(
            np.count_nonzero(positive & reference_positive),
            np.count_nonzero(positive | reference_positive),
        ),
        f"{prefix}manders_m1_reference": _safe_ratio(
            np.sum(ref_weights[positive]), np.sum(ref_weights)
        ),
        f"{prefix}manders_m2_signal": _safe_ratio(
            np.sum(signal_weights[reference_positive]), np.sum(signal_weights)
        ),
    }


def summarize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-cell table to one summary row per image/reference-set pair.

    Args:
        cells: Per-cell DataFrame as returned by :func:`measure_masks` with
            ``source`` and ``reference_set`` columns already inserted.  Expected
            to also contain ``cell_id``, ``area_px``, any
            ``signal_*_positive_cell`` columns, and any ``signal_*_mean``
            columns.

    Returns:
        DataFrame with one row per ``(source, reference_set)`` group.  Columns
        include cell count, total/mean/median cell area, a positive-cell count
        for each signal, and a mean intensity for each signal channel.  Returns
        a minimal skeleton DataFrame when ``cells`` is empty.
    """
    if cells.empty:
        return pd.DataFrame(columns=["source", "reference_set", "cell_count"])

    group_columns = ["source", "reference_set"]
    base = (
        cells.groupby(group_columns, sort=False)
        .agg(
            cell_count=("cell_id", "count"),
            total_cell_area_px=("area_px", "sum"),
            mean_cell_area_px=("area_px", "mean"),
            median_cell_area_px=("area_px", "median"),
        )
        .reset_index()
    )
    positive_columns = [
        column for column in cells.columns if column.endswith("_positive_cell")
    ]
    intensity_columns = [
        column
        for column in cells.columns
        if column.endswith("_mean") and column.startswith("signal_")
    ]
    for column in positive_columns:
        counts = cells.groupby(group_columns, sort=False)[column].sum().rename(f"{column}_count")
        base = base.merge(counts.reset_index(), on=group_columns, how="left")
    for column in intensity_columns:
        means = cells.groupby(group_columns, sort=False)[column].mean().rename(f"mean_{column}")
        base = base.merge(means.reset_index(), on=group_columns, how="left")
    return base


def _nan_stat(function, values: np.ndarray) -> float:
    """Apply a reduction function over finite values, returning NaN for empty arrays."""
    finite = values[np.isfinite(values)]
    return float(function(finite)) if finite.size else np.nan


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Pearson correlation coefficient, ignoring non-finite value pairs."""
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(left, right)[0, 1])


def _otsu_or_constant(image: np.ndarray) -> float:
    """Return the Otsu threshold, or the constant pixel value when the image is uniform."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.nan
    if np.all(finite == finite[0]):
        return float(finite[0])
    return float(threshold_otsu(finite))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else np.nan


def _positive_weights(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return per-pixel weights clipped to zero at or below the threshold baseline."""
    finite = values[np.isfinite(values)]
    baseline = threshold if np.isfinite(threshold) else (np.min(finite) if finite.size else 0)
    return np.clip(values - baseline, 0, None)
