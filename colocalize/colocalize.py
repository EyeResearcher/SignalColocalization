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
    """Expand labels without allowing neighboring objects to merge."""
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
    """Calculate one image-level positivity threshold for a signal channel."""
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
    source: str,
    reference_set: str,
    reference_image: np.ndarray,
    masks: np.ndarray,
    signals: dict[str, np.ndarray],
    signal_specs: dict[str, SignalChannel],
) -> pd.DataFrame:
    """Return one row of morphology and signal descriptors per labeled mask."""
    reference = np.asarray(reference_image, dtype=float)
    labels = np.asarray(masks)
    thresholds = {
        name: signal_threshold(image, signal_specs[name])
        for name, image in signals.items()
    }
    reference_threshold = _otsu_or_constant(reference)
    rows: list[dict[str, float | int | str | bool]] = []
    columns = [
        "source",
        "reference_set",
        "cell_id",
        "centroid_y",
        "centroid_x",
        "area_px",
        "perimeter_px",
        "equivalent_diameter_px",
        "eccentricity",
        "solidity",
        "reference_mean",
        "reference_median",
        "reference_integrated",
    ]
    signal_suffixes = [
        "threshold",
        "mean",
        "median",
        "max",
        "integrated",
        "positive_area_px",
        "positive_fraction",
        "positive_cell",
        "pearson_r",
        "positive_jaccard",
        "manders_m1_reference",
        "manders_m2_signal",
    ]
    columns.extend(
        f"signal_{name}_{suffix}"
        for name in signals
        for suffix in signal_suffixes
    )

    for region in regionprops(labels):
        coords = region.coords
        ref_values = reference[coords[:, 0], coords[:, 1]]
        reference_positive = ref_values > reference_threshold
        row: dict[str, float | int | str | bool] = {
            "source": source,
            "reference_set": reference_set,
            "cell_id": int(region.label),
            "centroid_y": float(region.centroid[0]),
            "centroid_x": float(region.centroid[1]),
            "area_px": int(region.area),
            "perimeter_px": float(region.perimeter),
            "equivalent_diameter_px": float(region.equivalent_diameter_area),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "reference_mean": _nan_stat(np.mean, ref_values),
            "reference_median": _nan_stat(np.median, ref_values),
            "reference_integrated": _nan_stat(np.sum, ref_values),
        }

        for name, image in signals.items():
            values = np.asarray(image, dtype=float)[coords[:, 0], coords[:, 1]]
            threshold = thresholds[name]
            positive = values > threshold
            intersection = reference_positive & positive
            union = reference_positive | positive
            ref_weights = _positive_weights(ref_values, reference_threshold)
            signal_weights = _positive_weights(values, threshold)
            prefix = f"signal_{name}_"
            row.update(
                {
                    f"{prefix}threshold": threshold,
                    f"{prefix}mean": _nan_stat(np.mean, values),
                    f"{prefix}median": _nan_stat(np.median, values),
                    f"{prefix}max": _nan_stat(np.max, values),
                    f"{prefix}integrated": _nan_stat(np.sum, values),
                    f"{prefix}positive_area_px": int(np.count_nonzero(positive)),
                    f"{prefix}positive_fraction": float(np.mean(positive)),
                    f"{prefix}positive_cell": bool(
                        np.mean(positive) >= signal_specs[name].positive_fraction_cutoff
                    ),
                    f"{prefix}pearson_r": _pearson(ref_values, values),
                    f"{prefix}positive_jaccard": _safe_ratio(
                        np.count_nonzero(intersection), np.count_nonzero(union)
                    ),
                    f"{prefix}manders_m1_reference": _safe_ratio(
                        np.sum(ref_weights[positive]), np.sum(ref_weights)
                    ),
                    f"{prefix}manders_m2_signal": _safe_ratio(
                        np.sum(signal_weights[reference_positive]), np.sum(signal_weights)
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def summarize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-cell table to one row per image/reference-set pair."""
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
        column for column in cells.columns if column.endswith("_mean") and column.startswith("signal_")
    ]
    for column in positive_columns:
        counts = cells.groupby(group_columns, sort=False)[column].sum().rename(f"{column}_count")
        base = base.merge(counts.reset_index(), on=group_columns, how="left")
    for column in intensity_columns:
        means = cells.groupby(group_columns, sort=False)[column].mean().rename(f"mean_{column}")
        base = base.merge(means.reset_index(), on=group_columns, how="left")
    return base


def _nan_stat(function, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(function(finite)) if finite.size else np.nan


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(left, right)[0, 1])


def _otsu_or_constant(image: np.ndarray) -> float:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.nan
    if np.all(finite == finite[0]):
        return float(finite[0])
    return float(threshold_otsu(finite))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _positive_weights(values: np.ndarray, threshold: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    baseline = threshold if np.isfinite(threshold) else (np.min(finite) if finite.size else 0)
    return np.clip(values - baseline, 0, None)
