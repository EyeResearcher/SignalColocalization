"""Per-mask morphology, intensity, colocalization, and signal-distribution measurements."""

from __future__ import annotations

import warnings

import tqdm

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import grid_points_in_poly, regionprops, regionprops_table

try:  # Cellpose installs OpenCV; keep a fallback for minimal installations.
    import cv2
except ImportError:  # pragma: no cover - exercised only without Cellpose/OpenCV
    cv2 = None

from .datasets import SignalChannel
from .background import  _robust_sigma


def expand_masks(masks: np.ndarray, expansion_radius: int = 1) -> np.ndarray:
    """Expand labels by at most ``expansion_radius`` without merging objects."""
    labels = np.asarray(masks)
    if expansion_radius <= 0 or not np.any(labels):
        return labels.copy()
    background = labels == 0
    distances, indices = ndi.distance_transform_edt(background, return_indices=True)
    expanded = labels.copy()
    grow = background & (distances <= expansion_radius)
    expanded[grow] = labels[tuple(indices[:, grow])]
    return expanded




def signal_threshold(image: np.ndarray, spec: SignalChannel) -> float:
    """Return the raw threshold used by legacy positivity/colocalization metrics.

    This threshold does not affect background subtraction or distributional
    metrics. ``none`` returns negative infinity, making all finite pixels
    positive for the legacy binary outputs.
    """
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    if spec.threshold_method == "otsu":
        return _otsu_or_constant(finite)
    if spec.threshold_method == "percentile":
        return float(np.percentile(finite, 99.0 if spec.threshold_value is None else spec.threshold_value))
    if spec.threshold_method == "absolute":
        if spec.threshold_value is None:
            raise ValueError(f"Signal {spec.name!r} needs an absolute threshold_value.")
        return float(spec.threshold_value)
    if spec.threshold_method == "percent_of_max":
        if spec.threshold_value is None:
            raise ValueError(f"Signal {spec.name!r} needs a percent_of_max threshold_value.")
        return float(spec.threshold_value * np.max(finite))
    if spec.threshold_method == "none":
        return -np.inf
    raise ValueError(f"Unknown threshold method {spec.threshold_method!r}.")

def measure_masks(
    *,
    reference_image: np.ndarray,
    masks: np.ndarray,
    signal_data: dict[str, tuple[np.ndarray, SignalChannel]],
    backgrounds: dict[str, dict],
    morphology_properties: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    """Measure morphology, raw signal, and corrected signal distributions per cell.

    Backgrounds are computed by the caller once per signal/reference-mask set.
    Measurements are made within every nonzero label in ``masks``.
    Raw intensity and colocalization outputs are retained
    for backwards compatibility. Corrected outputs are based on
    ``raw_signal - background`` and intentionally preserve negative values.

    
    Output columns include:

    * Cell morphology: ``cell_id``, centroid, area, perimeter, diameter,
      eccentricity, and solidity.
    * ``reference_*``: raw reference-channel mean, median, and integral.
    * ``signal_<name>_mean``, ``median``, ``max``, ``integrated``, positivity,
      Pearson, Jaccard, and Manders metrics: raw signal values.
    * ``signal_<name>_bg_level`` and ``bg_sigma``: background sampled at the
      cell and image-level robust noise scale.
    * ``corrected_mean``, ``corrected_median``, and ``corrected_integrated``:
      untransformed background-subtracted raw units.
    * ``q05`` through configured quantiles, ``iqr``, and ``tail_mean``:
      background-subtracted values after optional noise normalization and
      optional asinh transformation.
    * ``frac_above_z*``: fraction of cell pixels with corrected value above
      the stated multiple of ``bg_sigma``; independent of asinh scaling.

    Args:
        reference_image: Raw reference-channel Y×X image.
        masks: Aligned integer label image; zero denotes non-cell pixels.
        signal_data: Mapping of output signal name to ``(image, spec)``.
        backgrounds: Mapping of signal names to scalar ``background`` and
            ``sigma`` results from ``background.compute_background``. Signals
            must remain raw; subtraction is performed here exactly once.
        morphology_properties: List of cell morphology properties to measure.
    """

    if reference_image.shape != masks.shape:
        raise ValueError("reference_image and masks must have matching shapes.")

    if masks.ndim not in (2, 3):
        raise ValueError("Masks must be a 2D or 3D label array.")
    
    if morphology_properties is None:
        morphology_properties = ["area", "equivalent_diameter_area"]

    properties = tuple(dict.fromkeys(
    ["label", "centroid", *morphology_properties]
    ))

    morphology = regionprops_table(masks, properties=properties)

# Preserve existing column names where appropriate.
    axes = "yx" if masks.ndim == 2 else "zyx"

    column_names = {
    "label": "cell_id",
    "area": "area_px" if masks.ndim == 2 else "volume_voxels",
    "perimeter": "perimeter_px",
    "equivalent_diameter_area": "equivalent_diameter_px",
    **{ f"centroid-{i}": f"centroid_{axis}"
        for i, axis in enumerate(axes)},
    }

    morphology = {column_names.get(name, name): values
                                    for name, values in morphology.items()}

    columns = [
    *morphology,
    "reference_mean",
    "reference_median",
    "reference_integrated",
]
    legacy = (
        "threshold", "mean", "median", "max", "integrated",
        "positive_area_px", "positive_fraction", "positive_cell",
        "pearson_r", "positive_jaccard", "manders_m1_reference",
        "manders_m2_signal",
    )
    for signal_name, (gd, spec) in signal_data.items():
        if gd.shape != masks.shape:
            raise ValueError(f"Signal {signal_name!r} must match the masks.")
        if signal_name not in backgrounds:
            raise ValueError(f"Missing background for signal {signal_name!r}.")
        result = backgrounds[signal_name]
        if not {"background", "sigma"}.issubset(result):
            raise ValueError(f"Background for {signal_name!r} needs background and sigma.")
        level, sigma = result["background"], result["sigma"]
        if np.ndim(level) != 0 or not np.isfinite(level):
            raise ValueError("Measurement requires a finite scalar background.")
        if np.ndim(sigma) != 0 or np.isinf(sigma) or sigma < 0:
            raise ValueError("Background sigma must be nonnegative or NaN when unavailable.")
        columns.extend(f"signal_{signal_name}_{suffix}" for suffix in legacy)
        columns.extend(f"signal_{signal_name}_{suffix}" for suffix in _distribution_suffixes(spec))

    thresholds = {name: signal_threshold(image, spec)
              for name, (image, spec) in signal_data.items()}
    ref_threshold = _otsu_or_constant(reference_image)
    # Batch the standard morphology properties to reduce Python overhead.
    # RegionProperties objects are still used for cached coordinates and by

   
    regions = regionprops(masks)
    rows = [
        _measure_region(
            region, index, morphology, reference_image, ref_threshold,
            signal_data, thresholds, backgrounds,
        )
        for index, region in enumerate(regions)
    ]
    return pd.DataFrame(rows, columns=columns)


def _measure_region(
    region, index, morphology, reference, reference_threshold,
    signal_data, thresholds, backgrounds,
) -> dict:
    """Build one complete measurement row for a labeled cell region.

    Reference morphology and raw intensity are calculated first. Each signal
    then contributes raw legacy colocalization values and distributional values
    generated from its already-estimated image-level background.
    """
    coords = region.coords
    indices = tuple(coords.T)
    ref_values = reference[indices]
    ref_positive, ref_weights = ref_values > reference_threshold, _positive_weights(ref_values, reference_threshold)
    row = {name: values[index]
        for name, values in morphology.items()}
    
    row.update({
        "reference_mean": _nan_stat(np.mean, ref_values),
        "reference_median": _nan_stat(np.median, ref_values),
        "reference_integrated": _nan_stat(np.sum, ref_values),
    })
    for name, (image, spec) in signal_data.items():
        values = image[indices]
        row.update(_measure_signal(values, name, thresholds[name], ref_values, ref_positive, ref_weights, spec))
        row.update(_distribution_row(values, coords, backgrounds[name]["background"], backgrounds[name]["sigma"], name, noise_normalize=spec.noise_normalize, asinh_scale=spec.asinh_scale, tail_fraction=spec.tail_fraction, quantiles=spec.quantiles, noise_thresholds=spec.noise_thresholds))
    return row


def _measure_signal(values, name, threshold, ref_values, reference_positive, ref_weights, spec) -> dict:
    """Return legacy raw-intensity and threshold-based colocalization outputs.

    No background subtraction, noise normalization, or nonlinear transform is
    performed here. This separation keeps historical ``signal_<name>_*``
    positivity and Manders values directly comparable with older analyses.
    """
    positive, signal_weights = values > threshold, _positive_weights(values, threshold)
    prefix = f"signal_{name}_"
    return {f"{prefix}threshold": threshold,
            f"{prefix}mean": _nan_stat(np.mean, values),
            f"{prefix}median": _nan_stat(np.median, values),
            f"{prefix}max": _nan_stat(np.max, values),
            f"{prefix}integrated": _nan_stat(np.sum, values),
            f"{prefix}positive_area_px": int(np.count_nonzero(positive)),
            f"{prefix}positive_fraction": float(np.mean(positive)),
            f"{prefix}positive_cell": bool(np.mean(positive) >= spec.positive_fraction_cutoff),
            f"{prefix}pearson_r": _pearson(ref_values, values),
            f"{prefix}positive_jaccard": _safe_ratio(np.count_nonzero(positive & reference_positive), np.count_nonzero(positive | reference_positive)),
            f"{prefix}manders_m1_reference": _safe_ratio(np.sum(ref_weights[positive]), np.sum(ref_weights)),
            f"{prefix}manders_m2_signal": _safe_ratio(np.sum(signal_weights[reference_positive]), np.sum(signal_weights))}


def _distribution_row(values, coords, background, sigma, name, noise_normalize=False, asinh_scale=None, tail_fraction=0.1, quantiles=(0.25, 0.5, 0.75), noise_thresholds=(1.0,)) -> dict:
    """Return one cell's background-corrected distributional signal outputs.

    ``corrected_*`` fields are calculated from ``values - background`` in raw
    intensity units. A separate temporary array is divided by ``sigma`` only
    when ``noise_normalize`` is enabled and transformed with
    ``asinh(value / asinh_scale)`` only when requested. Quantiles, IQR, and
    upper-tail mean use that temporary array. ``frac_above_z*`` always uses
    untransformed corrected values divided by ``sigma``.
    """
    prefix = f"signal_{name}_"
    background_values, corrected = background, np.asarray(values, dtype=float) - float(background)
    transformed = corrected.copy()

    if noise_normalize and np.isfinite(sigma) and sigma > 0:
        transformed /= sigma
    if asinh_scale is not None:
        transformed = np.arcsinh(transformed / asinh_scale)

    finite = transformed[np.isfinite(transformed)]
    # np.quantile partitions its input.  Calling it once per requested
    # quantile repeated that work (ten times with the default configuration).
    # Include the IQR endpoints in the same call and reuse the results.
    requested = tuple(quantiles)
    if finite.size:
        quantile_levels = np.asarray((*requested, 0.25, 0.75), dtype=float)
        quantile_values = np.quantile(finite, quantile_levels)
        iqr = float(quantile_values[-1] - quantile_values[-2])
    else:
        quantile_values = np.full(len(requested) + 2, np.nan)
        iqr = np.nan
    result = {f"{prefix}bg_level": _nan_stat(np.median, background_values), f"{prefix}bg_sigma": float(sigma), f"{prefix}iqr": iqr, f"{prefix}corrected_mean": _nan_stat(np.mean, corrected), f"{prefix}corrected_median": _nan_stat(np.median, corrected), f"{prefix}corrected_integrated": _nan_stat(np.sum, corrected), f"{prefix}tail_mean": _tail_mean(transformed, tail_fraction)}
    for quantile, value in zip(requested, quantile_values):
        result[f"{prefix}{_quantile_label(quantile)}"] = float(value)
    z_values = corrected / sigma if np.isfinite(sigma) and sigma > 0 else np.full(corrected.shape, np.nan)
    for threshold in noise_thresholds:
        result[f"{prefix}frac_above_z{_threshold_label(threshold)}"] = float(np.nanmean(z_values > threshold)) if np.any(np.isfinite(z_values)) else np.nan
    return result


def summarize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cells to image/reference-set rows, with cells as replicates."""

    groups = ["source", "reference_set"]
    aggregations = {"cell_count": ("cell_id", "count"),}

    for size_column in ("area_px", "volume_voxels"):
        if size_column in cells.columns:
            aggregations.update({f"total_cell_{size_column}": (
                size_column, lambda values: values.sum(min_count=1),
            ),
            f"mean_cell_{size_column}": (size_column, "mean"),
            f"median_cell_{size_column}": (size_column, "median"),
        })

    base = (cells.groupby(groups, sort=False).agg(**aggregations).reset_index())

    signal_columns = [c for c in cells if c.startswith("signal_")]
    wanted = [c for c in signal_columns if c.endswith("_positive_cell") or (c.endswith("_mean") and not c.endswith(("_corrected_mean", "_tail_mean"))) or any(token in c for token in ("_corrected_median", "_corrected_integrated", "_q50", "_q90", "_tail_mean", "_frac_above_z"))]
    for column in dict.fromkeys(wanted):
        values = cells.groupby(groups, sort=False)[column].sum() if column.endswith("_positive_cell") else cells.groupby(groups, sort=False)[column].mean()
        name = f"{column}_count" if column.endswith("_positive_cell") else f"mean_{column}"
        base = base.merge(values.rename(name).reset_index(), on=groups, how="left")
    return base



def _hist_mode(values):
    """Return the midpoint of the most populated automatic histogram bin."""
    values = np.asarray(values, dtype=float)
    if not values.size or np.all(values == values[0]):
        return float(values[0]) if values.size else np.nan
    counts, edges = np.histogram(values, bins="auto")
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) / 2)



def _distribution_suffixes(spec):
    """Return output suffixes emitted by :func:`_distribution_row` for a spec."""
    return ("bg_level", "bg_sigma", *(_quantile_label(q) for q in spec.quantiles), "iqr", "corrected_mean", "corrected_median", "corrected_integrated", "tail_mean", *(f"frac_above_z{_threshold_label(t)}" for t in spec.noise_thresholds))


def _fast_solidity(region) -> float:
    """Return skimage-compatible solidity without one Qhull call per cell.

    On Windows, SciPy's Qhull wrapper opens a temporary file for every hull,
    making ``RegionProperties.solidity`` unusually expensive for images with
    thousands of cells.  OpenCV computes the same 2-D hull in memory.  The
    half-pixel diamond offsets and integer-grid inclusion below match
    ``skimage.morphology.convex_hull_image`` semantics.
    """
    if cv2 is None:
        return float(region.solidity)

    image = np.ascontiguousarray(region.image, dtype=np.uint8)
    occupied_rows = np.flatnonzero(np.any(image, axis=1))
    occupied_cols = np.flatnonzero(np.any(image, axis=0))
    left = np.argmax(image[occupied_rows], axis=1)
    right = image.shape[1] - 1 - np.argmax(image[occupied_rows, ::-1], axis=1)
    top = np.argmax(image[:, occupied_cols], axis=0)
    bottom = image.shape[0] - 1 - np.argmax(image[::-1, occupied_cols], axis=0)
    candidates = np.concatenate(
        (
            np.column_stack((occupied_rows, left)),
            np.column_stack((top, occupied_cols)),
            np.column_stack((occupied_rows, right)),
            np.column_stack((bottom, occupied_cols)),
        )
    )
    offsets = np.asarray(((-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)))
    points = np.unique(
        (candidates[:, None, :] + offsets).reshape(-1, 2), axis=0
    ).astype(np.float32, copy=False)
    vertices = cv2.convexHull(points, returnPoints=True).reshape(-1, 2)
    convex_area = np.count_nonzero(grid_points_in_poly(image.shape, vertices))
    return float(region.area / convex_area) if convex_area else np.nan


def _quantile_label(value): return f"q{int(round(value * 100)):02d}"
def _threshold_label(value): return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
def _tail_mean(values, fraction):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size: return np.nan
    count = max(1, int(np.ceil(finite.size * fraction)))
    return float(np.mean(np.partition(finite, finite.size - count)[-count:]))
def _nan_stat(function, values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(function(finite)) if finite.size else np.nan
def _pearson(left, right):
    finite = np.isfinite(left) & np.isfinite(right); left, right = left[finite], right[finite]
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0: return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return float(np.corrcoef(left, right)[0, 1])
def _otsu_or_constant(image):
    finite = np.asarray(image)[np.isfinite(image)]
    if not finite.size: return np.nan
    return float(finite[0]) if np.all(finite == finite[0]) else float(threshold_otsu(finite))
def _safe_ratio(numerator, denominator): return float(numerator / denominator) if denominator > 0 else np.nan
def _positive_weights(values, threshold):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.clip(values - (threshold if np.isfinite(threshold) else (np.min(finite) if finite.size else 0)), 0, None)
