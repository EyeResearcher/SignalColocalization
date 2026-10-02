"""Cell-percentage summaries evaluated at explicit ECDF cutoffs."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _signal_values(cells, signal, metric, normalize_by_noise):
    column = f"signal_{signal}_{metric}"
    values = cells[column].to_numpy(dtype=float, na_value=np.nan)
    if normalize_by_noise:
        if metric not in ("corrected_mean", "corrected_median"):
            raise ValueError("Noise normalization requires corrected_mean or corrected_median in raw units.")
        sigma = cells[f"signal_{signal}_bg_sigma"].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(sigma) & (sigma > 0)
        values = np.divide(values, sigma, out=np.full(values.shape, np.nan), where=valid)
    return values[np.isfinite(values)]


def _cutoffs(thresholds):
    values = np.asarray(tuple(thresholds), dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("thresholds must be a sequence of finite numbers.")
    return list(dict.fromkeys(values.tolist()))


def summarize_signal_ecdf(
    cells: pd.DataFrame,
    signal: str,
    *,
    metric: str = "corrected_median",
    group: str | Sequence[str] = ("source", "reference_set"),
    thresholds: Sequence[float] = (0.0,),
    normalize_by_noise: bool = False,
) -> pd.DataFrame:
    """Return counts and percentages strictly above each ECDF cutoff.

    ``percent_above = 100 * (1 - ecdf_at_threshold)``. Ties belong to the
    CDF (values <= threshold), not the positive tail. The denominator is the
    number of finite cell measurements in each group; excluded measurements
    are reported explicitly. Groups without valid measurements have NaN
    percentages, and empty inputs return a table with the output schema.

    By default each image/reference set is summarized separately. Grouping by
    treatment instead pools cells, giving images with more cells more weight.
    For comparisons, first summarize images, then aggregate within biological
    samples using explicit sample metadata.

    For corrected_mean/median, normalize_by_noise divides each cell's value
    by its saved positive, finite bg_sigma. Cutoffs then have units of robust
    background sigma. These are descriptive enrichment thresholds, not
    significance tests or calibrated false-positive rates. Zero sigma is
    excluded. Tail means/quantiles may already be transformed, so cannot use
    this option. Without normalization, thresholds use the metric's own units.

    For pixel extent use metric='frac_above_z3' and thresholds such as (0.1,
    0.5, 0.8): percentages then describe cells with >10%, >50%, or >80% of
    their pixels above background + 3 sigma. They are not pixel percentages.
    All results describe signal within reference masks, not molecular overlap.
    """
    groups = [group] if isinstance(group, str) else list(group)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("group must contain one or more distinct column names.")
    thresholds = _cutoffs(thresholds)
    # Validate even for empty input.
    _signal_values(cells, signal, metric, normalize_by_noise)
    cells[groups]
    columns = [*groups, "signal", "metric", "normalize_by_noise", "threshold",
               "cell_count", "valid_cell_count", "excluded_cell_count",
               "cells_above", "ecdf_at_threshold", "percent_above"]
    rows = []
    for keys, table in cells.groupby(groups, sort=False, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = _signal_values(table, signal, metric, normalize_by_noise)
        for threshold in thresholds:
            above = int(np.count_nonzero(values > threshold))
            rows.append({
                **dict(zip(groups, keys)), "signal": signal, "metric": metric,
                "normalize_by_noise": normalize_by_noise, "threshold": threshold,
                "cell_count": len(table), "valid_cell_count": values.size,
                "excluded_cell_count": len(table) - values.size, "cells_above": above,
                "ecdf_at_threshold": (values.size - above) / values.size if values.size else np.nan,
                "percent_above": 100 * above / values.size if values.size else np.nan,
            })
    return pd.DataFrame(rows, columns=columns)
