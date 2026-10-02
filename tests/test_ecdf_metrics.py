"""Known-count checks for ECDF colocalization proxies."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from colocalize import summarize_signal_ecdf, plot_signal_ecdf


def example_cells():
    return pd.DataFrame({
        "source": ["a"] * 6 + ["b"] * 2,
        "reference_set": ["ref"] * 8,
        "signal_HD_corrected_median": [-1, 0, 2, 2, np.nan, np.inf, 4, 8],
        "signal_HD_bg_sigma": [1, 1, 1, 2, 1, 1, 0, np.nan],
    })


def test_exact_tail_percentages_ties_and_exclusions():
    result = summarize_signal_ecdf(example_cells(), "HD", thresholds=(0, 2))
    a = result[result.source.eq("a")].set_index("threshold")
    assert a.loc[0, "percent_above"] == 50
    assert a.loc[2, "percent_above"] == 0
    assert a.loc[0, "cells_above"] == 2
    assert a.loc[0, "valid_cell_count"] == 4
    assert a.loc[0, "excluded_cell_count"] == 2
    np.testing.assert_allclose(result.percent_above, 100 * (1 - result.ecdf_at_threshold))


def test_noise_cutoffs_use_each_cells_sigma_and_exclude_invalid_sigma():
    result = summarize_signal_ecdf(example_cells(), "HD", thresholds=(1,), normalize_by_noise=True)
    a, b = result.iloc[0], result.iloc[1]
    assert a.percent_above == 25  # normalized values: -1, 0, 2, 1
    assert b.valid_cell_count == 0
    assert b.excluded_cell_count == 2
    assert np.isnan(b.percent_above)
    assert np.isnan(b.ecdf_at_threshold)


def test_reference_sets_stay_separate_and_empty_schema_is_preserved():
    cells = example_cells()
    cells.loc[2:3, "reference_set"] = "other"
    result = summarize_signal_ecdf(cells, "HD")
    assert len(result) == 3
    assert result.loc[result.reference_set.eq("other"), "percent_above"].item() == 100
    empty = summarize_signal_ecdf(cells.iloc[:0], "HD")
    assert empty.empty
    assert list(empty.columns) == list(result.columns)


def test_extent_cutoffs_are_cell_percentages_and_strict():
    cells = example_cells().iloc[:4].copy()
    cells["signal_HD_frac_above_z3"] = [0, .1, .5, 1]
    result = summarize_signal_ecdf(cells, "HD", metric="frac_above_z3", thresholds=(.1, .5, .8))
    assert result.percent_above.tolist() == [50, 25, 25]


def test_plot_markers_match_summary_and_do_not_clip_negative_values():
    cells = example_cells()
    cells.loc[0, "signal_HD_corrected_median"] = -200
    figure = plot_signal_ecdf(cells, "HD", thresholds=(0, 1), normalize_by_noise=True)
    summary = summarize_signal_ecdf(cells, "HD", thresholds=(0, 1), normalize_by_noise=True)
    markers = [line for line in figure.axes[0].lines if line.get_marker() == "o"]
    np.testing.assert_allclose([line.get_ydata()[0] for line in markers], summary.iloc[:2].ecdf_at_threshold)
    assert figure.axes[0].get_xlim()[0] < -200
    figure.canvas.draw()
    plt.close(figure)


def test_invalid_thresholds_and_transformed_noise_metrics_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize_signal_ecdf(example_cells(), "HD", thresholds=(np.inf,))
    cells = example_cells().assign(signal_HD_tail_mean=1)
    with pytest.raises(ValueError, match="raw units"):
        summarize_signal_ecdf(cells, "HD", metric="tail_mean", normalize_by_noise=True)


def test_export_preserves_unavailable_images_and_skips_absent_extent(tmp_path):
    from scripts.export_ecdf_metrics import export_metrics

    result, paths = export_metrics(example_cells(), "HD", tmp_path)
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    assert len(result) == 10
    compact = pd.read_csv(tmp_path / "ecdf_proxy_percentages.csv")
    b = compact.loc[compact.source.eq("b")].iloc[0]
    assert b["median_above_background > 0"] == 100
    assert pd.isna(b["median_noise_sweep > 0"])


def test_eye_pooling_weights_valid_cells_and_preserves_unavailable_groups():
    from scripts.export_ecdf_metrics import pool_metrics_by_eye

    cells = pd.DataFrame({
        "source": ["1_L_image.tif"] * 2 + ["2_L_image.tif"] * 8 + ["3_L_image.tif", "1_R_image.tif"],
        "reference_set": ["ref"] * 12,
        "signal_HD_corrected_median": [2, 0] + [0] * 8 + [np.nan, np.nan],
    })
    metrics = summarize_signal_ecdf(cells, "HD").assign(proxy="median_above_background")
    pooled = pool_metrics_by_eye(metrics).set_index("eye")
    assert pooled.loc["L", "percent_above"] == 10  # 1 / 10 cells; not the 25% image average
    assert pooled.loc["L", "cell_count"] == 11
    assert pooled.loc["L", "excluded_cell_count"] == 1
    assert pooled.loc["L", "image_count"] == 3
    assert pd.isna(pooled.loc["R", "percent_above"])


def test_tiled_source_assignments_pool_fields_into_biological_eyes(tmp_path):
    import json

    from scripts.export_ecdf_metrics import group_tiled_sources

    cells = pd.DataFrame({
        "source": ["scan_G001_0001.oir", "scan_G002_0001.oir", "scan_G007_0001.oir"],
        "value": [1, 2, 3],
    })
    eyes = tmp_path / "eyes.json"
    animals = tmp_path / "animal_id.json"
    eyes.write_text(json.dumps({"R": ["001", "002"], "L": ["007"]}), encoding="utf-8")
    animals.write_text(json.dumps({"9911": ["001", "002", "007"]}), encoding="utf-8")

    grouped = group_tiled_sources(cells, eyes, animals)

    assert grouped.source.tolist() == ["9911_R", "9911_R", "9911_L"]
    assert grouped.field_id.tolist() == ["001", "002", "007"]
    assert grouped.field_source.tolist() == cells.source.tolist()
