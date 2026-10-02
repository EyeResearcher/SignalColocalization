"""Synthetic checks for configurable extracellular background correction."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import regionprops

from colocalize.colocalize import measure_masks
from colocalize.background import compute_background
from colocalize.datasets import SignalChannel
from colocalize.visualization import plot_cell_mean_histograms


def test_measure_masks_preserves_solidity_for_sparse_concave_labels():
    signal = np.arange(20 * 20, dtype=float).reshape(20, 20)
    masks = np.zeros(signal.shape, dtype=np.int32)
    masks[2:10, 2:5] = 3
    masks[7:10, 5:11] = 3
    masks[12:18, 13:18] = 20
    spec = SignalChannel(name="GD", channel=0, background_method="none")

    cells = measure_masks(
        reference_image=signal,
        masks=masks,
        signal_data={"GD": (signal, spec)},
        backgrounds={"GD": {"background": 0.0, "sigma": np.nan}},
    )
    expected = {region.label: region.solidity for region in regionprops(masks)}

    assert cells["cell_id"].tolist() == [3, 20]
    assert cells["solidity"].tolist() == [expected[3], expected[20]]


def test_tissue_filtered_oop_excludes_black_off_tissue_and_reports_enrichment():
    rng = np.random.default_rng(2)
    signal = np.zeros((120, 120), dtype=float)
    signal[:, :60] = 20 + rng.normal(0, 1, (120, 60))
    masks = np.zeros_like(signal, dtype=np.int32)
    masks[40:60, 20:40] = 1
    signal[40:60, 20:40] += 20
    spec = SignalChannel(
        name="GD",
        channel=0,
        background_method="tissue_filtered_oop",
        background_buffer_px=2,
    )

    diags = compute_background(signal, masks, signal, signal, spec)
    background, sigma = diags["background"], diags["sigma"]
    cells = measure_masks(reference_image=signal, masks=masks, signal_data={"GD": (signal, spec)},
                          backgrounds={"GD": compute_background(signal, masks, signal, signal, spec)})

    assert 18 < background < 22
    assert 0 < sigma < 2
    assert cells.loc[0, "signal_GD_corrected_median"] > 15
    assert cells.loc[0, "signal_GD_frac_above_z3"] > 0.9


def test_none_background_method_preserves_raw_signal_as_corrected_values():
    signal = np.full((80, 80), 12.0)
    masks = np.zeros_like(signal, dtype=np.int32)
    masks[20:40, 20:40] = 1
    spec = SignalChannel(name="GD", channel=0, background_method="none")

    cells = measure_masks(reference_image=signal, masks=masks, signal_data={"GD": (signal, spec)},
                          backgrounds={"GD": compute_background(signal, masks, signal, signal, spec)})

    assert cells.loc[0, "signal_GD_bg_level"] == 0
    assert cells.loc[0, "signal_GD_corrected_mean"] == cells.loc[0, "signal_GD_mean"]


def test_oop_background_uses_background_pixels_for_robust_sigma():
    rng = np.random.default_rng(12)
    signal = 20.0 + rng.normal(0, 2, (80, 80))
    masks = np.zeros(signal.shape, dtype=np.int32)
    masks[25:45, 25:45] = 1
    signal[masks > 0] += 20
    dapi = np.ones(signal.shape, dtype=float)
    spec = SignalChannel(
        name="GD",
        channel=0,
        background_method="tissue_filtered_oop",
        background_percentile=50,
        background_buffer_px=0,
        background_dapi_threshold=0.5,
        debris_percentile=99,
        debris_buffer_px=0,
        debris_min_area_cell_sd=5,
    )

    background = compute_background(
        signal, masks, dapi=dapi, reference=signal, spec=spec
    )
    cells = measure_masks(
        reference_image=signal,
        masks=masks,
        signal_data={"GD": (signal, spec)},
        backgrounds={"GD": background},
    )

    assert background["sigma"] > 0
    assert np.isfinite(cells.loc[0, "signal_GD_bg_sigma"])
    assert cells.loc[0, "signal_GD_frac_above_z3"] > 0.9


def test_cell_mean_histograms_use_relative_frequency_and_shared_bins():
    import pandas as pd

    cells = pd.DataFrame(
        {
            "source": ["left"] * 2 + ["right"] * 4,
            "signal_GD_corrected_mean": [-2.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    figure = plot_cell_mean_histograms(cells, "GD", bins=3)
    axis = figure.axes[0]

    assert axis.get_ylabel() == "Relative cell frequency"
    assert len(axis.patches) == 2  # one outlined histogram polygon per group


def test_cell_mean_histograms_reports_empty_grouping_column():
    import pandas as pd

    cells = pd.DataFrame(
        {
            "animal_id": [None, None],
            "signal_GD_corrected_mean": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="'animal_id' contains no non-missing values"):
        plot_cell_mean_histograms(cells, "GD", group="animal_id")
