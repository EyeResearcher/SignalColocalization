"""Regression checks for dispatch, pixel selection, and pipeline measurement."""

from dataclasses import asdict

import numpy as np
import pytest

from colocalize.background import (
    background_pixels, compute_background, estimate_background,
)
from colocalize.colocalize import measure_masks
from colocalize.datasets import AnalysisConfig, ReferenceSet, SignalChannel
from colocalize.readers import MicroscopyImage
from colocalize import pipeline


def scene():
    signal = np.full((20, 20), 10, dtype=np.uint16)
    masks = np.zeros(signal.shape, dtype=np.int32)
    masks[8:12, 8:12] = 1
    signal[masks > 0] = 100
    return signal, masks


@pytest.mark.parametrize("method, expected", [
    ("none", 0), ("median", 10), ("percentile", 10), ("fixed", 25),
])
def test_dispatch_methods_without_tissue(method, expected):
    signal, masks = scene()
    spec = SignalChannel(name="GD", channel=0, background_method=method,
                         background_fixed_value=25)
    result = compute_background(signal, masks, spec=spec)
    assert result["background"] == expected
    assert not np.any(result["sample_mask"] & (masks > 0))
    assert result["sample_mask"].dtype == bool
    cells = measure_masks(reference_image=signal, masks=masks,
                          signal_data={"GD": (signal, spec)}, backgrounds={"GD": result})
    assert cells.loc[0, "signal_GD_mean"] == 100
    assert cells.loc[0, "signal_GD_corrected_mean"] == 100 - expected
    if method in {"none", "fixed"}:
        assert np.isnan(result["sigma"])
        assert not result["sample_mask"].any()
        assert np.isnan(cells.loc[0, "signal_GD_frac_above_z3"])


def test_none_bypasses_sampling_even_when_cells_cover_every_pixel():
    result = compute_background(np.ones((2, 2)), np.ones((2, 2)), method="none")
    assert result["background"] == 0


def test_simple_helpers_ignore_cells_nonfinite_pixels_and_outside_region():
    signal, masks = scene()
    signal = signal.astype(float)
    signal[0, 0] = np.nan
    signal[0, 1] = np.inf
    signal[:, :5] = 1000
    region = np.zeros_like(signal, dtype=bool)
    region[:, 5:] = True
    selected = background_pixels(signal, masks, buffer_px=0, region_mask=region)
    assert not np.any(selected & (masks > 0))
    assert estimate_background(signal, selected).level == 10
    assert compute_background(signal, masks, method="median", region_mask=region)["background"] == 10


def test_explicit_percentile_is_used_instead_of_cell_intensities():
    signal, masks = scene()
    signal = signal.astype(float)
    signal[masks == 0] = np.arange(np.count_nonzero(masks == 0))
    spec = SignalChannel(name="GD", channel=0, background_method="percentile",
                         background_buffer_px=0, background_anchor_percentile=20)
    result = compute_background(signal, masks, spec=spec)
    assert result["background"] == np.percentile(signal[masks == 0], 20)


def test_full_workflow_excludes_dim_cells_and_preserves_diagnostics():
    signal, masks = scene()
    signal = signal.astype(float)
    signal[masks > 0] = 2  # Would re-enter the old final mask below the Otsu cutoff.
    signal[:, 15:] = 200
    dapi = np.full(signal.shape, 5.)
    ref = np.zeros(signal.shape)
    spec = SignalChannel(name="GD", channel=0, background_method="tissue_filtered_oop",
                         background_buffer_px=0)
    result = compute_background(signal, masks, dapi, ref, spec)
    assert result["background"] == 10
    assert np.array_equal(result["buffered_cells"], masks > 0)
    assert not np.any(result["sample_mask"] & (masks > 0))
    assert np.all(~result["sample_mask"] | result["background_candidates"])
    assert result["otsu_cutoff"] is None
    assert result["debris_cutoff"] is not None
    assert result["method"] == "tissue_filtered_oop"
    assert result["debris_filter_applied"]
    assert np.array_equal(result["sample_mask"], result["tissue_for_background"])


def test_explicit_method_overrides_signal_configuration():
    signal, masks = scene()
    spec = SignalChannel(name="GD", channel=0, background_method="none")
    result = compute_background(signal, masks, spec=spec, method="median")
    assert result["background"] == 10
    assert result["method"] == "median"


def test_missing_tissue_and_empty_background_fail_explicitly():
    signal, masks = scene()
    with pytest.raises(ValueError, match="requires a tissue"):
        compute_background(signal, masks, method="tissue_filtered_oop")
    with pytest.raises(ValueError, match="No finite background"):
        compute_background(signal, np.ones_like(masks), method="median")


@pytest.mark.parametrize(
    "retired_method",
    ["simple", "global", "global_surface", "otsu_low_percentile", "tissue_filtered"],
)
def test_retired_background_methods_are_rejected(retired_method):
    with pytest.raises(ValueError, match="Unknown background_method"):
        SignalChannel(name="GD", channel=0, background_method=retired_method)


def test_tissue_filtered_oop_configuration_roundtrip():
    signal, masks = scene()
    spec = SignalChannel(name="GD", channel=0, background_method="tissue_filtered_oop",
                         debris_percentile=90)
    spec = SignalChannel(**asdict(spec))
    result = compute_background(signal, masks, np.full(signal.shape, 5), signal, spec)
    assert result["method"] == "tissue_filtered_oop"


def test_measurement_uses_supplied_background_once_and_retains_negative_values():
    signal, masks = scene()
    spec = SignalChannel(name="GD", channel=0)
    cells = measure_masks(reference_image=signal, masks=masks,
                          signal_data={"GD": (signal, spec)},
                          backgrounds={"GD": {"background": 120., "sigma": 2.}})
    assert cells.loc[0, "signal_GD_mean"] == 100
    assert cells.loc[0, "signal_GD_corrected_mean"] == -20
    with pytest.raises(ValueError, match="Missing background"):
        measure_masks(reference_image=signal, masks=masks,
                      signal_data={"GD": (signal, spec)}, backgrounds={})


def test_run_analysis_selects_methods_without_tissue(tmp_path, monkeypatch):
    signal, masks = scene()
    acquisition = MicroscopyImage(tmp_path / "image.tif", signal[None], ("signal",))
    config = AnalysisConfig(
        input_dir=tmp_path, output_dir=tmp_path / "results",
        reference_sets=[ReferenceSet(name="cells", channel=0)],
        signal_channels=[
            SignalChannel(name="raw", channel=0, background_method="none"),
            SignalChannel(name="corrected", channel=0, background_method="median"),
        ],
        save_masks=False, save_segmentation=False,
    )
    monkeypatch.setattr(pipeline, "_input_paths", lambda config: [acquisition.path])
    monkeypatch.setattr(pipeline, "build_dataset", lambda *args: [acquisition])
    monkeypatch.setattr(pipeline, "_get_masks", lambda **kwargs: (masks, None))
    monkeypatch.setattr(pipeline, "_free_gpu_cache", lambda: None)
    monkeypatch.setattr(pipeline, "save_mask", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "make_segmentation_views", lambda **kwargs: {})
    monkeypatch.setattr(pipeline, "save_segmentation_views", lambda **kwargs: [])
    result = pipeline.run_analysis(config)
    row = result.cells.iloc[0]
    assert row.signal_raw_corrected_mean == 100
    assert row.signal_corrected_corrected_mean == 90
    assert (config.output_dir / "cells.csv").is_file()


def test_process_signal_honors_per_signal_tissue_channel(tmp_path):
    signal, masks = scene()
    dapi = np.full(signal.shape, 5.)
    acquisition = MicroscopyImage(tmp_path / "image.tif", np.stack([signal, dapi]), ("GD", "DAPI"))
    spec = SignalChannel(name="GD", channel=0, tissue_channel=1,
                         background_method="tissue_filtered_oop")
    table, diagnostics = pipeline._process_signal(
        acquisition=acquisition, masks=masks, reference=ReferenceSet(name="cells", channel=0),
        signal_channels=[spec],
    )
    assert diagnostics["GD"]["background"] == 10
    assert table.loc[0, "signal_GD_corrected_mean"] == 90
