"""Checks for background/table consistency and correct sampling-unit aggregation."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import tifffile

from colocalize.background import compute_background
from colocalize.colocalize import measure_masks
from colocalize.datasets import SignalChannel, ReferenceSet
from colocalize.readers import MicroscopyImage
from colocalize import slide_figures as slides


def test_summary_weights_images_equally_and_preserves_unpaired_ids():
    rows = pd.DataFrame({
        "source": ["a"] * 3 + ["b", "c", "d"],
        "reference_set": ["ref"] * 6,
        "cell_id": [1, 2, 3, 1, 1, 1],
        "signal_GD_corrected_median": [0., 0., 0., 10., 20., 30.],
    })
    metadata = pd.DataFrame({"source": ["a", "b", "c", "d"],
                             "sample_id": ["1", "1", "1", "2"],
                             "group": ["L treatment", "L treatment", "R treatment", "R treatment"]})
    images, samples = slides.summarize_for_slides(rows, metadata, "GD", "ref")
    assert len(images) == 4
    assert samples.set_index(["sample_id", "group"]).loc[("1", "L treatment"), "value"] == 5.
    assert samples.set_index(["sample_id", "group"]).loc[("2", "R treatment"), "value"] == 30.
    metadata.loc[0, "sample_id"] = None
    _, samples = slides.summarize_for_slides(rows, metadata, "GD", "ref")
    assert samples.empty


def test_stale_background_is_rejected():
    rows = pd.DataFrame({"signal_GD_bg_level": [20.], "signal_GD_bg_sigma": [3.]})
    with pytest.raises(ValueError, match="Re-run run_analysis"):
        slides._check_saved_background(rows, "GD", 50., 3.)


def test_signal_examples_use_categories_and_typical_cells():
    masks = np.zeros((20, 20), dtype=int)
    for label in range(1, 10):
        masks[2, label + 1] = label
    masks[0, 0] = 10  # A truncated cell is ineligible even if otherwise typical.
    data = SimpleNamespace(signal_name="GD", masks=masks, rows=pd.DataFrame({
        "cell_id": list(range(1, 11)),
        "signal_GD_frac_above_z3": [0., .005, .01, .02, .1, .24, .25, .5, .9, .1],
    }))
    selected = slides._select_signal_examples(data, (.01, .25))
    assert [row.cell_id for row in selected.values()] == [2, 5, 8]
    data.rows = data.rows.loc[data.rows.cell_id.isin([1, 3, 4, 7, 10])]
    selected = slides._select_signal_examples(data, (.01, .25))
    assert selected["No detectable signal"].cell_id == 1
    assert selected["Some signal"].cell_id == 4
    assert selected["Heavy signal"].cell_id == 7
    data.rows = data.rows.loc[data.rows.cell_id.eq(7)]
    selected = slides._select_signal_examples(data, (.01, .25))
    assert selected["No detectable signal"] is None
    assert selected["Some signal"] is None
    assert selected["Heavy signal"].cell_id == 7
    with pytest.raises(ValueError, match="cutoffs"):
        slides._select_signal_examples(data, (.5, .1))


def test_workflow_and_result_exports_with_saved_sparse_masks(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    shape = (128, 128)
    signal = rng.normal(100., 12., shape)
    reference = rng.normal(10., 1., shape)
    masks = np.zeros(shape, dtype=np.int32)
    for label, (y, x) in zip([2, 12, 100], [(20, 20), (55, 55), (90, 90)]):
        masks[y:y+12, x:x+12] = label
        signal[y:y+12, x:x+12] += label / 2
        reference[y:y+12, x:x+12] += 100
    dapi = np.full(shape, 5.)
    spec = SignalChannel(
        name="GD",
        channel=0,
        background_method="tissue_filtered_oop",
        background_dapi_threshold=0,
    )
    ref = ReferenceSet(name="ref", channel=1)
    cells = measure_masks(reference_image=reference, masks=masks,
                          signal_data={"GD": (signal, spec)},
                          backgrounds={"GD": compute_background(signal, masks, dapi, reference, spec)})
    source = "100_100_L_merged.tif"
    cells.insert(0, "reference_set", "ref")
    cells.insert(0, "source", source)
    cells["timepoint"] = "Day_3"
    path = tmp_path / source
    acquisition = MicroscopyImage(path, np.stack([signal, reference, dapi]), ("GD", "ref", "DAPI"))
    config = SimpleNamespace(reference_sets=[ref], signal_channels=[spec],
                             tissue_channel=SimpleNamespace(channel=2), output_dir=tmp_path)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(mask_dir / "100_100_L_merged__ref_masks.tif", masks)
    monkeypatch.setattr(slides, "build_dataset", lambda config, paths=None:
                        SimpleNamespace(paths=[path]) if paths is None else [acquisition])
    data = slides.prepare_workflow_image(config, source, cells, roi=(0,128,0,128))
    assert data.roi == (0, 128, 0, 128)  # Explicit ROIs bypass the default jitter.
    central = slides.prepare_workflow_image(config, source, cells,
                                           crop_size=64, crop_jitter=0)
    assert central.roi == (28, 92, 28, 92)
    jittered_rois = []
    for seed in range(5):
        jittered = slides.prepare_workflow_image(config, source, cells,
                                                crop_size=64, crop_jitter=20,
                                                random_seed=seed)
        y0, y1, x0, x1 = jittered.roi
        assert y1 - y0 == x1 - x0 == 64
        assert abs(y0 - central.roi[0]) <= 20
        assert abs(x0 - central.roi[2]) <= 20
        jittered_rois.append(jittered.roi)
    assert len(set(jittered_rois)) > 1
    repeated = slides.prepare_workflow_image(config, source, cells,
                                            crop_size=64, crop_jitter=20,
                                            random_seed=0)
    assert repeated.roi == jittered_rois[0]
    bounded = slides.prepare_workflow_image(config, source, cells,
                                           crop_size=64, random_seed=0)
    y0, y1, x0, x1 = bounded.roi
    assert 0 <= y0 < y1 <= 128 and 0 <= x0 < x1 <= 128
    assert y1 - y0 == x1 - x0 == 64
    full = slides.prepare_workflow_image(config, source, cells, random_seed=0)
    assert full.roi == (0, 128, 0, 128)  # Smaller images cannot shift.
    with pytest.raises(ValueError, match="crop_jitter"):
        slides.prepare_workflow_image(config, source, cells, crop_jitter=-1)
    exact = compute_background(signal, masks, dapi, reference, spec)
    assert data.background == exact["background"]
    retained = signal[exact["tissue_for_background"]]
    assert data.background_p20 == np.percentile(retained, 20)
    output = tmp_path / "exports"
    paths = slides.export_workflow_figures(data, output, dpi=60)
    assert len(paths) == 6
    example_rows = pd.read_csv(paths[0].parent / "slide_10_example_cell_values.csv")
    for _, row in example_rows.iterrows():
        yy, xx = np.nonzero(masks == row.cell_id)
        assert row.crop_y0 == max(0, yy.min() - 32)
        assert row.crop_y1 == min(shape[0], yy.max() + 33)
        assert row.crop_x0 == max(0, xx.min() - 32)
        assert row.crop_x1 == min(shape[1], xx.max() + 33)
    control_path, summary = slides.export_comparison_figure(data, output, dpi=60)
    raw_median = cells.signal_GD_median
    assert summary["median_cell_median_p20"] == np.median(raw_median-data.background_p20)
    result_paths, images, samples = slides.export_result_figures(
        cells, slides.source_metadata(cells), output, signal_name="GD", reference_name="ref",
        representative=data, dpi=60)
    assert len(result_paths) == 3
    assert len(samples) == 1
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths + [control_path] + result_paths)
    with pytest.raises(ValueError, match="ROI"):
        slides.prepare_workflow_image(config, source, cells, roi=(0,129,0,128))
