from pathlib import Path

import numpy as np
import pytest

from colocalize.datasets import AnalysisConfig, ReferenceSet, SignalChannel
from colocalize.io import save_mask
from colocalize.pipeline import _get_masks, run_analysis
from colocalize.readers import MicroscopyImage


@pytest.mark.parametrize("external_mask_dir", [False, True])
def test_get_masks_loads_named_tiff_from_mask_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, external_mask_dir: bool
) -> None:
    output_mask_dir = tmp_path / "output" / "masks"
    source_dir = tmp_path / "existing_masks" if external_mask_dir else output_mask_dir
    reference = ReferenceSet(name="BRN3A", channel=0)
    acquisition = MicroscopyImage(
        path=tmp_path / "retina.ome.tif",
        data=np.zeros((2, 2, 2)),
        channel_names=("reference", "signal"),
    )
    expected = np.array([[0, 1], [2, 2]], dtype=np.int32)
    save_mask(
        masks=expected,
        acquisition_path=acquisition.path,
        reference=reference,
        output_dir=source_dir,
    )
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        reference_sets=[reference],
        signal_channels=[SignalChannel(name="signal", channel=1)],
        load_masks=True,
        mask_dir=source_dir if external_mask_dir else None,
    )

    def unexpected_segmenter(*args, **kwargs):
        raise AssertionError("Existing masks should be loaded without segmenting")

    monkeypatch.setattr("colocalize.pipeline.CellposeSegmenter", unexpected_segmenter)

    masks, segmenter = _get_masks(output_mask_dir, acquisition, reference, config)

    np.testing.assert_array_equal(masks, expected)
    assert segmenter is None


def test_loaded_mask_analysis_does_not_touch_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "retina.tif"
    source.touch()
    acquisition = MicroscopyImage(
        path=source,
        data=np.ones((2, 4, 4), dtype=float),
        channel_names=("reference", "signal"),
    )
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    reference = ReferenceSet(name="BRN3A", channel=0)
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        reference_sets=[reference],
        signal_channels=[SignalChannel(name="signal", channel=1)],
        load_masks=True,
        save_masks=False,
        progress=False,
    )

    monkeypatch.setattr("colocalize.pipeline._input_paths", lambda config: [source])
    monkeypatch.setattr("colocalize.pipeline.build_dataset", lambda *args: [acquisition])
    monkeypatch.setattr(
        "colocalize.pipeline._get_masks", lambda **kwargs: (masks, None)
    )

    def unexpected_cuda_call():
        raise AssertionError("Loading masks should not initialize or touch CUDA")

    monkeypatch.setattr("colocalize.pipeline._free_gpu_cache", unexpected_cuda_call)

    result = run_analysis(config)

    assert len(result.cells) == 1


def test_run_analysis_can_promote_2d_channels_to_loaded_3d_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "retina.tif"
    source.touch()
    acquisition = MicroscopyImage(
        path=source,
        data=np.stack(
            [
                np.full((4, 4), 10.0),
                np.full((4, 4), 20.0),
            ]
        ),
        channel_names=("reference", "signal"),
    )
    masks = np.zeros((3, 4, 4), dtype=np.int32)
    masks[:, 1:3, 1:3] = 1
    reference = ReferenceSet(name="BRN3A", channel=0)
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        reference_sets=[reference],
        signal_channels=[SignalChannel(name="signal", channel=1)],
        load_masks=True,
        save_masks=False,
        progress=False,
        promote_2d_to_3d=True,
    )

    monkeypatch.setattr("colocalize.pipeline._input_paths", lambda config: [source])
    monkeypatch.setattr("colocalize.pipeline.build_dataset", lambda *args: [acquisition])
    monkeypatch.setattr(
        "colocalize.pipeline._get_masks", lambda **kwargs: (masks, None)
    )

    result = run_analysis(config)

    assert len(result.cells) == 1
    assert "volume_voxels" in result.cells.columns


def test_run_analysis_rejects_2d_channels_with_loaded_3d_masks_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "retina.tif"
    source.touch()
    acquisition = MicroscopyImage(
        path=source,
        data=np.stack(
            [
                np.full((4, 4), 10.0),
                np.full((4, 4), 20.0),
            ]
        ),
        channel_names=("reference", "signal"),
    )
    masks = np.zeros((3, 4, 4), dtype=np.int32)
    masks[:, 1:3, 1:3] = 1
    reference = ReferenceSet(name="BRN3A", channel=0)
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        reference_sets=[reference],
        signal_channels=[SignalChannel(name="signal", channel=1)],
        load_masks=True,
        save_masks=False,
        progress=False,
    )

    monkeypatch.setattr("colocalize.pipeline._input_paths", lambda config: [source])
    monkeypatch.setattr("colocalize.pipeline.build_dataset", lambda *args: [acquisition])
    monkeypatch.setattr(
        "colocalize.pipeline._get_masks", lambda **kwargs: (masks, None)
    )

    with pytest.raises(ValueError, match="promote_2d_to_3d"):
        run_analysis(config)
