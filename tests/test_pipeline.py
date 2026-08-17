from __future__ import annotations

from pathlib import Path

import numpy as np

from colocalize.datasets import AnalysisConfig, ReferenceSet, SignalChannel
from colocalize.pipeline import _input_paths, run_analysis
from colocalize.readers import MicroscopyImage


def _config(tmp_path: Path, *, exclude=(), recursive: bool = False) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        reference_sets=[ReferenceSet(name="cells", channel=0)],
        signal_channels=[SignalChannel(name="signal", channel=1)],
        exclude=exclude,
        recursive=recursive,
    )


def test_input_paths_exclude_exact_filename_case_insensitively(tmp_path):
    keep = tmp_path / "keep.oir"
    excluded = tmp_path / "Skip.OIR"
    keep.touch()
    excluded.touch()

    paths = _input_paths(_config(tmp_path, exclude="skip.oir"))

    assert paths == [keep]


def test_input_paths_exclude_globs_and_relative_paths(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    keep = nested / "keep.oir"
    globbed = nested / "preview_G013.tif"
    relative = nested / "specific.oir"
    for path in (keep, globbed, relative):
        path.touch()

    paths = _input_paths(
        _config(
            tmp_path,
            exclude=("*_G013.tif", "nested/specific.oir"),
            recursive=True,
        )
    )

    assert paths == [keep]


def test_input_paths_exclude_absolute_path(tmp_path):
    keep = tmp_path / "keep.oir"
    excluded = tmp_path / "skip.oir"
    keep.touch()
    excluded.touch()

    paths = _input_paths(_config(tmp_path, exclude=(excluded,)))

    assert paths == [keep]


def test_run_analysis_saves_grid_and_panels_for_every_input_image(
    tmp_path, monkeypatch
):
    paths = [tmp_path / "first.tif", tmp_path / "second.tif"]
    for path in paths:
        path.touch()
    acquisitions = [
        MicroscopyImage(
            path=path,
            data=np.stack(
                [
                    np.arange(16, dtype=float).reshape(4, 4),
                    np.arange(16, 32, dtype=float).reshape(4, 4),
                ]
            ),
            channel_names=("reference", "signal"),
        )
        for path in paths
    ]

    class FakeSegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment(self, image, **kwargs):
            masks = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0]] * 2)
            return masks, None

    monkeypatch.setattr(
        "colocalize.pipeline.build_dataset",
        lambda config, paths=None: acquisitions,
    )
    monkeypatch.setattr("colocalize.pipeline.CellposeSegmenter", FakeSegmenter)
    qc_dir = tmp_path / "all-qc-images"
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "results",
        reference_sets=[ReferenceSet(name="cells", channel=0)],
        signal_channels=[SignalChannel(name="marker", channel=1)],
        save_masks=False,
        save_segmentation=True,
        segmentation_output_dir=qc_dir,
    )

    result = run_analysis(config)

    assert len(result.segmentation_paths) == 10
    assert all(path.is_file() for path in result.segmentation_paths)
    assert {path.parent for path in result.segmentation_paths} == {qc_dir}
    assert sum(path.name.endswith("__grid.png") for path in result.segmentation_paths) == 2
