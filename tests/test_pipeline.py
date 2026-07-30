from __future__ import annotations

from pathlib import Path

from colocalize.datasets import AnalysisConfig, ReferenceSet, SignalChannel
from colocalize.pipeline import _input_paths


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
