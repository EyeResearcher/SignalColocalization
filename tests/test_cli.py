from __future__ import annotations

import json

import numpy as np
import tifffile

from colocalize.cli import load_config, save_or_show_segmentations
from colocalize.datasets import AnalysisConfig, ReferenceSet, SignalChannel
from colocalize.readers import MicroscopyImage


def test_load_config_builds_analysis_config_and_resolves_relative_directories(tmp_path):
    config_path = tmp_path / "analysis.json"
    config_path.write_text(
        json.dumps(
            {
                "input_dir": "images",
                "output_dir": "results",
                "reference_sets": [
                    {"name": "cells", "channel": "DAPI", "model": "nuclei"}
                ],
                "signal_channels": [
                    {"name": "marker", "channel": 1, "threshold_method": "otsu"}
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input_dir == tmp_path / "images"
    assert config.output_dir == tmp_path / "results"
    assert config.reference_sets[0].channel == "DAPI"
    assert config.reference_sets[0].model == "nuclei"
    assert config.signal_channels[0].channel == 1


def test_save_segmentation_writes_qc_png(tmp_path, monkeypatch):
    image_path = tmp_path / "example.tif"
    acquisition = MicroscopyImage(
        path=image_path,
        data=np.stack(
            [np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4)]
        ),
        channel_names=("reference", "signal"),
    )
    config = AnalysisConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "results",
        reference_sets=[ReferenceSet(name="cells", channel=0)],
        signal_channels=[SignalChannel(name="marker", channel=1)],
    )
    mask_path = tmp_path / "masks.tif"
    tifffile.imwrite(mask_path, np.asarray([[0, 0, 0, 0], [0, 1, 1, 0]] * 2))
    monkeypatch.setattr("colocalize.cli.build_dataset", lambda config: [acquisition])

    saved = save_or_show_segmentations(config, [mask_path], save=True)

    assert len(saved) == 1
    assert saved[0].is_file()
    assert saved[0].parent == config.output_dir / "segmentation_qc"
