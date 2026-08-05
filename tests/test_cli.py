from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from colocalize.cli import (
    build_parser,
    config_from_args,
    load_config,
    save_or_show_segmentations,
)
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


def test_cli_without_json_uses_complete_defaults():
    config = config_from_args(build_parser().parse_args([]))

    assert config.input_dir == Path(".")
    assert config.output_dir == Path("results")
    assert config.reference_sets == [
        ReferenceSet(name="reference", channel=0, model="cpdino_BRN3A")
    ]
    assert config.signal_channels == [SignalChannel(name="signal", channel=1)]
    assert config.z_projection == "max"
    assert config.device == "auto"
    assert config.save_masks is True


def test_cli_accepts_all_serializable_config_fields(tmp_path):
    config = config_from_args(
        build_parser().parse_args(
            [
                "--input-dir",
                str(tmp_path / "images"),
                "--output-dir",
                str(tmp_path / "results"),
                "--extensions",
                ".oir",
                ".tif",
                "--recursive",
                "--exclude",
                "calibration/*",
                "--reference-name",
                "RPBMS",
                "--reference-channel",
                "3",
                "--model",
                "cpdino_RPBMS",
                "--diameter",
                "24",
                "--flow-threshold",
                "0.5",
                "--cellprob-threshold",
                "-0.2",
                "--min-size",
                "20",
                "--no-normalize",
                "--signal-name",
                "GD",
                "--signal-channel",
                "Marker",
                "--threshold-method",
                "percentile",
                "--threshold-value",
                "99.5",
                "--positive-fraction-cutoff",
                "0.05",
                "--time-index",
                "2",
                "--scene-index",
                "1",
                "--z-projection",
                "mean",
                "--device",
                "cuda:1",
                "--no-save-masks",
            ]
        )
    )

    assert config.input_dir == tmp_path / "images"
    assert config.output_dir == tmp_path / "results"
    assert config.extensions == (".oir", ".tif")
    assert config.recursive is True
    assert config.exclude == ("calibration/*",)
    assert config.reference_sets == [
        ReferenceSet(
            name="RPBMS",
            channel=3,
            model="cpdino_RPBMS",
            diameter=24,
            flow_threshold=0.5,
            cellprob_threshold=-0.2,
            min_size=20,
            normalize=False,
        )
    ]
    assert config.signal_channels == [
        SignalChannel(
            name="GD",
            channel="Marker",
            threshold_method="percentile",
            threshold_value=99.5,
            positive_fraction_cutoff=0.05,
        )
    ]
    assert config.time_index == 2
    assert config.scene_index == 1
    assert config.z_projection == "mean"
    assert config.device == "cuda:1"
    assert config.save_masks is False


def test_cli_options_override_json_values(tmp_path):
    config_path = tmp_path / "analysis.json"
    config_path.write_text(
        json.dumps(
            {
                "input_dir": "images",
                "output_dir": "results",
                "reference_sets": [{"name": "cells", "channel": 0}],
                "signal_channels": [{"name": "marker", "channel": 1}],
            }
        ),
        encoding="utf-8",
    )

    config = config_from_args(
        build_parser().parse_args(
            [str(config_path), "--device", "cpu", "--reference-channel", "DAPI"]
        )
    )

    assert config.input_dir == tmp_path / "images"
    assert config.device == "cpu"
    assert config.reference_sets[0].channel == "DAPI"


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
