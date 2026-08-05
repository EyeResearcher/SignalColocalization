from __future__ import annotations

import json

import pytest

from colocalize.elabftw import ElabError
from colocalize.job import job_schema_path, load_job


def _payload():
    return {
        "schema_version": 1,
        "sources": ["experiment:42", "upload:9"],
        "analysis": {
            "output_dir": "results",
            "reference_sets": [
                {
                    "name": "RPBMS",
                    "channel": 3,
                    "model": "cpdino_RPBMS",
                    "diameter": None,
                }
            ],
            "signal_channels": [
                {
                    "name": "GD",
                    "channel": "GD",
                    "threshold_method": "percentile",
                    "threshold_value": 99.5,
                    "positive_fraction_cutoff": 0.05,
                }
            ],
            "extensions": [".oir", ".ome.tif"],
            "time_index": 0,
            "scene_index": 0,
            "z_projection": "max",
            "device": "cuda:1",
            "save_masks": True,
        },
        "execution": {"inspect": False, "save_segmentation": True},
    }


def test_load_job_builds_complete_interface_request(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    job = load_job(path)

    assert job.sources == ["experiment:42", "upload:9"]
    assert job.analysis.output_dir == tmp_path / "results"
    assert job.analysis.reference_sets[0].model == "cpdino_RPBMS"
    assert job.analysis.signal_channels[0].channel == "GD"
    assert job.analysis.device == "cuda:1"
    assert job.save_segmentation is True


def test_load_job_rejects_unknown_interface_fields(tmp_path):
    payload = _payload()
    payload["api_key"] = "must-not-be-accepted"
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ElabError, match="Unknown job field"):
        load_job(path)


def test_load_job_enforces_conditional_absolute_threshold(tmp_path):
    payload = _payload()
    payload["analysis"]["signal_channels"][0].update(
        threshold_method="absolute", threshold_value=None
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ElabError, match="requires threshold_value"):
        load_job(path)


def test_distributable_schema_is_valid_json():
    schema = json.loads(job_schema_path().read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == 1
