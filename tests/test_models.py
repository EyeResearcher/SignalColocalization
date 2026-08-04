from __future__ import annotations

import sys
import types

import pytest

from colocalize import models


def test_known_retinal_model_downloads_from_configured_hub_repo(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(models, "_cellpose_model_dir", lambda: tmp_path / "cellpose")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )

    resolved = models.resolve_model_source("cpdino_RPBMS")

    assert resolved == str(tmp_path / "cpdino_RPBMS")
    assert calls == [
        {
            "repo_id": "mmzinn12/cellpose-retinal-models",
            "filename": "cpdino_RPBMS",
            "repo_type": "model",
        }
    ]


def test_explicit_hf_uri_supports_nested_model_path(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "downloaded-model")

    monkeypatch.setattr(models, "_cellpose_model_dir", lambda: tmp_path / "cellpose")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )

    resolved = models.resolve_model_source("hf://owner/repo/models/model-file")

    assert resolved == str(tmp_path / "downloaded-model")
    assert calls[0]["repo_id"] == "owner/repo"
    assert calls[0]["filename"] == "models/model-file"


def test_existing_cellpose_cache_is_preferred(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "cpdino_BRN3A"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(models, "_cellpose_model_dir", lambda: model_dir)

    assert models.resolve_model_source("cpdino_BRN3A") == str(model_path)


def test_cellpose_builtin_name_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "_cellpose_model_dir", lambda: tmp_path)

    assert models.resolve_model_source("cpsam_v2") == "cpsam_v2"


def test_malformed_hf_uri_is_rejected():
    with pytest.raises(ValueError, match="hf://OWNER/REPOSITORY"):
        models.resolve_model_source("hf://owner/repo")
