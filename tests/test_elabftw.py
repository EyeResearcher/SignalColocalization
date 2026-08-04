from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from colocalize.datasets import AnalysisResult
from colocalize.elabftw import (
    DownloadedUpload,
    ElabClient,
    ElabError,
    parse_metadata,
    write_workbook,
)


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


def test_list_experiments_always_requests_extended(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        return _Response(b"[]")

    monkeypatch.setattr("colocalize.elabftw.urlopen", fake_urlopen)
    client = ElabClient(api_key="secret", base_url="https://example.test/api/v2")

    assert client.list_experiments(limit=1) == []
    assert captured["url"] == "https://example.test/api/v2/experiments?extended=1&limit=1"
    assert captured["authorization"] == "secret"


def test_download_uploads_filters_extensions_and_preserves_provenance(tmp_path, monkeypatch):
    client = ElabClient(api_key="secret", base_url="https://example.test/api/v2")
    monkeypatch.setattr(
        client,
        "list_uploads",
        lambda experiment_id, **kwargs: [
            {"id": 11, "real_name": "stored-11", "long_name": "retina.ome.tif", "comment": "raw image"},
            {"id": 12, "real_name": "stored-12", "long_name": "notes.pdf"},
        ],
    )
    monkeypatch.setattr(client, "get_record", lambda entity_type, record_id: {"id": record_id})
    monkeypatch.setattr(client, "_bytes", lambda endpoint, params: b"image bytes")

    uploads = client.download_uploads(42, tmp_path)

    assert len(uploads) == 1
    assert uploads[0].upload_id == 11
    assert uploads[0].original_name == "retina.ome.tif"
    assert uploads[0].local_path.read_bytes() == b"image bytes"


def test_download_uploads_reports_requested_non_image_id(tmp_path, monkeypatch):
    client = ElabClient(api_key="secret", base_url="https://example.test/api/v2")
    monkeypatch.setattr(
        client,
        "list_uploads",
        lambda experiment_id, **kwargs: [{"id": 12, "long_name": "notes.pdf"}],
    )
    monkeypatch.setattr(client, "get_record", lambda entity_type, record_id: {"id": record_id})

    with pytest.raises(ElabError, match="not found or were not supported"):
        client.download_uploads(42, tmp_path, upload_ids=[12])


def test_parse_metadata_accepts_elab_json_string():
    metadata = parse_metadata(
        {"metadata": json.dumps({"extra_fields": {"Animal": {"value": "A-123"}}})}
    )

    assert metadata["extra_fields"]["Animal"]["value"] == "A-123"


def test_write_workbook_contains_analysis_and_provenance(tmp_path):
    result = AnalysisResult(
        cells=pd.DataFrame([{"source": "retina.tif", "cell_id": 1}]),
        images=pd.DataFrame([{"source": "retina.tif", "cell_count": 1}]),
    )
    upload = DownloadedUpload(7, "Original Retina.tif", Path("retina.tif"), "raw")
    destination = tmp_path / "analysis.xlsx"

    write_workbook(
        destination,
        result=result,
        records=[
            (
                "experiments",
                {
                    "id": 42,
                    "title": "Retina experiment",
                    "metadata": json.dumps(
                        {"extra_fields": {"Animal": {"value": "A-123", "type": "text", "group_id": 2}}}
                    ),
                },
            )
        ],
        sources=[upload],
        base_url="https://example.test/api/v2",
    )

    workbook = pd.ExcelFile(destination)
    assert workbook.sheet_names == [
        "Run info",
        "Records",
        "Record fields",
        "Source images",
        "Image summary",
        "Cells",
    ]
    cells = pd.read_excel(destination, sheet_name="Cells")
    assert cells.loc[0, "upload_id"] == 7
    assert cells.loc[0, "original_name"] == "Original Retina.tif"


def test_connection_error_mentions_vpn_without_leaking_key(monkeypatch):
    monkeypatch.setattr(
        "colocalize.elabftw.urlopen", lambda request, timeout: (_ for _ in ()).throw(URLError("offline"))
    )
    client = ElabClient(api_key="top-secret", base_url="https://example.test/api/v2")

    with pytest.raises(ElabError, match="VPN") as error:
        client.list_experiments()
    assert "top-secret" not in str(error.value)
