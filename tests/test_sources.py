from __future__ import annotations

from pathlib import Path

import pytest

from colocalize.elabftw import DownloadedUpload, ElabError
from colocalize.sources import parse_pointer, resolve_sources


def test_parse_pointer_accepts_record_upload_and_modern_urls():
    assert parse_pointer("experiment:42") == ("experiments", "42")
    assert parse_pointer("resource:9") == ("items", "9")
    assert parse_pointer("upload:7") == ("upload", "7")
    assert parse_pointer("https://lab.test/api/v2/experiments/42") == ("experiments", "42")
    assert parse_pointer("https://lab.test/items/9") == ("items", "9")


def test_parse_pointer_accepts_legacy_elab_urls():
    assert parse_pointer("https://lab.test/experiments.php?mode=view&id=42") == (
        "experiments",
        "42",
    )
    assert parse_pointer("https://lab.test/database.php?mode=view&id=9") == ("items", "9")


def test_resolve_sources_combines_directory_glob_and_manifest(tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    first = group / "first.tif"
    second = group / "second.oir"
    ignored = group / "notes.txt"
    for path in (first, second, ignored):
        path.write_bytes(b"test")
    third = tmp_path / "third.czi"
    third.write_bytes(b"test")
    manifest = tmp_path / "sources.txt"
    manifest.write_text("# mixed sources\ndir:group\nfile:third.czi\n", encoding="utf-8")

    resolved = resolve_sources([f"@{manifest}"], tmp_path / "staged")

    assert [image.original_name for image in resolved.images] == ["first.tif", "second.oir", "third.czi"]
    assert all(image.local_path.is_file() for image in resolved.images)
    assert resolved.client is None


def test_resolve_sources_deduplicates_overlapping_local_pointers(tmp_path):
    image = tmp_path / "same.tif"
    image.write_bytes(b"test")

    resolved = resolve_sources([str(image), f"glob:{tmp_path / '*.tif'}"], tmp_path / "staged")

    assert len(resolved.images) == 1


class _Client:
    base_url = "https://lab.test/api/v2"

    def get_record(self, entity_type, record_id):
        return {"id": record_id, "title": "Record"}

    def download_uploads(self, record_id, destination, **kwargs):
        path = Path(destination) / "record.tif"
        path.write_bytes(b"test")
        return [
            DownloadedUpload(
                10,
                "record.tif",
                path,
                record_type=kwargs["entity_type"],
                record_id=record_id,
            )
        ]

    def download_upload_ids(self, upload_ids, destination, **kwargs):
        path = Path(destination) / "direct.oir"
        path.write_bytes(b"test")
        return [DownloadedUpload(upload_ids[0], "direct.oir", path)]


def test_resolve_sources_combines_elab_record_and_direct_upload(tmp_path):
    resolved = resolve_sources(
        ["experiment:42", "upload:11"], tmp_path / "staged", client=_Client()
    )

    assert len(resolved.images) == 2
    assert resolved.records == [("experiments", {"id": 42, "title": "Record"})]


def test_manifest_cycle_is_rejected(tmp_path):
    manifest = tmp_path / "cycle.txt"
    manifest.write_text("@cycle.txt\n", encoding="utf-8")

    with pytest.raises(ElabError, match="cycle"):
        resolve_sources([f"@{manifest}"], tmp_path / "staged")
