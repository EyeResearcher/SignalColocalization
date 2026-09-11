"""Read-only helpers for running colocalization from eLabFTW uploads."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .datasets import AnalysisResult
from .readers import SUPPORTED_EXTENSIONS


DEFAULT_ELAB_BASE = "https://johnsonlab.wilmer.jhu.edu/api/v2"


class ElabError(RuntimeError):
    """A safe, user-facing eLabFTW API error."""


@dataclass(frozen=True)
class DownloadedUpload:
    """Provenance for one resolved input image."""

    upload_id: int | None
    original_name: str
    local_path: Path
    comment: str = ""
    source_type: str = "upload"
    source_pointer: str = ""
    record_type: str = ""
    record_id: int | None = None
    record_title: str = ""


class ElabClient:
    """Small read-only client for the eLabFTW endpoints used by this tool."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Configure the eLabFTW HTTP client.

        Args:
            api_key: Personal access token for the eLabFTW REST API.  Reads
                ``ELAB_APIKEY`` from the environment when not supplied directly.
            base_url: Root URL of the eLabFTW instance (e.g.
                ``https://elabftw.example.org``).  Reads ``ELAB_BASE`` from the
                environment when not supplied directly.
            timeout: Socket timeout in seconds for every HTTP request.

        Raises:
            ElabError: If no API key is available after checking both the
                argument and the ``ELAB_APIKEY`` environment variable.
        """
        self.api_key = api_key or os.environ.get("ELAB_APIKEY", "")
        self.base_url = (base_url or os.environ.get("ELAB_BASE") or DEFAULT_ELAB_BASE).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ElabError(
                "ELAB_APIKEY is not set. Create an eLabFTW API key and expose it "
                "as an environment variable; do not put it in the config file."
            )

    def list_experiments(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """List accessible experiments, always requesting extended metadata."""
        params: dict[str, Any] = {"extended": 1}
        if limit is not None:
            params["limit"] = limit
        payload = self._json("experiments", params=params)
        if not isinstance(payload, list):
            raise ElabError("eLabFTW returned an unexpected experiments response.")
        return payload

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        """Fetch a single experiment record by numeric ID."""
        return self.get_record("experiments", experiment_id)

    def get_record(self, entity_type: str, record_id: int) -> dict[str, Any]:
        """Fetch a single record of any entity type by numeric ID.

        Args:
            entity_type: eLabFTW entity type string, e.g. ``'experiments'`` or
                ``'items'``.
            record_id: Numeric identifier of the record.

        Returns:
            The API payload as a parsed dict.

        Raises:
            ElabError: If the API returns a non-dict response.
        """
        entity_type = _entity_type(entity_type)
        payload = self._json(f"{entity_type}/{record_id}")
        if not isinstance(payload, dict):
            raise ElabError(f"eLabFTW returned an unexpected {entity_type} response.")
        return payload

    def list_uploads(
        self, record_id: int | None = None, *, entity_type: str = "experiments"
    ) -> list[dict[str, Any]]:
        """List file uploads attached to a record, or all uploads if no record is given.

        Args:
            record_id: Numeric identifier of the experiment or resource.
                When ``None``, lists all uploads accessible to the authenticated
                user.
            entity_type: eLabFTW entity type owning the uploads.

        Returns:
            List of upload metadata dicts as returned by the API.

        Raises:
            ElabError: If the API returns a non-list response.
        """
        endpoint = "uploads"
        if record_id is not None:
            endpoint = f"{_entity_type(entity_type)}/{record_id}/uploads"
        payload = self._json(endpoint)
        if not isinstance(payload, list):
            raise ElabError("eLabFTW returned an unexpected uploads response.")
        return payload

    def download_uploads(
        self,
        experiment_id: int,
        destination: str | Path,
        *,
        upload_ids: Iterable[int] | None = None,
        entity_type: str = "experiments",
        used_names: set[str] | None = None,
        exclude_upload_ids: Iterable[int] | None = None,
        record: dict[str, Any] | None = None,
        extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    ) -> list[DownloadedUpload]:
        """Download supported microscopy attachments from an experiment/resource."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        selected_ids = set(upload_ids or ())
        excluded_ids = set(exclude_upload_ids or ())
        downloaded: list[DownloadedUpload] = []
        used_names = used_names if used_names is not None else set()
        entity_type = _entity_type(entity_type)
        record = record or self.get_record(entity_type, experiment_id)

        for upload in self.list_uploads(experiment_id, entity_type=entity_type):
            upload_id = _upload_id(upload)
            if upload_id in excluded_ids:
                continue
            if selected_ids and upload_id not in selected_ids:
                continue
            original_name = _upload_name(upload)
            if not _is_supported_image(original_name, extensions):
                continue
            safe_name = _unique_filename(original_name, upload_id, used_names)
            local_path = destination / safe_name
            local_path.write_bytes(self._bytes(f"uploads/{upload_id}", params={"format": "binary"}))
            downloaded.append(
                DownloadedUpload(
                    upload_id=upload_id,
                    original_name=original_name,
                    local_path=local_path,
                    comment=str(upload.get("comment") or ""),
                    source_pointer=f"{entity_type[:-1]}:{experiment_id}",
                    record_type=entity_type,
                    record_id=experiment_id,
                    record_title=str(record.get("title") or ""),
                )
            )

        if selected_ids:
            found = {item.upload_id for item in downloaded}
            missing = selected_ids - found
            if missing:
                raise ElabError(
                    "Requested upload IDs were not found or were not supported microscopy "
                    f"files: {sorted(missing)}"
                )
        return downloaded

    def download_upload_ids(
        self,
        upload_ids: Iterable[int],
        destination: str | Path,
        *,
        used_names: set[str] | None = None,
        extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    ) -> list[DownloadedUpload]:
        """Download individual uploads selected independently of a parent record."""
        selected = set(upload_ids)
        if not selected:
            return []
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        used_names = used_names if used_names is not None else set()
        downloaded = []
        for upload in self.list_uploads():
            upload_id = _upload_id(upload)
            if upload_id not in selected:
                continue
            original_name = _upload_name(upload)
            if not _is_supported_image(original_name, extensions):
                continue
            local_path = destination / _unique_filename(original_name, upload_id, used_names)
            local_path.write_bytes(self._bytes(f"uploads/{upload_id}", params={"format": "binary"}))
            downloaded.append(
                DownloadedUpload(
                    upload_id=upload_id,
                    original_name=original_name,
                    local_path=local_path,
                    comment=str(upload.get("comment") or ""),
                    source_pointer=f"upload:{upload_id}",
                )
            )
        missing = selected - {source.upload_id for source in downloaded}
        if missing:
            raise ElabError(
                "Requested upload IDs were not found or were not supported microscopy "
                f"files: {sorted(missing)}"
            )
        return downloaded

    def _json(self, endpoint: str, *, params: dict[str, Any] | None = None) -> Any:
        payload = self._request(endpoint, params=params, accept="application/json")
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ElabError("eLabFTW returned invalid JSON.") from exc

    def _bytes(self, endpoint: str, *, params: dict[str, Any] | None = None) -> bytes:
        return self._request(endpoint, params=params, accept="application/octet-stream")

    def _request(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None,
        accept: str,
    ) -> bytes:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={"Authorization": self.api_key, "Accept": accept},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.reason or "request failed"
            raise ElabError(f"eLabFTW request failed ({exc.code}: {detail}).") from exc
        except (URLError, TimeoutError) as exc:
            raise ElabError(
                "Could not reach eLabFTW. Confirm the JHU VPN is connected and "
                "ELAB_BASE is correct."
            ) from exc


def parse_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize eLabFTW's JSON-string metadata field to a dictionary."""
    metadata = record.get("metadata")
    if not metadata:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise ElabError("The experiment metadata field is not valid JSON.") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ElabError("The experiment metadata field has an unexpected format.")


def write_workbook(
    destination: str | Path,
    *,
    result: AnalysisResult,
    records: list[tuple[str, dict[str, Any]]],
    sources: list[DownloadedUpload],
    base_url: str,
) -> Path:
    """Write analysis tables and flexible source provenance to an Excel workbook."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    run_info = pd.DataFrame(
        [
            {"field": "input_image_count", "value": len(sources)},
            {"field": "eLabFTW_record_count", "value": len(records)},
            {"field": "eLabFTW_base", "value": base_url},
        ]
    )
    source_table = pd.DataFrame(
        [
            {
                "source": source.local_path.name,
                "original_name": source.original_name,
                "source_type": source.source_type,
                "source_pointer": source.source_pointer,
                "upload_id": source.upload_id,
                "record_type": source.record_type,
                "record_id": source.record_id,
                "record_title": source.record_title,
                "comment": source.comment,
            }
            for source in sources
        ]
    )
    record_rows = []
    extra_field_rows = []
    for record_type, record in records:
        record_id = record.get("id", "")
        record_rows.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "title": record.get("title", ""),
                "date": record.get("date", ""),
                "api_url": f"{base_url}/{record_type}/{record_id}",
            }
        )
        extra_fields = parse_metadata(record).get("extra_fields") or {}
        if isinstance(extra_fields, dict):
            for name, details in extra_fields.items():
                details = details if isinstance(details, dict) else {"value": details}
                extra_field_rows.append(
                    {
                        "record_type": record_type,
                        "record_id": record_id,
                        "name": name,
                        "value": details.get("value", ""),
                        "type": details.get("type", ""),
                        "group_id": details.get("group_id", ""),
                    }
                )

    provenance = {row["source"]: row for row in source_table.to_dict("records")}
    cells = _with_upload_provenance(result.cells, provenance)
    images = _with_upload_provenance(result.images, provenance)

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        run_info.to_excel(writer, sheet_name="Run info", index=False)
        pd.DataFrame(
            record_rows, columns=["record_type", "record_id", "title", "date", "api_url"]
        ).to_excel(writer, sheet_name="Records", index=False)
        pd.DataFrame(
            extra_field_rows,
            columns=["record_type", "record_id", "name", "value", "type", "group_id"],
        ).to_excel(
            writer, sheet_name="Record fields", index=False
        )
        source_table.to_excel(writer, sheet_name="Source images", index=False)
        images.to_excel(writer, sheet_name="Image summary", index=False)
        cells.to_excel(writer, sheet_name="Cells", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.columns:
                values = [str(cell.value or "") for cell in column[:200]]
                width = min(max((len(value) for value in values), default=8) + 2, 60)
                worksheet.column_dimensions[column[0].column_letter].width = width
    return destination


def _with_upload_provenance(
    table: pd.DataFrame, provenance: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    table = table.copy()
    if "source" not in table.columns:
        return table
    columns = ("original_name", "source_type", "source_pointer", "upload_id", "record_type", "record_id")
    for offset, column in enumerate(columns, start=1):
        table.insert(
            offset,
            column,
            table["source"].map(
                lambda name, key=column: provenance.get(name, {}).get(
                    key, name if key == "original_name" else None
                )
            ),
        )
    return table


def _entity_type(value: str) -> str:
    aliases = {"experiment": "experiments", "experiments": "experiments", "item": "items", "items": "items", "resource": "items", "resources": "items"}
    try:
        return aliases[value.casefold()]
    except KeyError as exc:
        raise ElabError(f"Unsupported eLabFTW record type: {value!r}.") from exc


def _upload_id(upload: dict[str, Any]) -> int:
    try:
        return int(upload["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ElabError("An eLabFTW upload is missing a valid ID.") from exc


def _upload_name(upload: dict[str, Any]) -> str:
    # eLabFTW exposes the user-facing attachment name as long_name and may use
    # real_name for its storage-side name. Prefer the former for extension
    # filtering and workbook provenance.
    for key in ("long_name", "real_name", "filename", "name"):
        value = upload.get(key)
        if value:
            return Path(str(value)).name
    raise ElabError(f"eLabFTW upload {_upload_id(upload)} is missing a filename.")


def _is_supported_image(filename: str, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> bool:
    folded = filename.casefold()
    return any(folded.endswith(extension.casefold()) for extension in extensions)


def _unique_filename(filename: str, upload_id: int, used: set[str]) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    safe = safe or f"upload_{upload_id}"
    candidate = safe
    if candidate.casefold() in used:
        suffixes = "".join(Path(safe).suffixes)
        stem = safe[: -len(suffixes)] if suffixes else safe
        candidate = f"{stem}__upload_{upload_id}{suffixes}"
    used.add(candidate.casefold())
    return candidate
