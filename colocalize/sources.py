"""Resolve flexible image pointers into one staged analysis directory."""

from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
import re
import shutil
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from .elabftw import DownloadedUpload, ElabClient, ElabError
from .readers import SUPPORTED_EXTENSIONS


@dataclass
class ResolvedSources:
    """Staged images plus the eLabFTW records that supplied them."""

    images: list[DownloadedUpload]
    records: list[tuple[str, dict]]
    client: ElabClient | None = None


def resolve_sources(
    pointers: Iterable[str],
    destination: str | Path,
    *,
    client: ElabClient | None = None,
) -> ResolvedSources:
    """Resolve record, upload, URL, path, glob, and manifest pointers."""
    expanded = _expand_manifests(list(pointers))
    if not expanded:
        raise ElabError("At least one image source pointer is required.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    images: list[DownloadedUpload] = []
    records: list[tuple[str, dict]] = []
    record_keys: set[tuple[str, int]] = set()
    remote_upload_ids: set[int] = set()
    local_paths: set[Path] = set()
    used_names: set[str] = set()

    for pointer in expanded:
        kind, value = parse_pointer(pointer)
        if kind in {"experiments", "items", "upload"}:
            if client is None:
                client = ElabClient()
            if kind == "upload":
                upload_id = int(value)
                if upload_id in remote_upload_ids:
                    continue
                found = client.download_upload_ids([upload_id], destination, used_names=used_names)
                images.extend(found)
                remote_upload_ids.update(source.upload_id for source in found if source.upload_id is not None)
                continue

            record_id = int(value)
            record = client.get_record(kind, record_id)
            key = (kind, record_id)
            if key not in record_keys:
                records.append((kind, record))
                record_keys.add(key)
            found = client.download_uploads(
                record_id,
                destination,
                entity_type=kind,
                used_names=used_names,
                exclude_upload_ids=remote_upload_ids,
                record=record,
            )
            images.extend(found)
            remote_upload_ids.update(source.upload_id for source in found if source.upload_id is not None)
            continue

        for path in _local_paths(kind, value):
            resolved = path.resolve()
            if resolved in local_paths:
                continue
            local_paths.add(resolved)
            name = _unique_local_name(path.name, used_names)
            staged = destination / name
            shutil.copy2(path, staged)
            images.append(
                DownloadedUpload(
                    upload_id=None,
                    original_name=path.name,
                    local_path=staged,
                    source_type="local",
                    source_pointer=str(resolved),
                )
            )

    if not images:
        raise ElabError("No supported microscopy images were resolved from the supplied pointers.")
    return ResolvedSources(images=images, records=records, client=client)


def parse_pointer(pointer: str) -> tuple[str, str]:
    """Return a normalized pointer kind and value."""
    pointer = pointer.strip()
    if not pointer:
        raise ElabError("Image source pointers cannot be empty.")
    if pointer.startswith(("http://", "https://")):
        return _parse_elab_url(pointer)

    match = re.match(r"^(experiment|experiments|item|items|resource|resources|upload):(.+)$", pointer, re.I)
    if match:
        aliases = {
            "experiment": "experiments",
            "experiments": "experiments",
            "item": "items",
            "items": "items",
            "resource": "items",
            "resources": "items",
            "upload": "upload",
        }
        value = match.group(2).strip()
        if not value.isdigit():
            raise ElabError(f"eLabFTW pointer requires a numeric ID: {pointer!r}.")
        return aliases[match.group(1).casefold()], value

    for prefix in ("file:", "dir:", "glob:"):
        if pointer.casefold().startswith(prefix):
            return prefix[:-1], pointer[len(prefix) :]
    path = Path(pointer)
    if path.is_file():
        return "file", pointer
    if path.is_dir():
        return "dir", pointer
    if any(character in pointer for character in "*?["):
        return "glob", pointer
    raise ElabError(f"Unrecognized or missing image source pointer: {pointer!r}.")


def _parse_elab_url(pointer: str) -> tuple[str, str]:
    parsed = urlparse(pointer)
    parts = [part for part in parsed.path.split("/") if part]
    aliases = {"experiments": "experiments", "items": "items", "uploads": "upload"}
    for index, part in enumerate(parts[:-1]):
        kind = aliases.get(part.casefold())
        if kind and parts[index + 1].isdigit():
            return kind, parts[index + 1]

    query_id = (parse_qs(parsed.query).get("id") or [""])[0]
    page = parts[-1].casefold() if parts else ""
    if query_id.isdigit() and page == "experiments.php":
        return "experiments", query_id
    if query_id.isdigit() and page in {"database.php", "items.php"}:
        return "items", query_id
    raise ElabError(f"URL does not identify an eLabFTW experiment, resource, or upload: {pointer!r}.")


def _local_paths(kind: str, value: str) -> list[Path]:
    if kind == "file":
        paths = [Path(value)]
    elif kind == "dir":
        directory = Path(value)
        if not directory.is_dir():
            raise ElabError(f"Image directory does not exist: {directory}.")
        paths = [path for path in directory.rglob("*") if path.is_file()]
    elif kind == "glob":
        paths = [Path(value) for value in glob.glob(value, recursive=True)]
        paths = [path for path in paths if path.is_file()]
    else:
        raise ElabError(f"Unsupported local source type: {kind!r}.")
    supported = [path for path in paths if _supported(path.name)]
    if kind == "file" and (not paths[0].is_file() or not supported):
        raise ElabError(f"Not a supported microscopy image: {paths[0]}.")
    return sorted(supported, key=lambda path: str(path).casefold())


def _expand_manifests(pointers: list[str], seen: set[Path] | None = None) -> list[str]:
    seen = seen or set()
    expanded = []
    for pointer in pointers:
        if not pointer.startswith("@"):
            expanded.append(pointer)
            continue
        manifest = Path(pointer[1:]).resolve()
        if manifest in seen:
            raise ElabError(f"Manifest cycle detected at {manifest}.")
        if not manifest.is_file():
            raise ElabError(f"Source manifest does not exist: {manifest}.")
        seen.add(manifest)
        lines = []
        for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            lines.append(_manifest_relative(value, manifest.parent))
        expanded.extend(_expand_manifests(lines, seen))
        seen.remove(manifest)
    return expanded


def _manifest_relative(pointer: str, base: Path) -> str:
    if pointer.startswith(("http://", "https://", "@")) or re.match(
        r"^(experiment|experiments|item|items|resource|resources|upload):", pointer, re.I
    ):
        if pointer.startswith("@"):
            return "@" + str(base / pointer[1:])
        return pointer
    for prefix in ("file:", "dir:", "glob:"):
        if pointer.casefold().startswith(prefix):
            value = pointer[len(prefix) :]
            if not Path(value).is_absolute():
                value = str(base / value)
            return prefix + value
    return str(base / pointer) if not Path(pointer).is_absolute() else pointer


def _supported(filename: str) -> bool:
    folded = filename.casefold()
    return any(folded.endswith(extension.casefold()) for extension in SUPPORTED_EXTENSIONS)


def _unique_local_name(filename: str, used: set[str]) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._") or "image"
    candidate = safe
    counter = 2
    while candidate.casefold() in used:
        suffixes = "".join(Path(safe).suffixes)
        stem = safe[: -len(suffixes)] if suffixes else safe
        candidate = f"{stem}__{counter}{suffixes}"
        counter += 1
    used.add(candidate.casefold())
    return candidate
