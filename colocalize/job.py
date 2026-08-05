"""Serializable job contract shared by the web form and Python runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .cli import config_from_mapping
from .datasets import AnalysisConfig
from .elabftw import ElabError


@dataclass(frozen=True)
class ColocalizationJob:
    """A complete, interface-submittable colocalization request."""

    sources: list[str]
    analysis: AnalysisConfig
    inspect: bool = False
    save_segmentation: bool = False


def load_job(path: str | Path) -> ColocalizationJob:
    """Load and minimally validate a website-generated job JSON file."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ElabError(f"Job file is not valid JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ElabError("Job JSON must be an object.")
    _reject_unknown(payload, {"schema_version", "sources", "analysis", "execution"}, "job")
    if payload.get("schema_version", 1) != 1:
        raise ElabError("Unsupported job schema_version; expected 1.")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(source, str) and source.strip() for source in sources
    ):
        raise ElabError("Job field 'sources' must be a non-empty list of pointers.")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ElabError("Job field 'analysis' must be an object.")
    analysis = dict(analysis)
    analysis.setdefault("input_dir", ".")
    try:
        config = config_from_mapping(analysis, base_dir=path.parent)
    except (TypeError, ValueError) as exc:
        raise ElabError(f"Invalid analysis settings in job: {exc}") from exc
    execution = payload.get("execution") or {}
    if not isinstance(execution, dict):
        raise ElabError("Job field 'execution' must be an object.")
    _reject_unknown(execution, {"inspect", "save_segmentation"}, "execution")
    _validate_analysis(config)
    return ColocalizationJob(
        sources=[source.strip() for source in sources],
        analysis=config,
        inspect=_boolean(execution.get("inspect", False), "execution.inspect"),
        save_segmentation=_boolean(
            execution.get("save_segmentation", False), "execution.save_segmentation"
        ),
    )


def job_schema_path() -> Path:
    """Return the distributable JSON Schema for interface builders."""
    return Path(__file__).with_name("colocalization_job.schema.json")


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ElabError(f"Job field {field!r} must be true or false.")
    return value


def _reject_unknown(payload: dict, allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ElabError(f"Unknown {label} field(s): {', '.join(sorted(unknown))}.")


def _validate_analysis(config: AnalysisConfig) -> None:
    if not str(config.output_dir):
        raise ElabError("analysis.output_dir cannot be empty.")
    if config.time_index < 0 or config.scene_index < 0:
        raise ElabError("Time and scene indexes must be zero or greater.")
    if config.z_projection not in {"max", "mean", "first"}:
        raise ElabError("analysis.z_projection must be max, mean, or first.")
    if not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", config.device):
        raise ElabError("analysis.device must be auto, cpu, mps, cuda, or cuda:N.")
    if not config.extensions or any(not str(extension).startswith(".") for extension in config.extensions):
        raise ElabError("analysis.extensions must contain dot-prefixed filename extensions.")
    for reference in config.reference_sets:
        _validate_channel(reference.channel, f"reference {reference.name!r}")
        if reference.diameter is not None and reference.diameter <= 0:
            raise ElabError(f"Reference {reference.name!r} diameter must be positive or null.")
        if reference.min_size < 0:
            raise ElabError(f"Reference {reference.name!r} min_size cannot be negative.")
    for signal in config.signal_channels:
        _validate_channel(signal.channel, f"signal {signal.name!r}")
        if signal.threshold_method not in {"otsu", "percentile", "absolute", "none"}:
            raise ElabError(f"Signal {signal.name!r} has an unsupported threshold method.")
        if signal.threshold_method == "absolute" and signal.threshold_value is None:
            raise ElabError(f"Signal {signal.name!r} requires threshold_value for absolute thresholding.")
        if not 0 <= signal.positive_fraction_cutoff <= 1:
            raise ElabError(f"Signal {signal.name!r} positive_fraction_cutoff must be between 0 and 1.")


def _validate_channel(value: int | str, label: str) -> None:
    if isinstance(value, int) and value < 0:
        raise ElabError(f"The channel for {label} cannot be negative.")
    if isinstance(value, str) and not value.strip():
        raise ElabError(f"The channel for {label} cannot be empty.")
