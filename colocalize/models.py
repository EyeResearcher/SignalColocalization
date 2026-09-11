"""Cellpose model adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np


HUGGING_FACE_REPO = "mmzinn12/cellpose-retinal-models"
HUGGING_FACE_MODELS = {
    "cpdino_BRN3A": "cpdino_BRN3A",
    "cpdino_RPBMS": "cpdino_RPBMS",
}


def resolve_model_source(model: str | Path) -> str:
    """Return a local model path, downloading configured Hub models as needed.

    Known retinal model names are downloaded from ``HUGGING_FACE_REPO``. An
    arbitrary Hub file may be selected with
    ``hf://OWNER/REPOSITORY/PATH/TO/FILE``. Existing local paths and Cellpose
    built-in model names pass through unchanged.
    """
    value = str(model)
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return str(expanded)

    if value.startswith("hf://"):
        repo_id, filename = _parse_hf_uri(value)
        return _download_model(repo_id, filename)

    cached_cellpose_model = _cellpose_model_dir() / value
    if cached_cellpose_model.is_file():
        return str(cached_cellpose_model)

    if value in HUGGING_FACE_MODELS:
        return _download_model(HUGGING_FACE_REPO, HUGGING_FACE_MODELS[value])
    return value


def _download_model(repo_id: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "Downloading Cellpose models from Hugging Face requires "
            "huggingface_hub. Install the project requirements."
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


def _parse_hf_uri(uri: str) -> tuple[str, str]:
    parts = uri.removeprefix("hf://").strip("/").split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise ValueError(
            "Hugging Face model references must use "
            "hf://OWNER/REPOSITORY/PATH/TO/FILE."
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def _cellpose_model_dir() -> Path:
    return Path.home() / ".cellpose" / "models"


class CellposeSegmenter:  # pylint: disable=too-few-public-methods
    """Small compatibility wrapper around the Cellpose Python API."""

    def __init__(self, model: str | Path = "cpsam_v2", device: str = "auto") -> None:
        self.model_name = resolve_model_source(model)
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        from cellpose import models  # pylint: disable=import-outside-toplevel

        gpu, resolved_device = self._resolve_device()
        kwargs = {"gpu": gpu, "pretrained_model": self.model_name}
        if resolved_device is not None:
            kwargs["device"] = resolved_device
        try:
            return models.CellposeModel(**kwargs)
        except (TypeError, ValueError):
            # Compatibility with Cellpose 3 model-zoo names such as cyto3/nuclei.
            legacy_kwargs = {"gpu": gpu, "model_type": self.model_name}
            if resolved_device is not None:
                legacy_kwargs["device"] = resolved_device
            if hasattr(models, "Cellpose"):
                return models.Cellpose(**legacy_kwargs)
            return models.CellposeModel(**legacy_kwargs)

    def _resolve_device(self):
        import torch  # pylint: disable=import-outside-toplevel

        requested = self.device.casefold()
        if requested == "auto":
            if torch.cuda.is_available():
                return True, torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return False, torch.device("mps")
            return False, torch.device("cpu")
        if requested.startswith("cuda"):
            return True, torch.device(self.device)
        return False, torch.device(self.device)

    def segment(
        self,
        image: np.ndarray,
        *,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        normalize: bool = True,
    ) -> tuple[np.ndarray, object]:
        """Return an integer label image and Cellpose flow output."""
        result = self.model.eval(
            np.asarray(image),
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            normalize=normalize,
        )
        masks, flows = result[0], result[1]
        return np.asarray(masks, dtype=np.int32), flows
