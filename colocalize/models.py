"""Cellpose model adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CellposeSegmenter:
    """Small compatibility wrapper around the Cellpose Python API."""

    def __init__(self, model: str | Path = "cpsam_v2", device: str = "auto") -> None:
        self.model_name = str(model)
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        from cellpose import models

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
        import torch

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
