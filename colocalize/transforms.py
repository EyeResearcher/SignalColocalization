"""Composable transforms for microscopy image stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


ArrayTransform = Callable[[np.ndarray], np.ndarray]


class Compose:
    """Apply transforms sequentially."""

    def __init__(self, transforms: Iterable[ArrayTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = np.asarray(image)
        for transform in self.transforms:
            result = np.asarray(transform(result))
        return result


@dataclass(frozen=True)
class MaxProjection:
    """Maximum-intensity projection of a CZYX stack along Z."""

    axis: int = 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.max(np.asarray(image), axis=self.axis)


@dataclass(frozen=True)
class MeanProjection:
    """Mean-intensity projection of a CZYX stack along Z."""

    axis: int = 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.mean(np.asarray(image), axis=self.axis)


@dataclass(frozen=True)
class SelectPlane:
    """Select one Z plane from a CZYX stack."""

    index: int = 0
    axis: int = 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.take(np.asarray(image), self.index, axis=self.axis)


@dataclass(frozen=True)
class PercentileNormalize:
    """Scale each channel independently using robust intensity percentiles."""

    lower: float = 1.0
    upper: float = 99.0
    channel_axis: int = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        moved = np.moveaxis(array, self.channel_axis, 0)
        normalized = np.empty_like(moved, dtype=np.float32)
        for index, channel in enumerate(moved):
            low, high = np.nanpercentile(channel, (self.lower, self.upper))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                normalized[index] = 0
            else:
                normalized[index] = np.clip((channel - low) / (high - low), 0, 1)
        return np.moveaxis(normalized, 0, self.channel_axis)


def projection_transform(method: str) -> ArrayTransform:
    """Construct a projection transform from the legacy configuration name."""
    normalized = method.casefold()
    if normalized == "max":
        return MaxProjection()
    if normalized == "mean":
        return MeanProjection()
    if normalized == "first":
        return SelectPlane(0)
    raise ValueError("z_projection must be 'max', 'mean', or 'first'.")
