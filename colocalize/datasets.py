"""Configuration and result containers for colocalization analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .readers import ImageReader, MicroscopyImage, SUPPORTED_EXTENSIONS
from .transforms import ArrayTransform, Compose, projection_transform


ChannelKey = int | str


class ImageDataset:
    """Lazily read CZYX acquisitions and apply a configurable transform chain."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        reader: ImageReader | None = None,
        transforms: Sequence[ArrayTransform] | ArrayTransform | None = None,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.reader = reader or ImageReader()
        if transforms is None:
            transforms = (projection_transform(self.reader.z_projection),)
        elif callable(transforms):
            transforms = (transforms,)
        self.transforms = Compose(transforms)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> MicroscopyImage:
        acquisition = self.reader.read_stack(self.paths[index])
        data = np.asarray(self.transforms(acquisition.data))
        if data.ndim != 3:
            raise ValueError(
                "ImageDataset transforms must produce a CYX array before analysis; "
                f"received shape {data.shape} for {acquisition.path.name}."
            )
        if data.shape[0] != len(acquisition.channel_names):
            raise ValueError(
                "Transforms changed the channel dimension. Transforms must preserve "
                "the leading C axis."
            )
        return MicroscopyImage(
            path=acquisition.path,
            data=data,
            channel_names=acquisition.channel_names,
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


@dataclass(frozen=True)
class ReferenceSet:
    """A channel and the Cellpose model/settings used to segment it."""

    name: str
    channel: ChannelKey
    model: str | Path = "cpdino_BRN3A"
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    normalize: bool = True


@dataclass(frozen=True)
class SignalChannel:
    """A channel measured inside every reference mask."""

    name: str
    channel: ChannelKey
    threshold_method: str = "otsu"
    threshold_value: float | None = None
    positive_fraction_cutoff: float = 0.80


@dataclass
class AnalysisConfig:
    """All experiment-specific choices for a directory-level analysis."""

    input_dir: str | Path
    output_dir: str | Path
    reference_sets: list[ReferenceSet]
    signal_channels: list[SignalChannel]
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS
    recursive: bool = False
    exclude: Sequence[str | Path] | str | Path = ()
    time_index: int = 0
    scene_index: int = 0
    z_projection: str = "max"
    device: str = "auto"
    save_masks: bool = True
    save_segmentation: bool = False
    segmentation_output_dir: str | Path | None = None
    transforms: Sequence[ArrayTransform] | None = None
    tile_size: tuple[int, int] | None = None
    stitch_masks: bool = False

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.segmentation_output_dir is not None:
            self.segmentation_output_dir = Path(self.segmentation_output_dir)
            self.save_segmentation = True
        if isinstance(self.exclude, (str, Path)):
            self.exclude = (self.exclude,)
        else:
            self.exclude = tuple(self.exclude)
        if self.tile_size is not None:
            h, w = self.tile_size
            if h <= 0 or w <= 0:
                raise ValueError("tile_size dimensions must be positive integers.")
            self.tile_size = (int(h), int(w))
        if not self.reference_sets:
            raise ValueError("At least one ReferenceSet is required.")
        if not self.signal_channels:
            raise ValueError("At least one SignalChannel is required.")
        for collection, label in (
            (self.reference_sets, "reference set"),
            (self.signal_channels, "signal channel"),
        ):
            names = [item.name for item in collection]
            if len(names) != len(set(names)):
                raise ValueError(f"Each {label} name must be unique.")

    def resolved_transforms(self) -> tuple[ArrayTransform, ...]:
        """Use explicit transforms, or the legacy projection setting by default."""
        if self.transforms is not None:
            return tuple(self.transforms)
        return (projection_transform(self.z_projection),)


@dataclass
class AnalysisResult:
    """Tables and saved mask locations produced by a run."""

    cells: pd.DataFrame
    images: pd.DataFrame
    mask_paths: list[Path] = field(default_factory=list)
    segmentation_paths: list[Path] = field(default_factory=list)

    def save_tables(self, output_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cells_path = output_dir / "cells.csv"
        images_path = output_dir / "image_summary.csv"
        self.cells.to_csv(cells_path, index=False)
        self.images.to_csv(images_path, index=False)
        return cells_path, images_path
