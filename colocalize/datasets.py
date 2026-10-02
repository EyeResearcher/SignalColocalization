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
BACKGROUND_METHODS = (
    "none",
    "fixed",
    "median",
    "percentile",
    "tissue_filtered_oop",
)


class ImageDataset:
    """Lazily read CZYX acquisitions and apply a configurable transform chain."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        reader: ImageReader | None = None,
        transforms: Sequence[ArrayTransform] | ArrayTransform | None = None,
    ) -> None:
        """Initialise the dataset with an ordered sequence of image paths.

        Args:
            paths: Ordered collection of file paths to load.
            reader: Reader used to decode each file.  Defaults to
                :class:`~colocalize.readers.ImageReader` with its own defaults.
            transforms: One or more callables that map the raw CZYX array to a
                CYX array suitable for analysis.  Defaults to the reader's
                configured Z-projection transform.
        """
        self.paths = tuple(Path(path) for path in paths)
        self.reader = reader or ImageReader()
        if transforms is None:
            transforms = (projection_transform(self.reader.z_projection),)
        elif callable(transforms):
            transforms = (transforms,)
        self.transforms = Compose(transforms)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> MicroscopyImage:
        """Load, transform, and return the image at the given index.

        Args:
            index: Zero-based position in :attr:`paths`.

        Returns:
            :class:`~colocalize.readers.MicroscopyImage` with the raw data
            replaced by the transformed CYX array.

        Raises:
            ValueError: If the transform chain produces an array that is not
                three-dimensional or changes the channel count.
        """
        acquisition = self.reader.read_stack(self.paths[index])
        data = np.asarray(self.transforms(acquisition.data))
        if data.ndim not in (3, 4):
            raise ValueError("Expected CYX or CZYX.")
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
        """Iterate over all images in path order, loading and transforming each one."""
        for index in range(len(self)):
            yield self[index]

@dataclass(frozen=True)
class TissueChannel:
    """A named channel and settings used to define the whole-tissue domain.

    ``threshold=None`` retains the automatic Otsu threshold.  Supplying an
    absolute threshold is useful for images such as the tiled DAPI scans,
    where a low, known cutoff preserves dim tissue that Otsu may omit.
    """

    name: str
    channel: ChannelKey
    threshold: float | None = None
    min_object_px: int = 5000
    min_hole_px: int = 5000
    fill_holes: bool = False

    def __post_init__(self) -> None:
        if self.min_object_px < 1:
            raise ValueError("min_object_px must be at least 1.")
        if self.min_hole_px < 1:
            raise ValueError("min_hole_px must be at least 1.")

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
    cellpose_kwargs: dict | None = None


@dataclass(frozen=True)
class SignalChannel:
    """A channel measured inside every reference mask."""

    name: str
    channel: ChannelKey
    threshold_method: str = "otsu"
    threshold_value: float | None = None
    positive_fraction_cutoff: float = 0.80
    # Distributional quantification options.  Defaults retain the historic raw
    # intensity measurements and do not apply a correction.
    background_method: str = "none"
    background_percentile: float = 50.0
    tissue_channel: ChannelKey | None = None
    background_buffer_px: int = 3
    background_anchor_percentile: float = 15.0
    background_fixed_value: float | None = None
    background_dapi_threshold: float = 1.0
    debris_percentile: float = 96.0
    debris_min_area_cell_sd: float = 5.0
    debris_buffer_px: int = 0
    noise_normalize: bool = False
    asinh_scale: float | None = None
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    tail_fraction: float = 0.10
    noise_thresholds: tuple[float, ...] = (3.0,)
    debris_thresh_property: str | list[str] = "area"
    prop_percentile_thresh: float = 50.0

    def __post_init__(self) -> None:
        """Validate and normalize distributional measurement options."""
        if self.background_method not in BACKGROUND_METHODS:
            raise ValueError(
                "Unknown background_method. Use none, fixed, median, percentile, "
                "or tissue_filtered_oop."
            )
        if self.background_fixed_value is not None and not np.isfinite(self.background_fixed_value):
            raise ValueError("background_fixed_value must be finite.")
        if self.background_method == "fixed" and self.background_fixed_value is None:
            raise ValueError("fixed background requires background_fixed_value.")
        if not np.isfinite(self.background_dapi_threshold):
            raise ValueError("background_dapi_threshold must be finite.")
        if not 0 <= self.background_percentile <= 100:
            raise ValueError("background_percentile must be between 0 and 100.")
        if not isinstance(self.background_buffer_px, (int, np.integer)) or self.background_buffer_px < 0:
            raise ValueError("background_buffer_px must be a nonnegative integer.")
        if not 0 <= self.background_anchor_percentile <= 100:
            raise ValueError("background_anchor_percentile must be between 0 and 100.")
        if self.debris_percentile is None or not 0 < self.debris_percentile < 100:
            raise ValueError("debris_percentile must be strictly between 0 and 100.")
        if self.debris_min_area_cell_sd < 0:
            raise ValueError("debris_min_area_cell_sd cannot be negative.")
        if self.debris_buffer_px < 0:
            raise ValueError("debris_buffer_px cannot be negative.")
        properties = (
            [self.debris_thresh_property]
            if isinstance(self.debris_thresh_property, str)
            else self.debris_thresh_property
        )
        if not isinstance(properties, list) or not properties or not all(
            isinstance(value, str) and value for value in properties
        ):
            raise ValueError("debris_thresh_property must be a non-empty string or list of strings.")
        if not 0 <= self.prop_percentile_thresh <= 100:
            raise ValueError("prop_percentile_thresh must be between 0 and 100.")
        if self.asinh_scale is not None and self.asinh_scale <= 0:
            raise ValueError("asinh_scale must be positive when supplied.")
        quantiles = tuple(float(value) for value in self.quantiles)
        if not quantiles or any(not 0 <= value <= 1 for value in quantiles):
            raise ValueError("quantiles must be a non-empty sequence between 0 and 1.")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1].")
        thresholds = tuple(float(value) for value in self.noise_thresholds)
        if any(value < 0 for value in thresholds):
            raise ValueError("noise_thresholds cannot contain negative values.")
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "noise_thresholds", thresholds)


@dataclass
class AnalysisConfig:
    """All experiment-specific choices for a directory-level analysis."""

    input_dir: str | Path
    output_dir: str | Path
    reference_sets: list[ReferenceSet]
    signal_channels: list[SignalChannel]
    tissue_channel: TissueChannel | None = None
    bit_depth: int = 2**8-1
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS
    recursive: bool = False
    exclude: Sequence[str | Path] | str | Path = ()
    time_index: int = 0
    scene_index: int = 0
    z_projection: str = "max"
    device: str = "auto"
    save_masks: bool = True
    save_segmentation: bool = False
    progress: bool = True
    load_masks: bool = False
    segmentation_output_dir: str | Path | None = None
    transforms: Sequence[ArrayTransform] | None = None
    mask_dir: str | Path | None = None
    morphology_properties: tuple[str, ...] | list[str] | None = None
    promote_2d_to_3d: bool = False

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
        if not self.reference_sets:
            raise ValueError("At least one ReferenceSet is required.")
        if not self.signal_channels:
            raise ValueError("At least one SignalChannel is required.")
        collections = [
            (self.reference_sets, "reference set"),
            (self.signal_channels, "signal channel"),
        ]
        if self.tissue_channel is not None:
            collections.append(([self.tissue_channel], "tissue channel"))
        for collection, label in collections:
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
        """Write cells.csv and image_summary.csv to output_dir.

        Args:
            output_dir: Directory in which to write the two CSV files.
                Created automatically if it does not exist.

        Returns:
            Tuple of ``(cells_path, images_path)`` for the two written files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cells_path = output_dir / "cells.csv"
        images_path = output_dir / "image_summary.csv"
        self.cells.to_csv(cells_path, index=False)
        self.images.to_csv(images_path, index=False)
        return cells_path, images_path
