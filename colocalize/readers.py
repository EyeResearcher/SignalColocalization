"""Microscopy image discovery and channel extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".czi", ".oir")


@dataclass(frozen=True)
class MicroscopyImage:
    """A single acquisition normalized to CZYX or transformed CYX."""

    path: Path
    data: np.ndarray
    channel_names: tuple[str, ...]

    def channel(self, key: int | str) -> np.ndarray:
        """Return a channel by zero-based index or metadata name."""
        if isinstance(key, int):
            index = key
        else:
            exact = {name: i for i, name in enumerate(self.channel_names)}
            folded = {name.casefold(): i for i, name in enumerate(self.channel_names)}
            if key in exact:
                index = exact[key]
            elif key.casefold() in folded:
                index = folded[key.casefold()]
            else:
                available = ", ".join(self.channel_names)
                raise KeyError(f"Channel {key!r} not found. Available channels: {available}")

        if not 0 <= index < self.data.shape[0]:
            raise IndexError(
                f"Channel index {index} is out of range for {self.path.name} "
                f"({self.data.shape[0]} channels)."
            )
        return np.asarray(self.data[index])


class ImageReader:
    """Read microscopy images as CZYX arrays."""

    def __init__(
        self,
        *,
        time_index: int = 0,
        scene_index: int = 0,
        z_projection: str = "max",
    ) -> None:
        self.time_index = time_index
        self.scene_index = scene_index
        self.z_projection = z_projection

    def read(self, path: str | Path) -> MicroscopyImage:
        """Read an acquisition and apply the configured legacy Z projection."""
        stack = self.read_stack(path)
        return MicroscopyImage(
            path=stack.path,
            data=self._project_z(stack.data),
            channel_names=stack.channel_names,
        )

    def read_stack(self, path: str | Path) -> MicroscopyImage:
        """Read an acquisition without projecting its CZYX Z stack."""
        path = Path(path)
        if path.suffix.casefold() == ".oir":
            return self._read_oir_stack(path)
        if any(path.name.casefold().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return self._read_bioio_stack(path)
        raise ValueError(f"Unsupported image format: {path}")

    def _read_bioio_stack(self, path: Path) -> MicroscopyImage:
        try:
            from bioio import BioImage
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.suffix} files requires BioIO and its format plugin. "
                "Install the project requirements."
            ) from exc

        image = BioImage(path)
        image.set_scene(self.scene_index)
        data = np.asarray(image.get_image_data("CZYX", T=self.time_index))
        names = self._channel_names(image.channel_names, data.shape[0])
        return MicroscopyImage(path=path, data=data, channel_names=names)

    def _read_oir_stack(self, path: Path) -> MicroscopyImage:
        try:
            from oirfile import OirFile
        except ImportError as exc:
            raise ImportError(
                "Reading .oir files requires OirFile. Install the project requirements."
            ) from exc

        if self.scene_index != 0:
            raise IndexError(
                f"OIR files contain one scene; scene index {self.scene_index} is invalid."
            )

        with OirFile(path, squeeze=False) as image:
            data = self._to_czyx(image.asarray(), image.dims, path)
            names = image.coords.get("C")
            if names is None:
                names = image.coords.get("S")
            channel_names = self._channel_names(
                names,
                data.shape[0],
            )
        return MicroscopyImage(path=path, data=data, channel_names=channel_names)

    def _to_czyx(
        self, data: np.ndarray, axes: Iterable[str] | str, path: Path
    ) -> np.ndarray:
        axes_list = [str(axis).upper() for axis in axes]
        array = np.asarray(data)
        if len(axes_list) != array.ndim:
            raise ValueError(f"Axes {axes!r} do not match shape {array.shape} for {path}.")

        if "T" in axes_list:
            position = axes_list.index("T")
            array = np.take(array, self.time_index, axis=position)
            axes_list.pop(position)

        # OirFile uses S instead of C for planar RGB samples.
        if "S" in axes_list:
            if "C" in axes_list:
                raise ValueError(f"OIR axes cannot contain both C and S: {axes!r}.")
            axes_list[axes_list.index("S")] = "C"

        unknown = [axis for axis in axes_list if axis not in {"C", "Z", "Y", "X"}]
        for axis in unknown:
            position = axes_list.index(axis)
            if array.shape[position] != 1:
                raise ValueError(
                    f"Cannot convert non-singleton OIR axis {axis!r} in {path.name} "
                    "to CZYX."
                )
            array = np.take(array, 0, axis=position)
            axes_list.pop(position)

        if "Y" not in axes_list or "X" not in axes_list:
            print(f"Warning: {path.name} does not contain Y and X axes; got {''.join(axes_list)!r}.")
            raise ValueError(f"Image axes must contain Y and X; got {''.join(axes_list)!r}.")
        if "C" not in axes_list:
            array = np.expand_dims(array, 0)
            axes_list.insert(0, "C")
        if "Z" not in axes_list:
            array = np.expand_dims(array, 1)
            axes_list.insert(1, "Z")

        order = [axes_list.index(axis) for axis in "CZYX"]
        return np.transpose(array, order)

    def _project_z(self, czyx: np.ndarray) -> np.ndarray:
        from .transforms import projection_transform

        return projection_transform(self.z_projection)(czyx)

    @staticmethod
    def _channel_names(names: Iterable[str] | None, count: int) -> tuple[str, ...]:
        provided = list(names) if names is not None else []
        return tuple(
            str(provided[i]) if i < len(provided) and provided[i] else f"channel_{i}"
            for i in range(count)
        )


def discover_images(
    directory: str | Path,
    *,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Return deterministically ordered microscopy files under a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    normalized = tuple(extension.casefold() for extension in extensions)
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and any(path.name.casefold().endswith(ext) for ext in normalized)
    ]
    return sorted(paths, key=lambda path: path.as_posix().casefold())
