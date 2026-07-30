"""Export multi-channel Z max projections from a reproducible OIR subset."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import tifffile
from oirfile import OirFile


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _to_czyx(
    data: np.ndarray,
    dims: tuple[str, ...] | list[str] | str,
    *,
    time_index: int = 0,
) -> np.ndarray:
    """Normalize an OirFile array to channel, Z, Y, X order."""
    axes = [str(axis).upper() for axis in dims]
    array = np.asarray(data)
    if len(axes) != array.ndim:
        raise ValueError(f"OIR dimensions {dims!r} do not match shape {array.shape}.")

    if "T" in axes:
        position = axes.index("T")
        array = np.take(array, time_index, axis=position)
        axes.pop(position)

    if "S" in axes:
        if "C" in axes:
            raise ValueError(f"OIR dimensions cannot contain both C and S: {dims!r}")
        axes[axes.index("S")] = "C"

    for axis in tuple(axes):
        if axis not in {"C", "Z", "Y", "X"}:
            position = axes.index(axis)
            if array.shape[position] != 1:
                raise ValueError(
                    f"Cannot discard non-singleton OIR dimension {axis!r} "
                    f"with size {array.shape[position]}."
                )
            array = np.take(array, 0, axis=position)
            axes.pop(position)

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"OIR dimensions must contain Y and X; received {dims!r}.")
    if "C" not in axes:
        array = np.expand_dims(array, 0)
        axes.insert(0, "C")
    if "Z" not in axes:
        array = np.expand_dims(array, 1)
        axes.insert(1, "Z")

    return np.transpose(array, [axes.index(axis) for axis in "CZYX"])


def export_subset(
    input_dir: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
    time_index: int = 0,
) -> list[dict[str, object]]:
    sources = sorted(
        (path for path in input_dir.iterdir() if path.suffix.casefold() == ".oir"),
        key=lambda path: path.name.casefold(),
    )
    if not sources:
        raise FileNotFoundError(f"No .oir files found in {input_dir}")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    rng = random.Random(seed)
    selected = rng.sample(sources, min(sample_size, len(sources)))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for source in selected:
        with OirFile(source, squeeze=False) as image:
            czyx = _to_czyx(
                image.asarray(),
                image.dims,
                time_index=time_index,
            )
            names = image.coords.get("C")
            if names is None:
                names = image.coords.get("S")
            names = list(names) if names is not None else []

        projections = np.max(czyx, axis=1)
        channel_names = [
            (
                str(names[channel_index])
                if channel_index < len(names) and names[channel_index]
                else f"channel_{channel_index}"
            )
            for channel_index in range(czyx.shape[0])
        ]
        destination = output_dir / f"{_safe_name(source.stem)}__zmax_channels.ome.tif"
        tifffile.imwrite(
            destination,
            projections,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": channel_names},
                "source_oir": source.name,
                "projection": "maximum",
            },
        )
        for channel_index in range(czyx.shape[0]):
            manifest.append(
                {
                    "source_oir": source.name,
                    "source_shape_czyx": str(tuple(czyx.shape)),
                    "output_shape_cyx": str(tuple(projections.shape)),
                    "channel_index": channel_index,
                    "channel_name": channel_names[channel_index],
                    "output_tif": str(destination.relative_to(output_dir)),
                    "dtype": str(projections.dtype),
                    "height": projections.shape[1],
                    "width": projections.shape[2],
                    "seed": seed,
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()

    manifest = export_subset(
        args.input_dir,
        args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        time_index=args.time_index,
    )
    source_count = len({row["source_oir"] for row in manifest})
    print(
        f"Exported {source_count} multi-channel projections "
        f"({len(manifest)} total channels) "
        f"to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
