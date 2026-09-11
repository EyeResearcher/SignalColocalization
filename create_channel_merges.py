"""Create pairwise RGB channel overlays from TIFF frame directories."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import tifffile


ROOT = Path(r"C:\Users\mzinn1\Desktop\Shashi_Colocal")
PAIRS = (("C003", "C005"), ("C004", "C005"))
CHUNK_ROWS = 128


def subject_label(frames_dir: Path) -> str:
    """Return a collision-safe label based on the animal and eye directory names."""
    animal = frames_dir.parents[1].name
    side = frames_dir.parent.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{animal}_{side}").strip("_")


def source_for(frames_dir: Path, channel: str) -> Path:
    matches = sorted(frames_dir.glob(f"*_{channel}T001.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {channel} TIFF in {frames_dir}, found {len(matches)}")
    return matches[0]


def merge_images(first: Path, second: Path, destination: Path) -> None:
    """Write the two source signals as a two-channel ImageJ TIFF stack.

    Frame exports are RGB false-color renderings.  Taking the maximum component
    recovers each source's 8-bit signal without preserving the display LUT.
    """
    a = tifffile.memmap(first, mode="r")
    b = tifffile.memmap(second, mode="r")
    if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 3 or a.shape[-1] != 3:
        raise RuntimeError(
            f"Incompatible source images: {first.name} {a.shape}/{a.dtype}; "
            f"{second.name} {b.shape}/{b.dtype}"
        )

    signals = np.empty((2, a.shape[0], a.shape[1]), dtype=a.dtype)
    for row in range(0, a.shape[0], CHUNK_ROWS):
        end = min(row + CHUNK_ROWS, a.shape[0])
        signals[0, row:end] = np.max(a[row:end], axis=-1)
        signals[1, row:end] = np.max(b[row:end], axis=-1)

    tifffile.imwrite(
        destination,
        signals,
        imagej=True,
        metadata={"axes": "CYX"},
        compression="zlib",
        rowsperstrip=CHUNK_ROWS,
    )


def main() -> int:
    frame_dirs = sorted(ROOT.rglob("*.tif.frames"))
    if not frame_dirs:
        raise RuntimeError(f"No .tif.frames directories found under {ROOT}")

    created: list[Path] = []
    for first_channel, second_channel in PAIRS:
        output_dir = ROOT / f"Merged_{first_channel}_{second_channel}"
        output_dir.mkdir(exist_ok=True)
        for frames_dir in frame_dirs:
            first = source_for(frames_dir, first_channel)
            second = source_for(frames_dir, second_channel)
            output = output_dir / f"{subject_label(frames_dir)}_{first_channel}_{second_channel}_merged.tif"
            merge_images(first, second, output)
            created.append(output)
            print(output)

    print(f"Created {len(created)} merged TIFFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
