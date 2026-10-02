"""Append matching DAPI images to the HD-RGC two-channel TIFF stacks."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import tifffile


HD_RGC_DIR = Path(r"C:\Users\mzinn1\Desktop\DendrimerColocalization\TiledRetina\HD-RGC")
DAPI_DIR = Path(r"C:\Users\mzinn1\Desktop\CellposeTrainingDatasets\TiledRetina\DAPI")
OUTPUT_DIR = HD_RGC_DIR.parent / "DAPI-HD-RGC"
CHUNK_ROWS = 128


def key(path: Path) -> tuple[str, str]:
    match = re.search(r"^(\d{4})[ _]+(?:\1[ _]+)?([LR])(?:[ _]|$)", path.name, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Cannot identify animal and side from {path.name}")
    return match.group(1), match.group(2).upper()


def main() -> int:
    hd_files = sorted(HD_RGC_DIR.glob("*.tif"))
    dapi_by_key = {key(path): path for path in DAPI_DIR.glob("*.tif")}
    hd_keys = {key(path) for path in hd_files}
    if len(dapi_by_key) != len(hd_files) or set(dapi_by_key) != hd_keys:
        raise RuntimeError(f"HD-RGC keys: {sorted(hd_keys)}; DAPI keys: {sorted(dapi_by_key)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for hd_path in hd_files:
        dapi_path = dapi_by_key[key(hd_path)]
        hd = tifffile.imread(hd_path)
        dapi = tifffile.memmap(dapi_path, mode="r")
        if hd.ndim != 3 or hd.shape[0] != 2 or dapi.shape != (hd.shape[1], hd.shape[2], 3):
            raise RuntimeError(f"Incompatible inputs: {hd_path.name} {hd.shape}; {dapi_path.name} {dapi.shape}")

        merged = np.empty((3, hd.shape[1], hd.shape[2]), dtype=hd.dtype)
        for row in range(0, hd.shape[1], CHUNK_ROWS):
            end = min(row + CHUNK_ROWS, hd.shape[1])
            merged[0, row:end] = np.max(dapi[row:end], axis=-1)
            merged[1:, row:end] = hd[:, row:end]

        output = OUTPUT_DIR / hd_path.name.replace("C004_C005_merged", "C001_C004_C005_merged")
        tifffile.imwrite(
            output,
            merged,
            imagej=True,
            metadata={"axes": "CYX"},
            compression="zlib",
            rowsperstrip=CHUNK_ROWS,
        )
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
