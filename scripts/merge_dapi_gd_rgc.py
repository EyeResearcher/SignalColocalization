"""Build three-channel GD/RPBMS/DAPI TIFFs from matched tiled-retina images."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import tifffile


GD_RGC_DIR = Path(r"C:\Users\mzinn1\Desktop\DendrimerColocalization\TiledRetina\GD-RGC")
DAPI_DIR = Path(r"C:\Users\mzinn1\Desktop\CellposeTrainingDatasets\TiledRetina\DAPI")
OUTPUT_DIR = GD_RGC_DIR.parent / "DAPI-GD-RGC"


def sample_key(path: Path) -> tuple[str, str]:
    """Return the four-digit sample ID and eye side from either file convention."""
    match = re.match(r"(\d{4})[ _]+(?:\1[ _]+)?([LR])(?:[_ ]|$)", path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot identify sample/eye in {path.name}.")
    return match.group(1), match.group(2).upper()


def main() -> None:
    gd_files = sorted(GD_RGC_DIR.glob("*.tif"))
    dapi_files = {sample_key(path): path for path in DAPI_DIR.glob("*.tif")}
    missing = [path.name for path in gd_files if sample_key(path) not in dapi_files]
    if missing:
        raise FileNotFoundError(f"No matching DAPI image(s): {', '.join(missing)}")
    OUTPUT_DIR.mkdir(exist_ok=False)
    for gd_path in gd_files:
        dapi_path = dapi_files[sample_key(gd_path)]
        gd_rgc = tifffile.imread(gd_path)
        dapi_rgb = tifffile.imread(dapi_path)
        if gd_rgc.ndim != 3 or gd_rgc.shape[0] != 2:
            raise ValueError(f"Expected GD/RPBMS CYX data in {gd_path.name}; got {gd_rgc.shape}.")
        if dapi_rgb.ndim != 3 or dapi_rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB DAPI data in {dapi_path.name}; got {dapi_rgb.shape}.")
        if gd_rgc.shape[1:] != dapi_rgb.shape[:2]:
            raise ValueError(f"Dimension mismatch: {gd_path.name} and {dapi_path.name}.")
        merged = np.empty((3, *gd_rgc.shape[1:]), dtype=np.uint8)
        merged[:2] = gd_rgc
        merged[2] = dapi_rgb[..., 2]  # Blue is the only non-empty DAPI component.
        destination = OUTPUT_DIR / gd_path.name
        tifffile.imwrite(destination, merged, compression="zlib", metadata={"axes": "CYX"})
        print(f"Wrote {destination.name}: {merged.shape} (GD, RPBMS, DAPI)")


if __name__ == "__main__":
    main()
