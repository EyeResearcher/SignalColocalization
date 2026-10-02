import numpy as np
from pathlib import Path
import tifffile
from typing import Any
from matplotlib.figure import Figure
from .datasets import ReferenceSet


def _safe_stem(path: Path) -> str:
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".oir"):
        if name.casefold().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem
def save_mask(
    *,
    masks: np.ndarray,
    acquisition_path: Path,
    reference: ReferenceSet,
    output_dir: Path,
    save_masks: bool = True,
) -> Path:
    """Write a single 2-D label TIFF and return its destination.

    The function rejects non-label arrays rather than silently writing a
    malformed mask. Compression is lossless, and an existing result is
    deliberately replaced when the analysis is rerun to the same output
    folder.
    """
    labels = np.asarray(masks)
    if labels.ndim not in (2, 3):
        raise ValueError(f"Masks for {acquisition_path.name} must be 2-D or 3-D; got {labels.shape}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Masks for {acquisition_path.name} must have an integer dtype.")
    output_dir.mkdir(parents=True, exist_ok=True)
   
    destination = output_dir / f"{_safe_stem(acquisition_path)}__{reference.name}_masks.tif"
    if not save_masks:
        return destination
    tifffile.imwrite(destination,
                    labels,
                    compression="zlib",
                    photometric="minisblack",
                    metadata={"axes": "ZYX" if labels.ndim == 3 else "YX"},
                    )
    return destination


def save_segmentation_views(
    figures: dict[str, Figure],
    *,
    output_dir: str | Path,
    name: str,
    dpi: int = 150,
    save_seg: bool = True,
) -> list[Path]:
    """Save prepared figures; save_seg=False saves only the grid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for panel_name, figure in figures.items():
        if panel_name != "grid" and not save_seg:
            continue

        destination = output_dir / f"{name}__{panel_name}.png"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        saved.append(destination)

    return saved
