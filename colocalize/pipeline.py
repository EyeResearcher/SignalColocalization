"""Directory-level orchestration for segmentation and colocalization."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .colocalize import measure_masks, summarize_cells
from .datasets import AnalysisConfig, AnalysisResult, ImageDataset
from .models import CellposeSegmenter
from .readers import ImageReader, MicroscopyImage, discover_images
from .tiling import generate_tiles, stitch_masks
from .visualization import save_segmentation_views


def inspect_inputs(config: AnalysisConfig) -> pd.DataFrame:
    """List files, shapes, and channel metadata before an expensive run."""
    rows = []
    for image in build_dataset(config):
        path = image.path
        rows.append(
            {
                "source": path.name,
                "path": str(path),
                "shape_cyx": tuple(image.data.shape),
                "channels": tuple(image.channel_names),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run every configured reference set against every configured signal channel."""
    paths = _input_paths(config)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {config.input_dir}.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = config.output_dir / "masks"
    if config.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    segmenters: dict[str, CellposeSegmenter] = {}
    tables: list[pd.DataFrame] = []
    mask_paths: list[Path] = []
    segmentation_paths: list[Path] = []
    analyzed_groups: list[dict[str, str]] = []

    for acquisition in build_dataset(config, paths):
        path = acquisition.path
        signal_specs = {spec.name: spec for spec in config.signal_channels}

        if config.tile_size is not None:
            tile_iter = list(generate_tiles(acquisition.data, config.tile_size))
        else:
            tile_iter = [(acquisition.data, 0, 0)]

        for reference in config.reference_sets:
            analyzed_groups.append(
                {"source": path.name, "reference_set": reference.name}
            )
            if reference.name not in segmenters:
                segmenters[reference.name] = CellposeSegmenter(
                    reference.model, device=config.device
                )

            tile_masks_buffer: list[tuple[int, int, np.ndarray]] = []

            for tile_data, tile_y, tile_x in tile_iter:
                tile_acq = MicroscopyImage(
                    path=path, data=tile_data, channel_names=acquisition.channel_names
                )
                signals = {
                    spec.name: tile_acq.channel(spec.channel)
                    for spec in config.signal_channels
                }
                reference_image = tile_acq.channel(reference.channel)
                masks, _ = segmenters[reference.name].segment(
                    reference_image,
                    diameter=reference.diameter,
                    flow_threshold=reference.flow_threshold,
                    cellprob_threshold=reference.cellprob_threshold,
                    min_size=reference.min_size,
                    normalize=reference.normalize,
                )
                table = measure_masks(
                    source=path.name,
                    reference_set=reference.name,
                    reference_image=reference_image,
                    masks=masks,
                    signals=signals,
                    signal_specs=signal_specs,
                    tile_y=tile_y,
                    tile_x=tile_x,
                )
                tables.append(table)

                if config.save_masks:
                    if config.stitch_masks:
                        tile_masks_buffer.append((tile_y, tile_x, masks))
                    elif config.tile_size is not None:
                        destination = (
                            mask_dir
                            / f"{_safe_stem(path)}__{reference.name}"
                            f"__tile_y{tile_y}_x{tile_x}_masks.tif"
                        )
                        tifffile.imwrite(destination, masks, compression="zlib")
                        mask_paths.append(destination)
                    else:
                        destination = mask_dir / f"{_safe_stem(path)}__{reference.name}_masks.tif"
                        tifffile.imwrite(destination, masks, compression="zlib")
                        mask_paths.append(destination)

                if config.save_segmentation:
                    qc_dir = (
                        config.segmentation_output_dir
                        if config.segmentation_output_dir is not None
                        else config.output_dir / "segmentation_qc"
                    )
                    tile_suffix = (
                        f"__tile_y{tile_y}_x{tile_x}" if config.tile_size is not None else ""
                    )
                    for signal_spec in config.signal_channels:
                        name = (
                            f"{_safe_name(_safe_stem(path))}__"
                            f"{_safe_name(reference.name)}__"
                            f"{_safe_name(signal_spec.name)}_segmentation"
                            f"{tile_suffix}"
                        )
                        segmentation_paths.extend(
                            save_segmentation_views(
                                reference_image,
                                masks,
                                signals[signal_spec.name],
                                output_dir=qc_dir,
                                name=name,
                                signal_spec=signal_spec,
                                title=(
                                    f"{path.name} | reference={reference.name} | "
                                    f"signal={signal_spec.name}"
                                    + (
                                        f" | tile y={tile_y} x={tile_x}"
                                        if config.tile_size is not None
                                        else ""
                                    )
                                ),
                            )
                        )

            if config.stitch_masks and config.save_masks and tile_masks_buffer:
                H, W = acquisition.data.shape[1:]
                stitched = stitch_masks((H, W), tile_masks_buffer)
                destination = mask_dir / f"{_safe_stem(path)}__{reference.name}_masks.tif"
                tifffile.imwrite(destination, stitched.astype(np.uint32), compression="zlib")
                mask_paths.append(destination)

    cells = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    groups = pd.DataFrame(analyzed_groups)
    images = groups.merge(
        summarize_cells(cells),
        on=["source", "reference_set"],
        how="left",
    )
    for column in (
        "cell_count",
        "total_cell_area_px",
        "mean_cell_area_px",
        "median_cell_area_px",
    ):
        if column not in images:
            images[column] = 0
    images["cell_count"] = images["cell_count"].fillna(0).astype(int)
    images["total_cell_area_px"] = images["total_cell_area_px"].fillna(0).astype(int)
    result = AnalysisResult(
        cells=cells,
        images=images,
        mask_paths=mask_paths,
        segmentation_paths=segmentation_paths,
    )
    result.save_tables(config.output_dir)
    return result


def _reader(config: AnalysisConfig) -> ImageReader:
    return ImageReader(
        time_index=config.time_index,
        scene_index=config.scene_index,
        z_projection=config.z_projection,
    )


def build_dataset(
    config: AnalysisConfig, paths: list[Path] | None = None
) -> ImageDataset:
    """Create the transform-aware dataset used by inspection, analysis, and QC."""
    return ImageDataset(
        paths if paths is not None else _input_paths(config),
        reader=_reader(config),
        transforms=config.resolved_transforms(),
    )


def _input_paths(config: AnalysisConfig) -> list[Path]:
    output = config.output_dir.resolve()
    return [
        path
        for path in discover_images(
            config.input_dir,
            extensions=config.extensions,
            recursive=config.recursive,
        )
        if not path.resolve().is_relative_to(output)
        and not _is_excluded(path, config)
    ]


def _is_excluded(path: Path, config: AnalysisConfig) -> bool:
    """Return whether a discovered image matches an exclusion name or glob."""
    if not config.exclude:
        return False

    resolved = path.resolve().as_posix().casefold()
    relative = path.resolve().relative_to(config.input_dir.resolve()).as_posix().casefold()
    name = path.name.casefold()
    candidates = (name, relative, resolved)
    return any(
        fnmatchcase(candidate, str(pattern).replace("\\", "/").casefold())
        for pattern in config.exclude
        for candidate in candidates
    )


def _safe_stem(path: Path) -> str:
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".oir"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
