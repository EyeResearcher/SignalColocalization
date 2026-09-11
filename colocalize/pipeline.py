"""Directory-level orchestration for segmentation and colocalization."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import tifffile

from .colocalize import measure_masks, summarize_cells
from .datasets import AnalysisConfig, AnalysisResult, ImageDataset, ReferenceSet
from .models import CellposeSegmenter
from .readers import ImageReader, MicroscopyImage, discover_images
from .tiling import generate_tiles, stitch_masks
from .visualization import save_segmentation_views


def inspect_inputs(config: AnalysisConfig) -> pd.DataFrame:
    """List discovered image files with their shapes and channel metadata.

    Useful for verifying that the correct files and channels are found before
    committing to a full (expensive) segmentation run.

    Args:
        config: Analysis configuration specifying the input directory, file
            extensions, scene/time indices, and transform chain.

    Returns:
        DataFrame with one row per discovered image.  Columns are ``source``
        (filename), ``path`` (absolute path string), ``shape_cyx`` (array
        dimensions after transforms), and ``channels`` (channel-name tuple).
    """
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
    """Segment and measure every image found under ``config.input_dir``.

    For each input image and each configured reference set, Cellpose is used to
    segment the reference channel.  Every configured signal channel is then
    measured inside those masks.  When ``config.tile_size`` is set the image is
    split into non-overlapping tiles before segmentation and the per-tile
    results are aggregated without double-counting.

    The three inner loops (acquisitions → reference sets → tiles) are each
    isolated in their own helper so that callers can wrap them with a progress
    bar without modifying this function.

    Args:
        config: Fully populated :class:`~colocalize.datasets.AnalysisConfig`
            describing input/output directories, channel assignments, Cellpose
            model settings, and optional tiling parameters.

    Returns:
        :class:`~colocalize.datasets.AnalysisResult` containing the per-cell
        and per-image summary DataFrames plus the paths of any saved mask and
        segmentation QC files.  Tables are also written to
        ``config.output_dir``.

    Raises:
        FileNotFoundError: If no images matching ``config.extensions`` are
            found in ``config.input_dir``.
    """
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
    
        tile_iter = _build_tile_iter(acquisition.data, config)
        for reference in config.reference_sets:
            analyzed_groups.append(
                {"source": acquisition.path.name, "reference_set": reference.name}
            )
            if reference.name not in segmenters:
                segmenters[reference.name] = CellposeSegmenter(
                    reference.model, device=config.device
                )
            ref_tables, ref_mask_paths, ref_seg_paths = _process_reference(
                acquisition, reference, tile_iter, config, segmenters[reference.name], mask_dir
            )
            tables.extend(ref_tables)
            mask_paths.extend(ref_mask_paths)
            segmentation_paths.extend(ref_seg_paths)

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


def _build_tile_iter(
    data: np.ndarray,
    config: AnalysisConfig,
) -> list[tuple[np.ndarray, int, int]]:
    """Return (tile_data, y0, x0) entries covering the image, or one entry for the full image."""
    if config.tile_size is not None:
        return list(generate_tiles(data, config.tile_size))
    return [(data, 0, 0)]


def _process_reference(  # pylint: disable=too-many-arguments
    acquisition: MicroscopyImage,
    reference: ReferenceSet,
    tile_iter: list[tuple[np.ndarray, int, int]],
    config: AnalysisConfig,
    segmenter: CellposeSegmenter,
    mask_dir: Path,
) -> tuple[list[pd.DataFrame], list[Path], list[Path]]:
    """Segment and measure all tiles for one reference set on one acquisition.

    Args:
        acquisition: Transformed CYX image for the current file.
        reference: Reference-set specification (channel, model, segmentation params).
        tile_iter: Pre-built list of ``(tile_data, y0, x0)`` entries from
            :func:`_build_tile_iter`.
        config: Full analysis configuration.
        segmenter: Already-loaded Cellpose model for this reference set.
        mask_dir: Directory in which to write per-tile or stitched mask files.

    Returns:
        Three-tuple of ``(tables, mask_paths, segmentation_paths)`` for this
        reference set.  Each list may be empty if the corresponding save option
        is disabled.
    """
    # Build tile images once; all reference channels sent to Cellpose as a single batch.
    tile_acqs = [
        (
            MicroscopyImage(
                path=acquisition.path, data=td, channel_names=acquisition.channel_names
            ),
            ty, tx,
        )
        for td, ty, tx in tile_iter
    ]
    all_masks = segmenter.segment_batch(
        [ta.channel(reference.channel) for ta, _, _ in tile_acqs],
        diameter=reference.diameter,
        flow_threshold=reference.flow_threshold,
        cellprob_threshold=reference.cellprob_threshold,
        min_size=reference.min_size,
        normalize=reference.normalize,
    )

    tables: list[pd.DataFrame] = []
    mask_paths: list[Path] = []
    seg_paths: list[Path] = []
    tile_masks_buffer: list[tuple[int, int, np.ndarray]] = []

    measure_iter: zip | tqdm = zip(tile_acqs, all_masks)
    if len(tile_acqs) > 1:
        measure_iter = tqdm(
            measure_iter, total=len(tile_acqs), desc="measuring tiles", leave=False, unit="tile"
        )
    print(f"Starting tile measuring for {acquisition.path.name}")
    for (tile_acq, tile_y, tile_x), masks in measure_iter:
        table, tile_seg_paths = _measure_tile(
            tile_acq, masks, tile_y, tile_x, acquisition.path, reference, config
        )
        table.insert(0, "reference_set", reference.name)
        table.insert(0, "source", acquisition.path.name)
        tables.append(table)
        seg_paths.extend(tile_seg_paths)

    print(f"Tile measuring complete for {acquisition.path.name}, starting saving segmentations")

    save_seg_iter = tqdm(
        zip(tile_acqs, all_masks), total=len(tile_acqs), desc="saving segmentations", leave=False, unit="tile"
    )
    for (tile_acq, tile_y, tile_x), masks in save_seg_iter:
        if config.save_masks:
            if config.stitch_masks:
                tile_masks_buffer.append((tile_y, tile_x, masks))
            elif config.tile_size is not None:
                destination = (
                    mask_dir
                    / f"{_safe_stem(acquisition.path)}__{reference.name}"
                    f"__tile_y{tile_y}_x{tile_x}_masks.tif"
                )
                tifffile.imwrite(destination, masks, compression="zlib")
                mask_paths.append(destination)
            else:
                destination = (
                    mask_dir / f"{_safe_stem(acquisition.path)}__{reference.name}_masks.tif"
                )
                tifffile.imwrite(destination, masks, compression="zlib")
                mask_paths.append(destination)

    if config.stitch_masks and config.save_masks and tile_masks_buffer:
        img_h, img_w = acquisition.data.shape[1:]
        stitched = stitch_masks((img_h, img_w), tile_masks_buffer)
        destination = mask_dir / f"{_safe_stem(acquisition.path)}__{reference.name}_masks.tif"
        tifffile.imwrite(destination, stitched.astype(np.uint32), compression="zlib")
        mask_paths.append(destination)

    return tables, mask_paths, seg_paths


def _measure_tile(  # pylint: disable=too-many-arguments
    tile_acq: MicroscopyImage,
    masks: np.ndarray,
    tile_y: int,
    tile_x: int,
    path: Path,
    reference: ReferenceSet,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute measurements and optional QC output for one pre-segmented tile."""
    signal_data = {
        spec.name: (tile_acq.channel(spec.channel), spec)
        for spec in config.signal_channels
    }
    reference_image = tile_acq.channel(reference.channel)
    table = measure_masks(
        reference_image=reference_image,
        masks=masks,
        signal_data=signal_data,
        tile_yx=(tile_y, tile_x),
    )

    seg_paths: list[Path] = []
    if config.save_segmentation:
        qc_dir = (
            config.segmentation_output_dir
            if config.segmentation_output_dir is not None
            else config.output_dir / "segmentation_qc"
        )
        tile_suffix = f"__tile_y{tile_y}_x{tile_x}" if config.tile_size is not None else ""
        for signal_spec in config.signal_channels:
            name = (
                f"{_safe_name(_safe_stem(path))}__"
                f"{_safe_name(reference.name)}__"
                f"{_safe_name(signal_spec.name)}_segmentation"
                f"{tile_suffix}"
            )
            seg_paths.extend(
                save_segmentation_views(
                    reference_image,
                    masks,
                    signal_data[signal_spec.name][0],
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

    return table, seg_paths


def _reader(config: AnalysisConfig) -> ImageReader:
    """Construct an ImageReader from the projection/scene/time settings in config."""
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
    """Return every image path under input_dir that is not inside output_dir or excluded."""
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
    """Return the filename stem with known multi-part microscopy extensions stripped."""
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".oir"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _safe_name(value: str) -> str:
    """Replace characters that are not alphanumeric, hyphens, or underscores with underscores."""
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
