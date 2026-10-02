"""Directory-level orchestration for segmentation and colocalization."""

from __future__ import annotations

from tqdm import tqdm
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile

from .colocalize import measure_masks, summarize_cells
from .background import compute_background
from .datasets import AnalysisConfig, AnalysisResult, ImageDataset, ReferenceSet, SignalChannel
from .models import CellposeSegmenter
from .readers import ImageReader, MicroscopyImage, discover_images
from .visualization import make_segmentation_views
from .io import save_mask, save_segmentation_views


def _align_image_to_masks(
    image: np.ndarray,
    masks: np.ndarray,
    *,
    image_name: str,
    promote_2d_to_3d: bool,
) -> np.ndarray:
    """Return an image array aligned to mask dimensionality for measurement."""
    image_array = np.asarray(image)
    mask_array = np.asarray(masks)
    if image_array.shape == mask_array.shape:
        return image_array
    if (
        promote_2d_to_3d
        and mask_array.ndim == 3
        and image_array.ndim == 2
        and image_array.shape == mask_array.shape[1:]
    ):
        return np.broadcast_to(image_array, mask_array.shape)
    raise ValueError(
        f"{image_name} shape {image_array.shape} does not match masks shape {mask_array.shape}. "
        "Enable AnalysisConfig.promote_2d_to_3d to broadcast 2D images across Z for 3D masks."
    )


def inspect_inputs(config: AnalysisConfig) -> pd.DataFrame:
    """List discovered image files with their post-transform shapes and channel names."""
    return pd.DataFrame(
        {
            "source": image.path.name,
            "path": str(image.path),
            "shape_cyx": tuple(image.data.shape),
            "channels": tuple(image.channel_names),
        }
        for image in build_dataset(config)
    )
def _get_masks(
    mask_dir: Path, 
    acquisition: MicroscopyImage, 
    reference: ReferenceSet, 
    config: AnalysisConfig,
) -> tuple[np.ndarray, CellposeSegmenter | None]:
    
    source_dir = Path(config.mask_dir) if config.mask_dir is not None else mask_dir
    mask_path = source_dir / f"{_safe_stem(acquisition.path)}__{reference.name}_masks.tif"
    print(mask_path)
    if config.load_masks and mask_path.is_file():

        return tifffile.imread(mask_path), None

    segmenter = CellposeSegmenter(
        reference.model, device=config.device
    )

    masks, _ = segmenter.segment(
        acquisition.channel(reference.channel),
        progress=config.progress,
        **reference.cellpose_kwargs,

    )
    return masks, segmenter
def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Segment each reference set, measure signals, and optionally write outputs.

    The optional raw tissue channel is loaded once per acquisition. Background
    estimation is selected per signal and performed after each reference set
    is segmented, before per-cell measurement.
    """
    paths = _input_paths(config)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {config.input_dir}.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = config.output_dir / "masks"
    if config.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = (
        config.segmentation_output_dir
        if config.segmentation_output_dir is not None
        else config.output_dir / "segmentation_qc"
    )

    segmenters: dict[str, CellposeSegmenter] = {}
    tables: list[pd.DataFrame] = []
    mask_paths: list[Path] = []
    segmentation_paths: list[Path] = []
    analyzed_groups: list[dict[str, str]] = []
    progress = tqdm(total=len(build_dataset(config, paths)), desc="Processing acquisitions")
    for acquisition in build_dataset(config, paths):

        tissue = acquisition.channel(config.tissue_channel.channel) if config.tissue_channel is not None else None
        for reference in config.reference_sets:
            analyzed_groups.append({"source": acquisition.path.name, "reference_set": reference.name})
            masks, segmenter = _get_masks(
                mask_dir=mask_dir,
                acquisition=acquisition,
                reference=reference,
                config=config,
            )

            if reference.name not in segmenters and segmenter is not None:
                segmenters[reference.name] = segmenter

            # Loading an existing mask does not allocate GPU memory. Avoid
            # initializing/touching CUDA in that path: besides adding overhead,
            # a driver-level CUDA failure can terminate Python without a
            # catchable traceback. Only clear the cache after segmentation.
            if segmenter is not None:
                _free_gpu_cache()

            table, backgrounds = _process_signal(
                acquisition=acquisition,
                masks=masks,
                reference=reference,
                signal_channels=config.signal_channels,
                tissue=tissue,
                morphology_properties=config.morphology_properties,
                promote_2d_to_3d=config.promote_2d_to_3d,
            )
            if config.progress:

                print(f"Processing acquisition {acquisition.path.name}, reference set {reference.name}")
                print(table.head())
            table.insert(0, "reference_set", reference.name)
            table.insert(0, "source", acquisition.path.name)
            tables.append(table)

            mask_paths.append(
                save_mask(
                    masks=masks,
                    acquisition_path=acquisition.path,
                    reference=reference,
                    output_dir=mask_dir,
                    save_masks=config.save_masks,
                )
            )
            if config.save_segmentation:
                for spec in config.signal_channels:
                    reference_image = _align_image_to_masks(
                        acquisition.channel(reference.channel),
                        masks,
                        image_name=f"reference channel {reference.name!r}",
                        promote_2d_to_3d=config.promote_2d_to_3d,
                    )
                    signal_image = _align_image_to_masks(
                        acquisition.channel(spec.channel),
                        masks,
                        image_name=f"signal channel {spec.name!r}",
                        promote_2d_to_3d=config.promote_2d_to_3d,
                    )
                    seg_figures = make_segmentation_views(
                        reference=reference_image,
                        masks=masks,
                        signal=signal_image,
                        signal_spec=spec,
                        title=f"{acquisition.path.name} | reference={reference.name}",
                        z_index=None,
                    )
                    seg_paths = save_segmentation_views(
                        figures=seg_figures,
                        output_dir=qc_dir,
                        name=f"{acquisition.path.stem}__{reference.name}_{spec.name}"
                    )
                    segmentation_paths.extend(seg_paths)

        progress.update(1)
    cells = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    images = pd.DataFrame(analyzed_groups).merge(
        summarize_cells(cells), on=["source", "reference_set"], how="left"
    )
    images["cell_count"] = images["cell_count"].fillna(0).astype(int)

    for size_column in ("area_px", "volume_voxels"):
        total_column = f"total_cell_{size_column}"
        if total_column in images.columns:
            images.loc[
                images["cell_count"].eq(0), total_column
            ] = 0
    result = AnalysisResult(cells, images, mask_paths, segmentation_paths)
    result.save_tables(config.output_dir)
    return result


def _process_signal(
    *,
    acquisition: MicroscopyImage,
    masks: np.ndarray,
    reference: ReferenceSet,
    signal_channels: Sequence[SignalChannel],
    tissue: np.ndarray | None = None,
    morphology_properties: tuple[str, ...] | list[str] | None = None,
    promote_2d_to_3d: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Estimate each signal's background, then measure raw channels once.

    ``tissue`` is an optional raw DAPI/tissue image, not a boolean mask.
    A per-signal tissue_channel overrides it. Full methods require that input;
    scalar and no-subtraction methods do not. Diagnostics are returned alongside
    the table for QC of the exact pixels used by measurement.
    """
    reference_image = _align_image_to_masks(
        acquisition.channel(reference.channel),
        masks,
        image_name=f"reference channel {reference.name!r}",
        promote_2d_to_3d=promote_2d_to_3d,
    )
    shared_tissue = (
        None
        if tissue is None
        else _align_image_to_masks(
            tissue,
            masks,
            image_name="configured tissue channel",
            promote_2d_to_3d=promote_2d_to_3d,
        )
    )
    signal_data = {}
    for spec in signal_channels:
        signal_data[spec.name] = (
            _align_image_to_masks(
                acquisition.channel(spec.channel),
                masks,
                image_name=f"signal channel {spec.name!r}",
                promote_2d_to_3d=promote_2d_to_3d,
            ),
            spec,
        )

    backgrounds = {}
    for name, (signal, spec) in signal_data.items():
        tissue_image = (
            _align_image_to_masks(
                acquisition.channel(spec.tissue_channel),
                masks,
                image_name=f"signal tissue channel for {spec.name!r}",
                promote_2d_to_3d=promote_2d_to_3d,
            )
            if spec.tissue_channel is not None
            else shared_tissue
        )

        backgrounds[name] = compute_background(
            signal, masks, dapi=tissue_image, reference=reference_image, spec=spec
        )

    table = measure_masks(
        reference_image=reference_image,
        masks=masks,
        signal_data=signal_data,
        backgrounds=backgrounds,
        morphology_properties=morphology_properties,
    )
    return table, backgrounds


def _free_gpu_cache() -> None:
    """Release cached GPU allocations between full-resolution images."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _reader(config: AnalysisConfig) -> ImageReader:
    return ImageReader(time_index=config.time_index, scene_index=config.scene_index, z_projection=config.z_projection)


def build_dataset(config: AnalysisConfig, paths: list[Path] | None = None) -> ImageDataset:
    """Create the transform-aware dataset used by inspection and analysis."""
    return ImageDataset(
        paths if paths is not None else _input_paths(config),
        reader=_reader(config),
        transforms=config.resolved_transforms(),
    )


def _input_paths(config: AnalysisConfig) -> list[Path]:
    """Return input images excluding outputs and user-provided patterns."""
    output = config.output_dir.resolve()
    return [
        path
        for path in discover_images(config.input_dir, extensions=config.extensions, recursive=config.recursive)
        if not path.resolve().is_relative_to(output) and not _is_excluded(path, config)
    ]


def _is_excluded(path: Path, config: AnalysisConfig) -> bool:
    if not config.exclude:
        return False
    resolved = path.resolve().as_posix().casefold()
    relative = path.resolve().relative_to(config.input_dir.resolve()).as_posix().casefold()
    candidates = (path.name.casefold(), relative, resolved)
    return any(
        fnmatchcase(candidate, str(pattern).replace("\\", "/").casefold())
        for pattern in config.exclude
        for candidate in candidates
    )


def _safe_stem(path: Path) -> str:
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".oir"):
        if name.casefold().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
