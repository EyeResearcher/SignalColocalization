"""Configuration-driven cell segmentation and signal colocalization."""

from .datasets import AnalysisConfig, AnalysisResult, ImageDataset, ReferenceSet, SignalChannel, TissueChannel
from .colocalize import  measure_masks, summarize_cells
from .background import BackgroundResult, background_pixels, compute_background, estimate_background
from .distributions import summarize_signal_ecdf
from .models import resolve_model_source
from .pipeline import build_dataset, inspect_inputs, run_analysis
from .readers import ImageReader, MicroscopyImage, discover_images
from .transforms import MaxProjection, MeanProjection, PercentileNormalize, SelectPlane
from .visualization import (
    plot_cell_distribution_heatmap,
    plot_cell_mean_histograms,
    plot_quantile_profiles,
    plot_signal_ecdf,
    random_crop,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "BackgroundResult",
    "background_pixels",
    "compute_background",
    "ImageDataset",
    "ImageReader",
    "MaxProjection",
    "MeanProjection",
    "MicroscopyImage",
    "PercentileNormalize",
    "ReferenceSet",
    "SignalChannel",
    "TissueChannel",
    "SelectPlane",
    "build_dataset",
    "discover_images",
    "estimate_background",
    "inspect_inputs",
    "measure_masks",
    "resolve_model_source",
    "random_crop",
    "run_analysis",
    "plot_cell_distribution_heatmap",
    "plot_cell_mean_histograms",
    "plot_quantile_profiles",
    "plot_signal_ecdf",
    "summarize_cells",
    "summarize_signal_ecdf",
]
