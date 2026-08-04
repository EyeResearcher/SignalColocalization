"""Configuration-driven cell segmentation and signal colocalization."""

from .datasets import AnalysisConfig, AnalysisResult, ImageDataset, ReferenceSet, SignalChannel
from .models import resolve_model_source
from .pipeline import build_dataset, inspect_inputs, run_analysis
from .readers import ImageReader, MicroscopyImage, discover_images
from .transforms import MaxProjection, MeanProjection, PercentileNormalize, SelectPlane

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ImageDataset",
    "ImageReader",
    "MaxProjection",
    "MeanProjection",
    "MicroscopyImage",
    "PercentileNormalize",
    "ReferenceSet",
    "SignalChannel",
    "SelectPlane",
    "build_dataset",
    "discover_images",
    "inspect_inputs",
    "resolve_model_source",
    "run_analysis",
]
