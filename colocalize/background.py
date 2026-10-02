"""Selectable background workflows and their pixel-level diagnostics."""

from dataclasses import dataclass

from skimage.measure import regionprops, label
from skimage.morphology import closing, disk, ball

from .datasets import BACKGROUND_METHODS, SignalChannel
import numpy as np
from scipy import ndimage as ndi



class CellMaskSummary:
    def __init__(
        self,
        masks: np.ndarray,
        percentile: float = 50,
        properties=("area",),
    ):
        
        self.masks = np.asarray(masks)

        if self.masks.ndim not in (2, 3):
            raise ValueError("Cell masks must be 2D or 3D.")
        if not np.issubdtype(self.masks.dtype, np.integer):
            raise TypeError("Cell masks must contain integer labels.")

        if isinstance(properties, str):
            properties = (properties,)
        self.property_names = tuple(dict.fromkeys(properties))

        # Preserve the original cell identities.
        self.props = regionprops(self.masks)
        self.n_cells = len(self.props)

        self.property_values = {}
        for name in self.property_names:
            values = np.asarray(
                [getattr(region, name) for region in self.props],
                dtype=float,
            )
            if values.ndim != 1:
                raise ValueError(f"Debris filtering requires scalar properties: {name!r}.")
            self.property_values[name] = values

        self.mean_sd_props_cache = self.mean_sd_props()
        self.percentile_props_cache = self.percentile_props(percentile)

    def mean_sd_props(self):
        return {
            name: (float(values.mean()), float(values.std()))
            for name, values in self.property_values.items()
            if values.size
        }

    def percentile_props(self, percentile):
        return {
            name: float(np.percentile(values, percentile))
            for name, values in self.property_values.items()
            if values.size
        }

    
class BackgroundSignal:
    """Encapsulates background signal estimation for an image."""

    def __init__(self, 
                 signal : np.ndarray, 
                 cell_mask_summary : CellMaskSummary):
        self.signal = np.asarray(signal, dtype=float)
        self.background = self.signal.copy()
        self.masks_summary = cell_mask_summary

    def tissue_mask(self, 
                    tissue : np.ndarray,
                    tissue_threshold : float) -> np.ndarray[bool]:
        """Apply a tissue mask to the background signal based on a threshold.

        Args:
            tissue (np.ndarray): Array representing tissue signal.
            tissue_threshold (float): Threshold to create the tissue mask.

        Returns:
            np.ndarray[bool]: Boolean mask of tissue regions.
        """
        tissue_mask = tissue > tissue_threshold
        self.background = self.background * tissue_mask
        return tissue_mask

    def exclude_cells(self, 
                      buffer_px : int = 0) -> np.ndarray[bool]:
        """Exclude cells from the background signal based on a cell mask.

        Args:
            buffer_px (int, optional): Number of pixels to dilate the cell mask. Defaults to 0.

        Returns:
            np.ndarray[bool]: Boolean mask of excluded cells.
        """
        cell_mask = self.masks_summary.masks > 0
        if buffer_px:
            cell_mask = ndi.binary_dilation(cell_mask, iterations=buffer_px)
        self.background = self.background * ~cell_mask
        return cell_mask
    
    def _debris_obj_thresh(self, property_name : str,
                           cell_mask_summary : CellMaskSummary,
                           n_sd = 0) -> float:
        if n_sd == 0:
            return cell_mask_summary.percentile_props_cache[property_name]
         
        mean, sd = cell_mask_summary.mean_sd_props_cache[property_name]
        return mean + n_sd * sd

    def _debris_thresh_func(self, debris_obj, property_name: str | list[str], n_sd = 0) -> None:

        if not isinstance(property_name, (str, list)):
            raise ValueError("debris_thresh_property must be a string or a list of strings.")
        if isinstance(property_name, list) and not all(isinstance(item, str) for item in property_name):
            raise ValueError("All items in debris_thresh_property list must be strings.")
        
        for prop in property_name if isinstance(property_name, list) else [property_name]:
            if debris_obj[prop] <= self._debris_obj_thresh(prop, self.masks_summary, n_sd=n_sd):
                return False
        return True
        
    def exclude_bright_debris(self, 
                              raw_threshold_percentile : float, 
                              threshold_property : str, 
                              *,
                              disk_r = 2, n_sd = 0) -> np.ndarray[bool]:
        """This method excludes bright debris objects from the background signal based on a raw intensity threshold and a specified property threshold.

            Args:
                raw_threshold_percentile (float): The raw intensity threshold percentile for candidate debris.
                threshold_property (str): The property of debris objects to threshold.
                disk_r (int, optional): Radius for binary closing. Defaults to 2.
                n_sd (float, optional): Number of standard deviations above the mean for the property threshold. Defaults to 0.
            Returns:
                np.ndarray[bool]: Boolean mask of excluded debris objects.
            """
        if self.masks_summary.n_cells == 0:
            return np.zeros(self.signal.shape, dtype=bool)
        
        raw_threshold = np.percentile(self.background, raw_threshold_percentile)
        candidate_mask = self.background > raw_threshold
        footprint = (disk(disk_r) if candidate_mask.ndim == 2
                else ball(disk_r))
        debris_candidates = closing(candidate_mask, footprint)

        debris_labels = label(debris_candidates)
        debris_objects = [debris.label for debris in regionprops(debris_labels) if self._debris_thresh_func(debris, threshold_property, n_sd=n_sd)]
        debris_mask = np.isin(debris_labels, debris_objects)
        self.background = self.background * ~debris_mask
        return debris_mask

    def get_background(self, percentile) -> float:
        """Return the current background intensity level at the given percentile."""
        return np.percentile(self.background, percentile)
@dataclass
class BackgroundComputationSummary:
    """Summary of the background computation for quality control purposes.

    **Fields**:
        background (float): The computed background intensity level.
        sigma (float): The robust estimate of the background noise level.
        sample_mask (np.ndarray): Exact pixels used for background computation, available for QC.
        buffered_cells (np.ndarray): Cells with buffer applied, available for QC.
        background_candidates (np.ndarray): Pixels considered as background candidates, available for QC.
        bright_debris (np.ndarray): Pixels identified as bright debris, available for QC.
        background_candidates_clean (np.ndarray): Background candidates after removing bright debris, available for QC.
        tissue_for_background (np.ndarray): Tissue mask used for background computation, available for QC.
        debris_cutoff (float): Debris intensity cutoff used, available for QC.
        min_debris_area (int): Minimum area for debris objects, available for QC.
        otsu_cutoff (float): Otsu threshold used for tissue segmentation, available for QC.
        method (str): Method used for background computation, available for QC.
        debris_filter_applied (bool): Whether debris filter was applied, available for QC.
        """
    background: float = np.nan
    sigma: float = np.nan
    sample_mask: np.ndarray | None = None  # Exact pixels used, available for QC
    tissue: np.ndarray | None = None  # Tissue mask used to constrain sampling
    buffered_cells: np.ndarray | None = None  # Cells with buffer applied, available for QC
    background_candidates: np.ndarray | None = None  # Pixels considered as background candidates, available for QC
    bright_debris: np.ndarray | None = None  # Pixels identified as bright debris, available for QC

    background_candidates_clean: np.ndarray | None = None  # Background candidates after removing bright debris, available for QC
    tissue_for_background: np.ndarray | None = None  # Tissue mask used for background computation, available for QC
    debris_cutoff: float = np.nan  # Debris intensity cutoff used, available for QC
    min_debris_area: int = 0  # Minimum area for debris objects, available for QC
    otsu_cutoff: float | None = None  # Reserved diagnostic; unsupported methods do not use Otsu
    method: str = ""  # Method used for background computation, available for QC
    debris_filter_applied: bool = False  # Whether debris filter was applied, available for QC

def _robust_sigma(values):
    """Return 1.4826 times the finite sample's median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if values.size else np.nan


def _buffered_cells(masks, buffer_px):
    if not isinstance(buffer_px, (int, np.integer)) or buffer_px < 0:
        raise ValueError("background_buffer_px must be a nonnegative integer.")
    cells = np.asarray(masks) > 0
    return ndi.binary_dilation(cells, iterations=buffer_px) if buffer_px else cells.copy()


@dataclass
class BackgroundResult:
    level: float
    sigma: float
    sample_mask: np.ndarray  # Exact pixels used, available for QC



def background_pixels(
    image,
    cell_masks,
    *,
    buffer_px=3,
    region_mask=None,
):
    """Select finite pixels outside cells and their optional buffer."""
    image = np.asarray(image)
    cell_masks = np.asarray(cell_masks)

    if image.shape != cell_masks.shape:
        raise ValueError("Image and cell masks must have matching shapes.")
    excluded = _buffered_cells(cell_masks, buffer_px)

    selected = np.isfinite(image) & ~excluded

    # Optional tissue mask or manually selected background region.
    if region_mask is not None:
        region_mask = np.asarray(region_mask, dtype=bool)
        if region_mask.shape != image.shape:
            raise ValueError("Region mask must match the image shape.")
        selected &= region_mask

    return selected


def estimate_background(
    image,
    sample_mask=None,
    *,
    method="median",
    percentile=10,
    fixed_value=None,
):
    """Estimate a scalar background without choosing or filtering pixels."""
    image = np.asarray(image, dtype=float)
    empty_mask = np.zeros(image.shape, dtype=bool)

    # These methods need no background sampling.
    if method == "none":
        return BackgroundResult(0.0, np.nan, empty_mask)

    if method == "fixed":
        if fixed_value is None or not np.isfinite(fixed_value):
            raise ValueError("Provide a finite fixed_value.")
        return BackgroundResult(float(fixed_value), np.nan, empty_mask)

    if method not in {"median", "percentile"}:
        raise ValueError(f"Unknown background method: {method!r}")

    if sample_mask is None:
        raise ValueError("This method requires a background sample mask.")

    selected = np.asarray(sample_mask, dtype=bool)
    if selected.shape != image.shape:
        raise ValueError("Sample mask must match the image shape.")

    selected = selected & np.isfinite(image)
    values = image[selected]
    if values.size == 0:
        raise ValueError("No finite background pixels remain.")

    median = np.median(values)

    if method == "median":
        level = median
    else:
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100.")
        level = np.percentile(values, percentile)

    # Estimate noise around the sample median, independently of the floor.
    sigma = _robust_sigma(values)

    return BackgroundResult(float(level), float(sigma), selected)


def _val_selected_method(spec: SignalChannel, method: str | None):
    """Validate and determine the selected background method for a given signal channel.

    Args:
        spec: The signal channel specification.
        method: The user-specified background method, if any.
    Returns:
        The validated and selected background method as a string.
    """
    selected_method = method if method is not None else spec.background_method
    if selected_method not in BACKGROUND_METHODS:
        raise ValueError(f"Unknown background method: {selected_method!r}")

    return selected_method

def compute_background(
    signal: np.ndarray,
    masks: np.ndarray,
    dapi: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    spec: SignalChannel | None = None,
    *,
    method: str | None = None,
    percentile: float | None = None,
    fixed_value: float | None = None,
    region_mask: np.ndarray | None = None,
) -> dict:
    """Dispatch once per signal/reference-mask set and return common diagnostics.

    Explicit ``method`` overrides the signal configuration.

    Simple methods use all finite pixels outside buffered cells unless an
    explicit boolean ``region_mask`` is supplied. DAPI is only needed by the
    full workflow. No-subtraction and fixed methods do not estimate noise.
    """
    if spec is None:
        spec = SignalChannel(name="signal", channel=0, background_method="median")

    selected_method = _val_selected_method(spec, method)

    signal = np.asarray(signal, dtype=float)
    masks = np.asarray(masks)

    if signal.ndim not in (2, 3) or signal.shape != masks.shape:
        raise ValueError(
            f"Signal and cell masks must be matching 2D or 3D arrays. Got signal shape {signal.shape} and masks shape {masks.shape}."
        )
    
    if selected_method == "tissue_filtered_oop":
        if dapi is None or reference is None:
            raise ValueError(
                "tissue_filtered_oop requires a tissue/DAPI image and reference image."
            )
        for name, image in (("dapi", dapi), ("reference", reference)):
            if np.shape(image) != signal.shape:
                raise ValueError(
                    f"{name} must match the signal shape {signal.shape}."
                )
        if region_mask is not None:
            raise ValueError("region_mask is for scalar sampling methods; tissue_filtered_oop derives its domain from DAPI/reference.")
        cell_mask_summary = CellMaskSummary(masks,
                                            percentile=spec.prop_percentile_thresh,
                                            properties=spec.debris_thresh_property)
        background_signal = BackgroundSignal(signal, cell_mask_summary)
        background_computation_summary = BackgroundComputationSummary()
        tissue_mask = background_signal.tissue_mask(
            dapi, spec.background_dapi_threshold
        )
        buffered_cells = background_signal.exclude_cells(
            buffer_px=spec.background_buffer_px
        )
        debris_cutoff = float(np.percentile(background_signal.background, spec.debris_percentile))
        background_computation_summary.bright_debris = background_signal.exclude_bright_debris(raw_threshold_percentile=spec.debris_percentile,
                                                              threshold_property=spec.debris_thresh_property,
                                                              disk_r=spec.debris_buffer_px,
                                                              n_sd=spec.debris_min_area_cell_sd)
        background_candidates = (
            tissue_mask & ~buffered_cells & np.isfinite(signal)
        )
        sample_mask = (
            background_candidates
            & ~background_computation_summary.bright_debris
        )
        background_values = np.asarray(signal, dtype=float)[sample_mask]
        if background_values.size == 0:
            raise ValueError("No finite background pixels remain after filtering.")

        background_computation_summary.sample_mask = sample_mask
        background_computation_summary.tissue = tissue_mask
        background_computation_summary.buffered_cells = buffered_cells
        background_computation_summary.background_candidates = background_candidates
        background_computation_summary.background_candidates_clean = sample_mask
        background_computation_summary.tissue_for_background = sample_mask
        background_computation_summary.debris_cutoff = debris_cutoff
        if "area" in background_signal.masks_summary.property_names:
            background_computation_summary.min_debris_area = int(np.ceil(
                background_signal._debris_obj_thresh(
                    "area",
                    background_signal.masks_summary,
                    n_sd=spec.debris_min_area_cell_sd,
                )
            ))
        background_computation_summary.background = float(
            np.percentile(background_values, spec.background_percentile)
        )
        background_computation_summary.sigma = _robust_sigma(background_values)
        background_computation_summary.method = selected_method
        background_computation_summary.debris_filter_applied = True
        return background_computation_summary.__dict__
    
    empty = np.zeros(signal.shape, dtype=bool)
    if selected_method in {"none", "fixed"}:
        # These modes bypass pixel sampling.
        candidates = empty.copy()
        domain = empty.copy()
        buffered = empty.copy()
        result = estimate_background(
            signal, method=selected_method,
            fixed_value=spec.background_fixed_value if fixed_value is None else fixed_value,
        )
    else:
        candidates = background_pixels(
            signal, masks, buffer_px=spec.background_buffer_px, region_mask=region_mask
        )
        domain = np.ones(signal.shape, dtype=bool) if region_mask is None else np.asarray(region_mask, dtype=bool)
        buffered = _buffered_cells(masks, spec.background_buffer_px)
        result = estimate_background(
            signal, candidates, method=selected_method,
            percentile=spec.background_anchor_percentile if percentile is None else percentile,
        )

    return {
        "background": result.level,
        "sigma": result.sigma,
        "sample_mask": result.sample_mask,
        "method": selected_method,
        "tissue": domain,
        "buffered_cells": buffered,
        "background_candidates": candidates,
        "bright_debris": empty.copy(),
        "background_candidates_clean": candidates.copy(),
        "tissue_for_background": result.sample_mask,
        "debris_cutoff": None,
        "min_debris_area": None,
        "otsu_cutoff": None,
        "debris_filter_applied": False,
    }



