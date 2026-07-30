"""Calibrate image-level signal thresholds from point annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu


ANNOTATION_COLUMNS = ["Area", "Mean", "Min", "Max", "X", "Y", "Ch", "Counter", "Count"]


def _read_incremental_annotations(folder: Path) -> pd.DataFrame:
    csv_paths = sorted(
        (
            path
            for path in folder.glob("*.csv")
            if path.name.casefold() != "manifest.csv"
        ),
        key=lambda path: (path.stat().st_mtime, path.name.casefold()),
    )
    if not csv_paths:
        raise FileNotFoundError(f"No annotation CSV files found in {folder}")

    previous = pd.DataFrame()
    tables: list[pd.DataFrame] = []
    for file_index, path in enumerate(csv_paths):
        current = pd.read_csv(path)
        current = current.loc[:, [column for column in ANNOTATION_COLUMNS if column in current]]
        if set(ANNOTATION_COLUMNS) - set(current):
            missing = sorted(set(ANNOTATION_COLUMNS) - set(current))
            raise ValueError(f"{path.name} is missing annotation columns: {missing}")

        prefix_length = 0
        if not previous.empty and len(current) >= len(previous):
            candidate = current.iloc[: len(previous)].reset_index(drop=True)
            if candidate.equals(previous.reset_index(drop=True)):
                prefix_length = len(previous)
        incremental = current.iloc[prefix_length:].copy()
        incremental = incremental.drop_duplicates(
            subset=["X", "Y", "Ch", "Counter"], keep="first"
        )

        counters = set(incremental["Counter"].dropna().astype(int))
        if file_index == 0:
            incremental["label"] = 1 - incremental["Counter"].astype(int)
            label_rule = "first_csv_reversed"
        elif len(counters) <= 1:
            incremental["label"] = 0
            label_rule = "single_point_type_all_negative"
        else:
            incremental["label"] = incremental["Counter"].astype(int)
            label_rule = "counter_0_negative_1_positive"

        stem = path.stem
        image_path = folder / f"{stem}__zmax_channels.ome.tif"
        if not image_path.exists():
            raise FileNotFoundError(f"No paired OME-TIFF found for {path.name}")
        incremental["source_csv"] = path.name
        incremental["source_image"] = image_path.name
        incremental["label_rule"] = label_rule
        tables.append(incremental)
        previous = current

    annotations = pd.concat(tables, ignore_index=True)
    annotations.insert(0, "annotation_id", np.arange(1, len(annotations) + 1))
    return annotations


def _otsu(image: np.ndarray) -> float:
    values = np.asarray(image)
    if values.size == 0 or np.all(values == values.flat[0]):
        return float(values.flat[0]) if values.size else np.nan
    return float(threshold_otsu(values))


def _percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    return float(np.searchsorted(sorted_values, value, side="right") / len(sorted_values))


def _local_values(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    center_x = int(round(x))
    center_y = int(round(y))
    y0, y1 = max(0, center_y - radius), min(image.shape[0], center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(image.shape[1], center_x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    return image[y0:y1, x0:x1][disk]


def _add_image_features(
    annotations: pd.DataFrame,
    folder: Path,
    *,
    gd_channel: int,
    reference_channel: int,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for image_name, group in annotations.groupby("source_image", sort=False):
        image = tifffile.imread(folder / image_name)
        if image.ndim != 3:
            raise ValueError(f"Expected CYX image in {image_name}; received {image.shape}")
        if max(gd_channel, reference_channel) >= image.shape[0]:
            raise IndexError(f"Requested channel is unavailable in {image_name}: {image.shape}")

        table = group.copy()
        for prefix, channel_index in (
            ("gd", gd_channel),
            ("reference", reference_channel),
        ):
            channel = np.asarray(image[channel_index], dtype=float)
            otsu = _otsu(channel)
            sorted_values = np.sort(channel.ravel())
            table[f"{prefix}_image_otsu"] = otsu
            for radius in (0, 1, 2, 3, 5):
                local = [
                    _local_values(channel, row.X, row.Y, radius)
                    for row in table.itertuples()
                ]
                table[f"{prefix}_r{radius}_mean"] = [
                    float(np.mean(values)) for values in local
                ]
                table[f"{prefix}_r{radius}_max"] = [
                    float(np.max(values)) for values in local
                ]
                table[f"{prefix}_r{radius}_mean_over_otsu"] = (
                    table[f"{prefix}_r{radius}_mean"] / otsu
                    if otsu > 0
                    else np.nan
                )
                table[f"{prefix}_r{radius}_mean_percentile"] = [
                    _percentile_rank(sorted_values, value)
                    for value in table[f"{prefix}_r{radius}_mean"]
                ]
        output.append(table)
    return pd.concat(output, ignore_index=True)


def _metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(labels & predicted))
    tn = int(np.sum(~labels & ~predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
        "precision": precision,
        "f1": 2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0,
    }


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values[np.isfinite(values)])
    if unique.size == 1:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2
    return np.r_[np.nextafter(unique[0], -np.inf), midpoints, unique[-1]]


def _fit_one_feature(table: pd.DataFrame, feature: str) -> dict[str, float | int | str]:
    values = table[feature].to_numpy(float)
    labels = table["label"].to_numpy(int)
    candidates = _candidate_thresholds(values)
    results = []
    for threshold in candidates:
        metrics = _metrics(labels, values >= threshold)
        results.append(
            {
                "feature": feature,
                "threshold": float(threshold),
                **metrics,
            }
        )
    return max(
        results,
        key=lambda row: (
            row["balanced_accuracy"],
            row["specificity"],
            row["sensitivity"],
            row["f1"],
        ),
    )


def _fit_fraction_cutoff(
    labels: np.ndarray,
    fractions: np.ndarray,
) -> dict[str, float | int]:
    results = []
    for cutoff in _candidate_thresholds(fractions):
        metrics = _metrics(labels, fractions >= cutoff)
        results.append({"positive_fraction_cutoff": float(cutoff), **metrics})
    return max(
        results,
        key=lambda row: (
            row["balanced_accuracy"],
            row["specificity"],
            row["sensitivity"],
            row["f1"],
        ),
    )


def _calibrate_annotated_cells(
    points: pd.DataFrame,
    image_dir: Path,
    mask_dir: Path,
    *,
    gd_channel: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, object]] = []
    image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for image_name, group in points.groupby("source_image", sort=False):
        stem = image_name.removesuffix("__zmax_channels.ome.tif")
        mask_path = mask_dir / f"{stem}__BRN3A_masks.tif"
        if not mask_path.exists():
            raise FileNotFoundError(f"No cell mask found for {image_name}: {mask_path}")
        gd = np.asarray(tifffile.imread(image_dir / image_name)[gd_channel], dtype=float)
        masks = np.asarray(tifffile.imread(mask_path))
        if gd.shape != masks.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {image_name}: {gd.shape} vs {masks.shape}"
            )
        image_cache[image_name] = (gd, masks)
        for row in group.itertuples():
            y, x = int(round(row.Y)), int(round(row.X))
            mask_label = int(masks[y, x])
            if mask_label == 0:
                continue
            values = gd[masks == mask_label]
            cells.append(
                {
                    "source_image": image_name,
                    "mask_label": mask_label,
                    "label": int(row.label),
                    "area_px": int(values.size),
                    "gd_mean": float(np.mean(values)),
                    "gd_max": float(np.max(values)),
                    "gd_image_otsu": _otsu(gd),
                    "_values": values,
                }
            )

    cell_table = pd.DataFrame(cells)
    conflicts = (
        cell_table.groupby(["source_image", "mask_label"])["label"].nunique().gt(1)
    )
    if conflicts.any():
        raise ValueError(
            f"{int(conflicts.sum())} cell masks contain conflicting point labels."
        )
    cell_table = cell_table.drop_duplicates(["source_image", "mask_label"]).reset_index(
        drop=True
    )
    labels = cell_table["label"].to_numpy(int)
    cell_values = list(cell_table["_values"])

    fits: list[dict[str, object]] = []
    otsu_fractions = np.array(
        [
            np.mean(values > otsu)
            for values, otsu in zip(cell_values, cell_table["gd_image_otsu"])
        ]
    )
    fits.append(
        {
            "threshold_method": "otsu",
            "threshold_value": np.nan,
            **_fit_fraction_cutoff(labels, otsu_fractions),
        }
    )

    percentile_grid = np.r_[np.arange(50.0, 99.0, 0.5), np.arange(99.0, 100.0, 0.1)]
    for percentile in percentile_grid:
        thresholds = {
            image_name: float(np.percentile(gd, percentile))
            for image_name, (gd, _) in image_cache.items()
        }
        fractions = np.array(
            [
                np.mean(values > thresholds[image_name])
                for values, image_name in zip(
                    cell_values, cell_table["source_image"]
                )
            ]
        )
        fits.append(
            {
                "threshold_method": "percentile",
                "threshold_value": float(percentile),
                **_fit_fraction_cutoff(labels, fractions),
            }
        )

    fit_table = pd.DataFrame(fits).sort_values(
        ["balanced_accuracy", "specificity", "sensitivity", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    best = fit_table.iloc[0]
    if best["threshold_method"] == "otsu":
        thresholds = {
            image_name: _otsu(gd) for image_name, (gd, _) in image_cache.items()
        }
    else:
        thresholds = {
            image_name: float(np.percentile(gd, best["threshold_value"]))
            for image_name, (gd, _) in image_cache.items()
        }
    cell_table["gd_threshold"] = cell_table["source_image"].map(thresholds)
    cell_table["gd_positive_fraction"] = [
        float(np.mean(values > threshold))
        for values, threshold in zip(cell_values, cell_table["gd_threshold"])
    ]
    cell_table["predicted_label"] = (
        cell_table["gd_positive_fraction"] >= best["positive_fraction_cutoff"]
    ).astype(int)
    return cell_table.drop(columns="_values"), fit_table


def calibrate(
    folder: Path,
    output_dir: Path,
    *,
    gd_channel: int = 2,
    reference_channel: int = 4,
    mask_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = _read_incremental_annotations(folder)
    points = _add_image_features(
        annotations,
        folder,
        gd_channel=gd_channel,
        reference_channel=reference_channel,
    )
    features = [
        column
        for column in points
        if column.startswith("gd_")
        and any(
            suffix in column
            for suffix in ("_mean", "_max", "_mean_over_otsu", "_mean_percentile")
        )
        and not column.endswith("image_otsu")
    ]
    fitted = pd.DataFrame([_fit_one_feature(points, feature) for feature in features])
    fitted = fitted.sort_values(
        ["balanced_accuracy", "specificity", "sensitivity", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_dir / "threshold_calibration_points.csv", index=False)
    fitted.to_csv(output_dir / "threshold_calibration_summary.csv", index=False)
    if mask_dir is not None:
        cells, cell_fits = _calibrate_annotated_cells(
            points,
            folder,
            mask_dir,
            gd_channel=gd_channel,
        )
        cells.to_csv(output_dir / "cell_threshold_calibration.csv", index=False)
        cell_fits.to_csv(output_dir / "cell_threshold_candidates.csv", index=False)
    return points, fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_folder", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("threshold_calibration"),
    )
    parser.add_argument("--gd-channel", type=int, default=2, help="Zero-based channel index")
    parser.add_argument(
        "--reference-channel",
        type=int,
        default=4,
        help="Zero-based channel index",
    )
    parser.add_argument("--mask-dir", type=Path)
    args = parser.parse_args()
    points, fitted = calibrate(
        args.annotation_folder,
        args.output_dir,
        gd_channel=args.gd_channel,
        reference_channel=args.reference_channel,
        mask_dir=args.mask_dir,
    )
    print(
        f"Cleaned {len(points)} unique annotations: "
        f"{int(points['label'].sum())} positive, "
        f"{int((points['label'] == 0).sum())} negative."
    )
    print(fitted.head(10).to_string(index=False))
    if args.mask_dir is not None:
        cell_fits = pd.read_csv(args.output_dir / "cell_threshold_candidates.csv")
        cells = pd.read_csv(args.output_dir / "cell_threshold_calibration.csv")
        print(
            f"\nLinked {len(cells)} annotations to unique cell masks "
            f"({int(cells['label'].sum())} positive, "
            f"{int((cells['label'] == 0).sum())} negative)."
        )
        print(cell_fits.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
