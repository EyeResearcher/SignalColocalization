"""Build aligned marker+DAPI RGB crop datasets for Cellpose annotation."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import tifffile


MARKERS = ("MG", "GD", "HD", "RGC")


def signal_plane(image: np.ndarray) -> np.ndarray:
    """Collapse a grayscale or pseudocolored RGB image to one signal plane."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].max(axis=-1)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def safe_specimen_name(specimen: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", specimen).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--crop-size", type=int, default=1024)
    args = parser.parse_args()

    root = args.root.resolve()
    source_manifest = root / f"Crops_{args.crop_size}_manifest.csv"
    output_root = root / f"Paired_Crops_{args.crop_size}"
    output_root.mkdir(parents=True, exist_ok=True)

    with source_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["class"], row["specimen"], int(row["crop_index"]))
        if key in indexed:
            raise RuntimeError(f"Duplicate manifest key: {key}")
        indexed[key] = row

    dapi_keys = sorted(
        (specimen, index)
        for class_name, specimen, index in indexed
        if class_name == "DAPI"
    )
    if not dapi_keys:
        raise RuntimeError("No DAPI crops found in the source manifest")

    output_rows: list[dict[str, object]] = []
    written = 0
    skipped = 0

    for marker in MARKERS:
        output_dir = output_root / f"{marker}_DAPI"
        output_dir.mkdir(parents=True, exist_ok=True)

        for specimen, crop_index in dapi_keys:
            dapi_row = indexed[("DAPI", specimen, crop_index)]
            marker_key = (marker, specimen, crop_index)
            if marker_key not in indexed:
                raise RuntimeError(f"Missing aligned crop: {marker_key}")
            marker_row = indexed[marker_key]

            if (dapi_row["x"], dapi_row["y"]) != (
                marker_row["x"],
                marker_row["y"],
            ):
                raise RuntimeError(f"Coordinate mismatch: {marker_key}")

            dapi_path = Path(dapi_row["crop"])
            marker_path = Path(marker_row["crop"])
            dapi = signal_plane(tifffile.imread(dapi_path))
            marker_signal = signal_plane(tifffile.imread(marker_path))
            if dapi.shape != marker_signal.shape:
                raise RuntimeError(
                    f"Shape mismatch for {marker_key}: {dapi.shape} vs "
                    f"{marker_signal.shape}"
                )

            # RGB display convention: marker=red, empty=green, DAPI=blue.
            paired = np.stack(
                (marker_signal, np.zeros_like(marker_signal), dapi), axis=-1
            )
            name = (
                f"{safe_specimen_name(specimen)}__{marker}_DAPI"
                f"__crop_{crop_index:02d}__y{int(dapi_row['y']):05d}"
                f"_x{int(dapi_row['x']):05d}_img.tif"
            )
            output_path = output_dir / name

            if output_path.exists():
                existing = tifffile.imread(output_path)
                if existing.shape != paired.shape or not np.array_equal(existing, paired):
                    raise RuntimeError(
                        f"Existing paired crop differs; refusing to overwrite: {output_path}"
                    )
                skipped += 1
            else:
                tifffile.imwrite(
                    output_path,
                    paired,
                    compression="deflate",
                    photometric="rgb",
                    metadata={
                        "axes": "YXS",
                        "channel_order": "R=marker,G=empty,B=DAPI",
                        "marker": marker,
                    },
                )
                written += 1

            output_rows.append(
                {
                    "dataset": f"{marker}_DAPI",
                    "specimen": specimen,
                    "crop_index": crop_index,
                    "x": dapi_row["x"],
                    "y": dapi_row["y"],
                    "marker_source": str(marker_path),
                    "dapi_source": str(dapi_path),
                    "paired_image": str(output_path),
                    "red_channel": marker,
                    "green_channel": "empty",
                    "blue_channel": "DAPI",
                }
            )

    output_manifest = output_root / "paired_dataset_manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    readme = output_root / "README.txt"
    readme.write_text(
        "Cellpose paired marker + DAPI crop datasets\n"
        "\n"
        "Image layout: RGB / Y-X-channel\n"
        "Red channel: specialized marker\n"
        "Green channel: empty (zeros)\n"
        "Blue channel: DAPI\n"
        "\n"
        "All images are 1024 x 1024 uint8 TIFFs. Files ending in _img.tif "
        "can be selected with Cellpose --img_filter _img.\n",
        encoding="utf-8",
    )

    print(f"datasets={len(MARKERS)}")
    print(f"images_per_dataset={len(dapi_keys)}")
    print(f"written={written}")
    print(f"skipped_existing={skipped}")
    print(f"output={output_root}")
    print(f"manifest={output_manifest}")


if __name__ == "__main__":
    main()
