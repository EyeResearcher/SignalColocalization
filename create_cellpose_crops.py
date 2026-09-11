"""Create reproducible, aligned random TIFF crops for Cellpose annotation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import tifffile


CLASSES = ("DAPI", "MG", "GD", "HD", "RGC")
CHANNEL_RE = re.compile(r"_C00[1-5]T\d+$", re.IGNORECASE)


def specimen_key(path: Path) -> str:
    return CHANNEL_RE.sub("", path.stem)


def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def crop_has_tissue(crop: np.ndarray) -> bool:
    if crop.ndim == 3:
        gray = crop.astype(np.float32).max(axis=-1)
    else:
        gray = crop.astype(np.float32)
    sampled = gray[::8, ::8]
    nonzero_fraction = float(np.count_nonzero(sampled)) / sampled.size
    contrast = float(np.percentile(sampled, 99) - np.percentile(sampled, 5))
    return nonzero_fraction >= 0.10 and contrast >= 3.0


def choose_coordinates(
    image: np.ndarray,
    crop_size: int,
    count: int,
    seed: int,
    key: str,
) -> list[tuple[int, int]]:
    height, width = image.shape[:2]
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"{key} is {height}x{width}, smaller than {crop_size}x{crop_size}"
        )

    rng = stable_rng(seed, key)
    accepted: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(1000, count * 200)

    for _ in range(max_attempts):
        y = int(rng.integers(0, height - crop_size + 1))
        x = int(rng.integers(0, width - crop_size + 1))
        coordinate = (y, x)
        if coordinate in seen:
            continue
        seen.add(coordinate)
        crop = image[y : y + crop_size, x : x + crop_size]
        if crop_has_tissue(crop):
            accepted.append(coordinate)
            if len(accepted) == count:
                return accepted

    raise RuntimeError(
        f"Only found {len(accepted)} tissue-containing crops for {key} "
        f"after {max_attempts} attempts"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--crops-per-image", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    root = args.root.resolve()
    files_by_class: dict[str, dict[str, Path]] = {}
    for class_name in CLASSES:
        class_dir = root / class_name
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )
        files_by_class[class_name] = {specimen_key(path): path for path in files}

    keys = sorted(files_by_class["DAPI"])
    if not keys:
        raise RuntimeError("No DAPI TIFF files found")
    for class_name in CLASSES:
        class_keys = set(files_by_class[class_name])
        if class_keys != set(keys):
            missing = sorted(set(keys) - class_keys)
            extra = sorted(class_keys - set(keys))
            raise RuntimeError(
                f"{class_name} specimen mismatch; missing={missing}, extra={extra}"
            )

    manifest_rows: list[dict[str, object]] = []
    coordinates_by_key: dict[str, list[tuple[int, int]]] = {}

    for key in keys:
        dapi = tifffile.imread(files_by_class["DAPI"][key])
        coordinates_by_key[key] = choose_coordinates(
            dapi, args.crop_size, args.crops_per_image, args.seed, key
        )
        del dapi

    written = 0
    skipped = 0
    for class_name in CLASSES:
        output_dir = root / class_name / f"Crops_{args.crop_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            source = files_by_class[class_name][key]
            image = tifffile.imread(source)
            for index, (y, x) in enumerate(coordinates_by_key[key], start=1):
                crop = image[y : y + args.crop_size, x : x + args.crop_size]
                output_name = (
                    f"{source.stem}__crop_{index:02d}__y{y:05d}_x{x:05d}.tif"
                )
                output_path = output_dir / output_name
                if output_path.exists():
                    with tifffile.TiffFile(output_path) as tif:
                        existing_shape = tif.pages[0].shape
                    if tuple(existing_shape) != tuple(crop.shape):
                        raise RuntimeError(
                            f"Existing crop has unexpected shape: {output_path}"
                        )
                    skipped += 1
                else:
                    tifffile.imwrite(
                        output_path,
                        crop,
                        compression="deflate",
                        photometric="rgb" if crop.ndim == 3 and crop.shape[-1] in (3, 4) else None,
                    )
                    written += 1
                manifest_rows.append(
                    {
                        "class": class_name,
                        "specimen": key,
                        "source": str(source),
                        "crop": str(output_path),
                        "crop_index": index,
                        "x": x,
                        "y": y,
                        "width": args.crop_size,
                        "height": args.crop_size,
                    }
                )
            del image

    manifest = root / f"Crops_{args.crop_size}_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"specimens={len(keys)}")
    print(f"classes={len(CLASSES)}")
    print(f"written={written}")
    print(f"skipped_existing={skipped}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
