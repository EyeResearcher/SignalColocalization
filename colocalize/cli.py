"""Command-line interface for colocalization analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile

from .datasets import AnalysisConfig, ReferenceSet, SignalChannel
from .models import resolve_model_source
from .pipeline import build_dataset, inspect_inputs, run_analysis
from .visualization import show_segmentation


def load_config(path: str | Path) -> AnalysisConfig:
    """Load an :class:`AnalysisConfig` from a JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    return config_from_mapping(values, base_dir=path.parent)


def config_from_mapping(values: dict, *, base_dir: str | Path = ".") -> AnalysisConfig:
    """Build an analysis configuration from a JSON-compatible mapping."""
    values = dict(values)
    base_dir = Path(base_dir)

    try:
        references = [ReferenceSet(**item) for item in values.pop("reference_sets")]
        signals = [SignalChannel(**item) for item in values.pop("signal_channels")]
    except KeyError as exc:
        raise ValueError(f"Configuration is missing required field {exc.args[0]!r}.") from exc

    for key in ("input_dir", "output_dir"):
        if key not in values:
            raise ValueError(f"Configuration is missing required field {key!r}.")
        candidate = Path(values[key])
        if not candidate.is_absolute():
            values[key] = base_dir / candidate

    if "extensions" in values:
        values["extensions"] = tuple(values["extensions"])
    return AnalysisConfig(
        reference_sets=references,
        signal_channels=signals,
        **values,
    )


def save_or_show_segmentations(
    config: AnalysisConfig,
    mask_paths: list[Path],
    *,
    show: bool = False,
    save: bool = False,
) -> list[Path]:
    """Render QC figures for every image/reference/signal combination."""
    if not show and not save:
        return []
    if save and not show:
        plt.switch_backend("Agg")

    expected = len(build_dataset(config)) * len(config.reference_sets)
    if len(mask_paths) != expected:
        raise RuntimeError(
            "Segmentation visualization requires saved masks for every image and "
            f"reference set; expected {expected}, found {len(mask_paths)}."
        )

    qc_dir = config.output_dir / "segmentation_qc"
    if save:
        qc_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    mask_index = 0
    for acquisition in build_dataset(config):
        for reference in config.reference_sets:
            masks = tifffile.imread(mask_paths[mask_index])
            mask_index += 1
            reference_image = acquisition.channel(reference.channel)
            for signal_spec in config.signal_channels:
                figure = show_segmentation(
                    reference_image,
                    masks,
                    acquisition.channel(signal_spec.channel),
                    signal_spec=signal_spec,
                    title=(
                        f"{acquisition.path.name} | reference={reference.name} | "
                        f"signal={signal_spec.name}"
                    ),
                )
                if save:
                    destination = qc_dir / (
                        f"{_safe_name(acquisition.path.stem)}__"
                        f"{_safe_name(reference.name)}__"
                        f"{_safe_name(signal_spec.name)}_segmentation.png"
                    )
                    figure.savefig(destination, dpi=150, bbox_inches="tight")
                    saved.append(destination)
                if show:
                    plt.show()
                plt.close(figure)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to an analysis JSON file.")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print discovered image shapes/channels and exit without segmentation.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download/cache configured Hugging Face models and exit.",
    )
    parser.add_argument(
        "--show-segmentation",
        action="store_true",
        help="Open each segmentation QC figure interactively after analysis.",
    )
    parser.add_argument(
        "--save-segmentation",
        action="store_true",
        help="Save segmentation QC PNGs under OUTPUT_DIR/segmentation_qc.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.download_models:
        for reference in config.reference_sets:
            resolved = resolve_model_source(reference.model)
            print(f"{reference.name}: {resolved}")
        return 0

    if args.inspect:
        table = inspect_inputs(config)
        print(table.to_string(index=False))
        print(f"\nDiscovered {len(table)} supported image(s).")
        return 0

    if (args.show_segmentation or args.save_segmentation) and not config.save_masks:
        config.save_masks = True

    result = run_analysis(config)
    saved = save_or_show_segmentations(
        config,
        result.mask_paths,
        show=args.show_segmentation,
        save=args.save_segmentation,
    )
    print(f"Analyzed {len(result.images)} image/reference set(s).")
    print(f"Measured {len(result.cells)} segmented cell(s).")
    print(f"Results: {config.output_dir.resolve()}")
    if saved:
        print(f"Saved {len(saved)} segmentation QC figure(s) to {saved[0].parent.resolve()}")
    return 0


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
