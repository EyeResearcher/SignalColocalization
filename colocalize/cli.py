"""Command-line interface for colocalization analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile

from .datasets import AnalysisConfig, ReferenceSet, SignalChannel
from .models import resolve_model_source
from .pipeline import build_dataset, inspect_inputs, run_analysis
from .visualization import save_segmentation_views, show_segmentation


CLI_DEFAULTS = {
    "input_dir": Path("."),
    "output_dir": Path("results"),
    "reference_name": "reference",
    "reference_channel": 0,
    "model": "cpdino_BRN3A",
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 15,
    "normalize": True,
    "signal_name": "signal",
    "signal_channel": 1,
    "threshold_method": "otsu",
    "threshold_value": None,
    "positive_fraction_cutoff": 0.80,
    "recursive": False,
    "exclude": (),
    "time_index": 0,
    "scene_index": 0,
    "z_projection": "max",
    "device": "auto",
    "save_masks": True,
    "save_segmentation": False,
    "segmentation_output_dir": None,
}


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

    if values.get("segmentation_output_dir") is not None:
        candidate = Path(values["segmentation_output_dir"])
        if not candidate.is_absolute():
            values["segmentation_output_dir"] = base_dir / candidate

    if "extensions" in values:
        values["extensions"] = tuple(values["extensions"])
    if "tile_size" in values and values["tile_size"] is not None:
        values["tile_size"] = tuple(values["tile_size"])
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
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Render and optionally save QC grids and their individual panels."""
    if not show and not save:
        return []

    expected = len(build_dataset(config)) * len(config.reference_sets)
    if len(mask_paths) != expected:
        raise RuntimeError(
            "Segmentation visualization requires saved masks for every image and "
            f"reference set; expected {expected}, found {len(mask_paths)}."
        )

    qc_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.output_dir / "segmentation_qc"
    )
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
                signal_image = acquisition.channel(signal_spec.channel)
                title = (
                    f"{acquisition.path.name} | reference={reference.name} | "
                    f"signal={signal_spec.name}"
                )
                name = (
                    f"{_safe_name(acquisition.path.stem)}__"
                    f"{_safe_name(reference.name)}__"
                    f"{_safe_name(signal_spec.name)}_segmentation"
                )
                if save:
                    saved.extend(
                        save_segmentation_views(
                            reference_image,
                            masks,
                            signal_image,
                            output_dir=qc_dir,
                            name=name,
                            signal_spec=signal_spec,
                            title=title,
                        )
                    )
                if show:
                    figure = show_segmentation(
                        reference_image,
                        masks,
                        signal_image,
                        signal_spec=signal_spec,
                        title=title,
                    )
                    plt.show()
                    plt.close(figure)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Optional analysis JSON file. Explicit CLI options override its values.",
    )
    _add_config_arguments(parser)
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
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save the QC grid and its four panels as PNGs.",
    )
    parser.add_argument(
        "--segmentation-output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Directory for saved segmentation grids and panels; implies "
            "--save-segmentation (default: OUTPUT_DIR/segmentation_qc)."
        ),
    )
    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line equivalents for serializable AnalysisConfig fields."""
    paths = parser.add_argument_group("input and output")
    paths.add_argument(
        "--input-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Default: current directory.",
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Default: results.",
    )
    paths.add_argument(
        "--extensions",
        nargs="+",
        default=argparse.SUPPRESS,
        help=(
            "Supported filename extensions, including the leading dot; "
            "defaults to all supported formats."
        ),
    )
    paths.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Search subdirectories (default: false).",
    )
    paths.add_argument(
        "--exclude",
        action="append",
        default=argparse.SUPPRESS,
        metavar="PATH_OR_GLOB",
        help="Path or glob to exclude; repeat for multiple values.",
    )

    reference = parser.add_argument_group("reference segmentation")
    reference.add_argument(
        "--reference-name", default=argparse.SUPPRESS, help="Default: reference."
    )
    reference.add_argument(
        "--reference-channel",
        type=_channel_key,
        default=argparse.SUPPRESS,
        help="Zero-based channel index or metadata channel name (default: 0).",
    )
    reference.add_argument(
        "--model", default=argparse.SUPPRESS, help="Default: cpdino_BRN3A."
    )
    reference.add_argument(
        "--diameter",
        type=float,
        default=argparse.SUPPRESS,
        help="Default: automatic.",
    )
    reference.add_argument(
        "--flow-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Default: 0.4.",
    )
    reference.add_argument(
        "--cellprob-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Default: 0.0.",
    )
    reference.add_argument(
        "--min-size", type=int, default=argparse.SUPPRESS, help="Default: 15."
    )
    reference.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable Cellpose normalization (default: true).",
    )

    signal = parser.add_argument_group("signal measurement")
    signal.add_argument(
        "--signal-name", default=argparse.SUPPRESS, help="Default: signal."
    )
    signal.add_argument(
        "--signal-channel",
        type=_channel_key,
        default=argparse.SUPPRESS,
        help="Zero-based channel index or metadata channel name (default: 1).",
    )
    signal.add_argument(
        "--threshold-method",
        choices=("otsu", "percentile", "absolute", "none"),
        default=argparse.SUPPRESS,
        help="Default: otsu.",
    )
    signal.add_argument(
        "--threshold-value",
        type=float,
        default=argparse.SUPPRESS,
        help="Percentile or absolute cutoff (default: method-dependent).",
    )
    signal.add_argument(
        "--positive-fraction-cutoff",
        type=float,
        default=argparse.SUPPRESS,
        help="Default: 0.80.",
    )

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--time-index", type=int, default=argparse.SUPPRESS, help="Default: 0."
    )
    analysis.add_argument(
        "--scene-index", type=int, default=argparse.SUPPRESS, help="Default: 0."
    )
    analysis.add_argument(
        "--z-projection",
        choices=("max", "mean", "first"),
        default=argparse.SUPPRESS,
        help="Default: max.",
    )
    analysis.add_argument(
        "--device", default=argparse.SUPPRESS, help="Default: auto."
    )
    analysis.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save label TIFF masks (default: true).",
    )
    analysis.add_argument(
        "--tile-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=argparse.SUPPRESS,
        help="Crop images into non-overlapping tiles before segmentation. Default: disabled.",
    )
    analysis.add_argument(
        "--stitch-masks",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Stitch per-tile masks into a single full-image mask file (default: false).",
    )


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Build a config from CLI defaults/options, optionally overlaying JSON."""
    if args.config is None:
        values = CLI_DEFAULTS
        config = AnalysisConfig(
            input_dir=values["input_dir"],
            output_dir=values["output_dir"],
            reference_sets=[
                ReferenceSet(
                    name=values["reference_name"],
                    channel=values["reference_channel"],
                    model=values["model"],
                    diameter=values["diameter"],
                    flow_threshold=values["flow_threshold"],
                    cellprob_threshold=values["cellprob_threshold"],
                    min_size=values["min_size"],
                    normalize=values["normalize"],
                )
            ],
            signal_channels=[
                SignalChannel(
                    name=values["signal_name"],
                    channel=values["signal_channel"],
                    threshold_method=values["threshold_method"],
                    threshold_value=values["threshold_value"],
                    positive_fraction_cutoff=values["positive_fraction_cutoff"],
                )
            ],
            recursive=values["recursive"],
            exclude=values["exclude"],
            time_index=values["time_index"],
            scene_index=values["scene_index"],
            z_projection=values["z_projection"],
            device=values["device"],
            save_masks=values["save_masks"],
            save_segmentation=values["save_segmentation"],
            segmentation_output_dir=values["segmentation_output_dir"],
        )
    else:
        config = load_config(args.config)

    reference_values = {
        field: getattr(args, argument)
        for argument, field in (
            ("reference_name", "name"),
            ("reference_channel", "channel"),
            ("model", "model"),
            ("diameter", "diameter"),
            ("flow_threshold", "flow_threshold"),
            ("cellprob_threshold", "cellprob_threshold"),
            ("min_size", "min_size"),
            ("normalize", "normalize"),
        )
        if hasattr(args, argument)
    }
    if reference_values:
        config.reference_sets[0] = replace(config.reference_sets[0], **reference_values)

    signal_values = {
        field: getattr(args, argument)
        for argument, field in (
            ("signal_name", "name"),
            ("signal_channel", "channel"),
            ("threshold_method", "threshold_method"),
            ("threshold_value", "threshold_value"),
            ("positive_fraction_cutoff", "positive_fraction_cutoff"),
        )
        if hasattr(args, argument)
    }
    if signal_values:
        config.signal_channels[0] = replace(config.signal_channels[0], **signal_values)

    for argument in (
        "input_dir",
        "output_dir",
        "recursive",
        "time_index",
        "scene_index",
        "z_projection",
        "device",
        "save_masks",
        "save_segmentation",
        "segmentation_output_dir",
        "stitch_masks",
    ):
        if hasattr(args, argument):
            setattr(config, argument, getattr(args, argument))
    if hasattr(args, "exclude"):
        config.exclude = tuple(args.exclude)
    if hasattr(args, "extensions"):
        config.extensions = tuple(args.extensions)
    if hasattr(args, "tile_size") and args.tile_size is not None:
        config.tile_size = tuple(args.tile_size)
    if config.segmentation_output_dir is not None and not (
        hasattr(args, "save_segmentation") and args.save_segmentation is False
    ):
        config.save_segmentation = True
    return config


def _channel_key(value: str) -> int | str:
    """Interpret an integer-looking channel as an index, otherwise as a name."""
    try:
        return int(value)
    except ValueError:
        return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

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

    if args.show_segmentation and not config.save_masks:
        config.save_masks = True

    result = run_analysis(config)
    if args.show_segmentation:
        save_or_show_segmentations(
            config,
            result.mask_paths,
            show=True,
        )
    saved = result.segmentation_paths
    print(f"Analyzed {len(result.images)} image/reference set(s).")
    print(f"Measured {len(result.cells)} segmented cell(s).")
    print(f"Results: {config.output_dir.resolve()}")
    if saved:
        print(f"Saved {len(saved)} segmentation QC image(s) to {saved[0].parent.resolve()}")
    return 0


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
