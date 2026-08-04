"""Run colocalization on images selected by flexible local/eLabFTW pointers."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile

from colocalize.cli import load_config, save_or_show_segmentations
from colocalize.elabftw import ElabClient, ElabError, write_workbook
from colocalize.pipeline import inspect_inputs, run_analysis
from colocalize.sources import resolve_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", help="Existing analysis JSON configuration.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="POINTER",
        help="Image or image-group pointer; repeat as needed. See README for accepted forms.",
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        action="append",
        default=[],
        help="Compatibility shorthand for --source experiment:ID; repeat as needed.",
    )
    parser.add_argument(
        "--item-id",
        type=int,
        action="append",
        default=[],
        help="Shorthand for an eLabFTW resource and all its image uploads.",
    )
    parser.add_argument(
        "--upload-id",
        type=int,
        action="append",
        default=[],
        help="Shorthand for an individual eLabFTW upload; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the config output directory.")
    parser.add_argument("--inspect", action="store_true", help="Inspect downloaded images without Cellpose.")
    parser.add_argument("--save-segmentation", action="store_true", help="Save segmentation QC PNGs.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify VPN, credentials, and an extended experiment LIST request, then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.smoke_test:
            client = ElabClient()
            experiments = client.list_experiments(limit=1)
            print(f"eLabFTW connection OK; {len(experiments)} experiment record(s) returned.")
            return 0
        if args.config is None:
            raise ElabError("config is required unless --smoke-test is used.")

        pointers = list(args.source)
        pointers.extend(f"experiment:{record_id}" for record_id in args.experiment_id)
        pointers.extend(f"item:{record_id}" for record_id in args.item_id)
        pointers.extend(f"upload:{upload_id}" for upload_id in args.upload_id)
        with tempfile.TemporaryDirectory(prefix="colocalization_sources_") as temp:
            input_dir = Path(temp) / "uploads"
            resolved = resolve_sources(pointers, input_dir)

            config = load_config(args.config)
            output_dir = args.output_dir or config.output_dir
            config = replace(config, input_dir=input_dir, output_dir=output_dir)
            if args.inspect:
                print(inspect_inputs(config).to_string(index=False))
                print(f"\nResolved and inspected {len(resolved.images)} supported image(s).")
                return 0

            if args.save_segmentation and not config.save_masks:
                config.save_masks = True
            result = run_analysis(config)
            if args.save_segmentation:
                save_or_show_segmentations(config, result.mask_paths, save=True)
            workbook_name = "colocalization_results.xlsx"
            if len(args.experiment_id) == 1 and len(pointers) == 1:
                workbook_name = f"experiment_{args.experiment_id[0]}_colocalization.xlsx"
            workbook = write_workbook(
                output_dir / workbook_name,
                result=result,
                records=resolved.records,
                sources=resolved.images,
                base_url=resolved.client.base_url if resolved.client else "",
            )
            print(f"Analyzed {len(resolved.images)} resolved image(s).")
            print(f"Workbook: {workbook.resolve()}")
            print("No data was written back to eLabFTW.")
            return 0
    except ElabError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
