"""Export descriptive cell-enrichment percentages from an existing cell table.

Run from the repository: python -m scripts.export_ecdf_metrics --help
"""

import argparse
import json
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    # Headless CLI exports must not change a notebook's backend on import.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from colocalize import plot_signal_ecdf, summarize_signal_ecdf


def _reverse_assignment_map(assignments, label):
    """Return field-to-group assignments while rejecting ambiguous fields."""
    reverse = {}
    for group, fields in assignments.items():
        for field in fields:
            field = str(field).zfill(3)
            if field in reverse:
                raise ValueError(
                    f"Field {field!r} appears more than once in {label}: "
                    f"{reverse[field]!r} and {group!r}."
                )
            reverse[field] = str(group)
    return reverse


def group_tiled_sources(cells, eyes_json, animal_ids_json):
    """Group field-level rows into biological animal-eye samples.

    The JSON files map three-digit field IDs to eye and animal. Original source
    names are retained in ``field_source`` for traceability, while ``source`` is
    replaced by ``<animal_id>_<eye>`` so the standard exporter pools all fields
    from the same biological eye before calculating its ECDF percentages.
    """
    with Path(eyes_json).open(encoding="utf-8") as stream:
        eyes = _reverse_assignment_map(json.load(stream), "eye assignments")
    with Path(animal_ids_json).open(encoding="utf-8") as stream:
        animals = _reverse_assignment_map(json.load(stream), "animal assignments")

    result = cells.copy()
    fields = result["source"].astype("string").str.extract(
        r"(?:^|_)G(\d+)(?=_|\.|$)", expand=False
    ).str.zfill(3)
    missing_field = fields.isna()
    if missing_field.any():
        examples = result.loc[missing_field, "source"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Could not extract a G### field ID from sources: {examples}")

    eye = fields.map(eyes)
    animal = fields.map(animals)
    missing = eye.isna() | animal.isna()
    if missing.any():
        examples = sorted(fields[missing].drop_duplicates().tolist())[:10]
        raise ValueError(f"Fields are missing eye or animal assignments: {examples}")

    result["field_source"] = result["source"]
    result["field_id"] = fields
    result["eye"] = eye
    result["animal_id"] = animal
    result["source"] = animal + "_" + eye
    return result


def _sort_by_eye(table):
    """Order L sources before R sources, with unrecognized eyes last."""
    eye = table["source"].astype("string").str.extract(r"(?:^|_)(L|R)(?=_|\.|$)", expand=False)
    return (table.assign(_eye_order=eye.map({"L": 0, "R": 1}).fillna(2))
            .sort_values(["_eye_order", "source", "reference_set"], kind="stable")
            .drop(columns="_eye_order").reset_index(drop=True))


def pool_metrics_by_eye(metrics):
    """Pool exact counts by eye; unavailable measurements never count as negative."""
    eye = metrics["source"].astype("string").str.extract(r"(?:^|_)(L|R)(?=_|\.|$)", expand=False)
    keys = ["eye", "reference_set", "signal", "proxy", "metric", "normalize_by_noise", "threshold"]
    pooled = (metrics.assign(eye=eye.fillna("Unknown"))
              .groupby(keys, sort=True, dropna=False, observed=True)
              .agg(image_count=("source", "nunique"),
                   cell_count=("cell_count", "sum"),
                   valid_cell_count=("valid_cell_count", "sum"),
                   excluded_cell_count=("excluded_cell_count", "sum"),
                   cells_above=("cells_above", "sum")).reset_index())
    denominator = pooled.valid_cell_count.where(pooled.valid_cell_count > 0)
    pooled["percent_above"] = 100 * pooled.cells_above / denominator
    pooled["ecdf_at_threshold"] = (pooled.valid_cell_count - pooled.cells_above) / denominator
    return pooled


def save_eye_metrics(metrics, output_dir):
    """Write detailed and compact cell-weighted L/R summaries."""
    output_dir = Path(output_dir)
    pooled = pool_metrics_by_eye(metrics)
    pooled.to_csv(output_dir / "ecdf_proxy_metrics_by_eye.csv", index=False, na_rep="NA")
    compact = pooled.assign(cutoff_label=pooled.proxy + " > " + pooled.threshold.map(lambda value: f"{value:g}"))
    compact = compact.pivot(index=["eye", "reference_set", "signal"], columns="cutoff_label", values="percent_above")
    compact.to_csv(output_dir / "ecdf_proxy_percentages_by_eye.csv", na_rep="NA")
    return pooled


def export_metrics(cells, signal, output_dir):
    """Save exact per-image counts, percentages, and three cutoff-sweep plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        ("median_above_background", "corrected_median", (0,), False),
        ("median_noise_sweep", "corrected_median", (0, 1, 2, 3), True),
    ]
    if f"signal_{signal}_frac_above_z3" in cells:
        settings.append(("pixel_extent_sweep", "frac_above_z3", (.1, .5, .8), False))
    summaries, paths = [], []
    for name, metric, thresholds, normalize in settings:
        options = dict(metric=metric, thresholds=thresholds, normalize_by_noise=normalize)
        summary = summarize_signal_ecdf(cells, signal, **options)
        summary.insert(0, "proxy", name)
        summaries.append(summary)
        for index, (reference, block) in enumerate(cells.groupby("reference_set", sort=False, dropna=False)):
            fig = plot_signal_ecdf(block, signal, **options)
            fig.set_size_inches(12, 7)
            fig.axes[0].set_title(f"{signal} within {reference} masks: {name.replace('_', ' ')}")
            fig.axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
            fig.text(.02, .02, "Above cutoff (%) = 100 × (1 − CDF). Strict > cutoffs; descriptive enrichment proxies.", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            path = output_dir / f"{name}_reference_{index + 1}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    result = _sort_by_eye(pd.concat(summaries, ignore_index=True))
    result.to_csv(output_dir / "ecdf_proxy_metrics.csv", index=False)
    # A readable companion table keeps all images, including unavailable values.
    compact = result.assign(cutoff_label=result.proxy + " > " + result.threshold.map(lambda value: f"{value:g}"))
    compact = compact.pivot(index=["source", "reference_set"], columns="cutoff_label", values="percent_above")
    compact = _sort_by_eye(compact.reset_index()).set_index(["source", "reference_set"])
    compact.to_csv(output_dir / "ecdf_proxy_percentages.csv", na_rep="NA")
    save_eye_metrics(result, output_dir)
    return result, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=Path("cells.csv"))
    parser.add_argument("--signal", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("presentation_figures/ecdf_metrics"))
    parser.add_argument("--eyes-json", type=Path)
    parser.add_argument("--animal-ids-json", type=Path)
    args = parser.parse_args()
    if (args.eyes_json is None) != (args.animal_ids_json is None):
        parser.error("--eyes-json and --animal-ids-json must be supplied together")
    cells = pd.read_csv(args.cells)
    if args.eyes_json is not None:
        cells = group_tiled_sources(cells, args.eyes_json, args.animal_ids_json)
    metrics, paths = export_metrics(cells, args.signal, args.output_dir)
    print(f"Saved {len(metrics)} summary rows and {len(paths)} plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
