"""Average cells within images, images within individual eyes, then eyes by side.

Example: python -m scripts.export_eye_intensity_means --gd-cells GD/cells.csv
         --hd-cells cells.csv --output-dir presentation_figures/eye_intensity_means
Filename IDs identify samples; L=no injury and R=ONC for this experiment.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


METRICS = ("corrected_mean", "corrected_median", "tail_mean", "frac_above_z3")


def summarize_images(cells, signal, reference="RPBMS"):
    cells = cells.loc[cells.reference_set.eq(reference)].copy()
    if cells.empty:
        raise ValueError(f"No cells for reference {reference}")
    if cells.duplicated(["source", "cell_id"]).any():
        raise ValueError("Duplicate cell IDs within an image")
    rows = []
    for source, block in cells.groupby("source", sort=True):
        match = re.match(r"^(\d+)_\1_([LR])_", str(source))
        if not match:
            raise ValueError(f"Cannot identify sample and eye from {source}")
        sample, side = match.groups()
        for metric in METRICS:
            values = block[f"signal_{signal}_{metric}"].to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            rows.append(dict(signal=signal, reference_set=reference, source=source,
                             sample_id=sample, eye=side,
                             treatment="No injury" if side == "L" else "ONC",
                             metric=metric, cell_count=len(values),
                             valid_cell_count=len(valid),
                             excluded_cell_count=len(values) - len(valid),
                             image_mean=float(valid.mean()) if len(valid) else np.nan))
    return pd.DataFrame(rows)


def aggregate_eyes(images):
    """Each finite image mean contributes equally, regardless of cell count."""
    keys = ["signal", "reference_set", "sample_id", "eye", "treatment", "metric"]
    eyes = images.groupby(keys, sort=True, dropna=False).agg(
        image_count=("source", "size"), valid_image_count=("image_mean", "count"),
        cell_count=("cell_count", "sum"), valid_cell_count=("valid_cell_count", "sum"),
        excluded_cell_count=("excluded_cell_count", "sum"),
        eye_mean=("image_mean", "mean"), image_sd=("image_mean", "std"),
    ).reset_index()
    groups = eyes.groupby(["signal", "reference_set", "eye", "treatment", "metric"], sort=True).agg(
        eye_count=("sample_id", "size"), valid_eye_count=("eye_mean", "count"),
        mean=("eye_mean", "mean"), sd_between_eyes=("eye_mean", "std"),
    ).reset_index()
    return eyes, groups


def plot_summary(output_dir):
    """Plot saved eye means; bars show treatment means and sample SD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    eyes = pd.read_csv(output_dir / "per_eye.csv")
    groups = pd.read_csv(output_dir / "by_treatment.csv")
    titles = ["Mean cell intensity", "Mean of cell medians", "Mean of cell tail means"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharey="row")
    colors = {"L": "#247ba0", "R": "#be4778"}
    offsets = {sample: offset for sample, offset in
               zip(sorted(eyes.sample_id.unique()), np.linspace(-.12, .12, eyes.sample_id.nunique()))}
    for row, signal in enumerate(("GD", "HD")):
        for col, metric in enumerate(("corrected_mean", "corrected_median", "tail_mean")):
            ax = axes[col, row]
            block = eyes[eyes.signal.eq(signal) & eyes.metric.eq(metric)]
            for _, point in block.iterrows():
                x = (0 if point.eye == "L" else 1) + offsets[point.sample_id]
                ax.scatter(x, point.eye_mean, color=colors[point.eye], s=45, zorder=3)
                label_offset = (4, 5)
                if signal == "HD" and metric == "tail_mean" and point.eye == "L":
                    label_offset = {5217: (-25, 12), 5220: (-12, -17), 5222: (3, 15)}[point.sample_id]
                ax.annotate(str(point.sample_id), (x, point.eye_mean), xytext=label_offset,
                            textcoords="offset points", fontsize=8, color="#444444")
            summary = groups[groups.signal.eq(signal) & groups.metric.eq(metric)]
            for _, point in summary.iterrows():
                x = (0 if point.eye == "L" else 1) + .28
                ax.errorbar(x, point["mean"], yerr=point.sd_between_eyes,
                            color=colors[point.eye], fmt="_", ms=17, capsize=5, lw=1.5)
            ax.set(title=f"{signal}: {titles[col]}", ylabel="Background-corrected intensity (a.u.)",
                   xticks=[0, 1], xticklabels=["No injury (L)\nn = 3 eyes", "ONC (R)\nn = 4 eyes"],
                   xlim=(-.4, 1.65))
            ax.axhline(0, color="#bbbbbb", lw=.8, ls="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelleft=True)
            ax.margins(y=.18)
    fig.suptitle("Image means summarized by eye and treatment", fontsize=19)
    fig.text(.04, .045, "Dots: individual eyes, labeled by sample ID. Side bars: treatment mean ± SD across eyes.", fontsize=10)
    fig.text(.04, .02, "Cells averaged within images; images equally weighted within eyes. These data contain one image per eye per signal.", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .95), h_pad=2)
    fig.savefig(output_dir / "eye_intensity_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gd-cells", type=Path, required=True)
    parser.add_argument("--hd-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {"GD": args.gd_cells, "HD": args.hd_cells}
    images = pd.concat([summarize_images(pd.read_csv(path), signal)
                        for signal, path in paths.items()], ignore_index=True)
    eyes, groups = aggregate_eyes(images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("per_image", images), ("per_eye", eyes), ("by_treatment", groups)):
        table.to_csv(args.output_dir / f"{name}.csv", index=False, na_rep="NA")
    record = {
        "sources": {s: str(p.resolve()) for s, p in paths.items()},
        "method": "Finite cell values averaged within image; image means equally weighted within sample ID and side; eye means equally weighted within treatment. No clipping of negative values.",
        "metrics": {
            "corrected_mean": "Mean of per-cell background-corrected pixel means, in raw intensity units. Cells receive equal weight, not pixels.",
            "corrected_median": "Mean of per-cell corrected medians, matching the intensity ECDF x values.",
            "tail_mean": "Mean of per-cell upper-tail means, matching the tail-mean ECDF x values. These source runs use the brightest 10% in raw corrected units.",
            "frac_above_z3": "Mean of per-cell fractions of pixels above background + 3 sigma, on a 0-1 scale; not intensity or fraction of positive cells.",
        },
        "notes": ["Filename IDs are treated as sample IDs; L=no injury, R=ONC per user context.",
                  "One image per eye per signal in these inputs. No 5214 L image is present.",
                  "SD is sample SD between finite eye means (ddof=1), not cell SD or SEM.",
                  "HD 5222 L pixel extent is unavailable (background sigma is zero); it is retained as NA. Its intensity means remain available.",
                  "HD uses the repository cells.csv snapshot that matches the supplied plots; the external HD cells.csv has different corrected values."]
    }
    (args.output_dir / "analysis_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(groups.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
