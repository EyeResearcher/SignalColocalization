"""Two-sided paired tests of ONC minus no-injury eye means for plotted metrics.

Run: python -m scripts.export_paired_significance --input-dir presentation_figures/eye_intensity_means
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ("corrected_mean", "corrected_median", "tail_mean")


def holm_adjust(pvalues):
    """Holm correction across a complete, finite family of tests."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def paired_tests(eyes):
    summaries, differences, exclusions = [], [], []
    for signal in ("GD", "HD"):
        for metric in METRICS:
            block = eyes.loc[eyes.signal.eq(signal) & eyes.metric.eq(metric)
                             & eyes.reference_set.eq("RPBMS")]
            if block.empty or block.duplicated(["sample_id", "eye"]).any():
                raise ValueError(f"Missing data or duplicate eye summaries: {signal}, {metric}")
            wide = block.pivot(index="sample_id", columns="eye", values="eye_mean").reindex(columns=["L", "R"])
            valid = np.isfinite(wide).all(axis=1)
            for sample, values in wide.loc[~valid].iterrows():
                exclusions.append(dict(signal=signal, metric=metric, sample_id=str(sample),
                                       L=values.L, R=values.R, reason="Missing or nonfinite eye mean"))
            pairs = wide.loc[valid].sort_index()
            if len(pairs) < 2:
                raise ValueError(f"Fewer than two complete pairs: {signal}, {metric}")
            d = (pairs.R - pairs.L).to_numpy()
            test = stats.ttest_rel(pairs.R, pairs.L, alternative="two-sided")
            ci = test.confidence_interval(confidence_level=.95)
            permutation = stats.permutation_test(
                (d,), np.mean, permutation_type="samples", vectorized=False,
                n_resamples=np.inf, alternative="two-sided")
            # Independent check of the exact symmetric sign-flip distribution.
            null = np.asarray([np.mean(d * signs) for signs in product((-1, 1), repeat=len(d))])
            manual_p = np.mean(np.abs(null) >= abs(d.mean()) - 1e-12)
            np.testing.assert_allclose(permutation.pvalue, manual_p)
            manual_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            np.testing.assert_allclose(test.statistic, manual_t)
            for sample, values in pairs.iterrows():
                differences.append(dict(signal=signal, metric=metric, sample_id=str(sample),
                                        no_injury_L=values.L, ONC_R=values.R, R_minus_L=values.R-values.L))
            summaries.append(dict(
                signal=signal, metric=metric, n_pairs=len(d),
                paired_sample_ids=";".join(pairs.index.astype(str)),
                mean_no_injury_L=pairs.L.mean(), mean_ONC_R=pairs.R.mean(),
                mean_R_minus_L=d.mean(), sd_difference=d.std(ddof=1),
                ci95_low=ci.low, ci95_high=ci.high, t_statistic=test.statistic,
                df=test.df, paired_t_p=test.pvalue, exact_permutation_p=permutation.pvalue,
            ))
    results = pd.DataFrame(summaries)
    results["paired_t_p_holm"] = holm_adjust(results.paired_t_p)
    results["exact_permutation_p_holm"] = holm_adjust(results.exact_permutation_p)
    return results, pd.DataFrame(differences), pd.DataFrame(exclusions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir / "per_eye.csv"
    eyes = pd.read_csv(source, dtype={"sample_id": str})
    results, differences, exclusions = paired_tests(eyes)
    out = args.input_dir / "paired_significance"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("results", results), ("animal_differences", differences), ("excluded_pairs", exclusions)):
        table.to_csv(out / f"{name}.csv", index=False, na_rep="NA")
    record = {
        "input": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "comparison": "ONC right eye minus no-injury left eye within the same animal",
        "tests": "Two-sided paired t-test and exact sign-flip permutation test of mean paired difference",
        "multiplicity": "Holm correction across six GD/HD x intensity-metric comparisons, separately for each test method",
        "confidence_intervals": "Pointwise 95% paired t intervals; not adjusted for multiple comparisons",
        "limitations": [
            "Three independent animal pairs; 5214 lacks a left-eye measurement and is excluded from paired tests.",
            "The paired t-test and its intervals assume normally distributed animal-level differences, difficult to assess with three pairs.",
            "Sign-flip inference assumes exchangeable within-animal treatment labels under the null or symmetric independent paired differences.",
            "All ONC eyes are right eyes; treatment and laterality are confounded. Tests quantify the observed R-minus-L contrast.",
            "Exact two-sided sign-flip p-values cannot fall below 0.25 with three pairs.",
            "Cell counts are not the biological sample size. No cell-level resampling was used.",
            "These are exploratory tests of the three displayed metrics, not a prospectively selected primary endpoint.",
        ],
    }
    (out / "methods.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
