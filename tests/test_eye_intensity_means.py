"""Check the analysis hierarchy with unequal cell and image counts."""

import numpy as np
import pandas as pd
import unittest

from scripts.export_eye_intensity_means import METRICS, aggregate_eyes, summarize_images


def test_equal_image_then_equal_eye_weights_and_missing_values():
    cells = pd.DataFrame({
        "source": ["1_1_L_a.tif"] * 2 + ["1_1_L_b.tif"] * 8 + ["2_2_L_a.tif", "3_3_R_a.tif"],
        "reference_set": ["RPBMS"] * 12,
        "cell_id": [1, 2] + list(range(1, 9)) + [1, 1],
    })
    for metric in METRICS:
        cells[f"signal_HD_{metric}"] = [-2, 2] + [10] * 8 + [20, np.nan]
    images = summarize_images(cells, "HD")
    eyes, groups = aggregate_eyes(images)
    means = eyes[eyes.metric.eq("corrected_mean")].set_index("sample_id")
    assert means.loc["1", "eye_mean"] == 5  # (0 + 10) / 2, not 8
    left = groups[groups.eye.eq("L") & groups.metric.eq("corrected_mean")].iloc[0]
    assert left["mean"] == 12.5  # (5 + 20) / 2, not average of three images
    right = groups[groups.eye.eq("R") & groups.metric.eq("corrected_mean")].iloc[0]
    assert right.valid_eye_count == 0
    assert np.isnan(right["mean"])
    assert means.loc["3", "excluded_cell_count"] == 1


def test_unrecognized_metadata_is_rejected():
    cells = pd.DataFrame({"source": ["unknown.tif"], "reference_set": ["RPBMS"], "cell_id": [1]})
    with unittest.TestCase().assertRaisesRegex(ValueError, "Cannot identify"):
        summarize_images(cells, "HD")


if __name__ == "__main__":
    test_equal_image_then_equal_eye_weights_and_missing_values()
    test_unrecognized_metadata_is_rejected()
    print("2 checks passed")
