"""Tests for src/features/summary.py and src/evaluation/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics, select_threshold
from src.features.summary import build_features, missingness_columns


def test_features_ignore_missing_readings_and_have_no_nans():
    dates = pd.date_range("2015-12-01", periods=62, freq="D")
    values = np.ones((2, 62), dtype=np.float32)
    values[1, :31] = 99.0  # stored value under the mask must not count
    mask = np.zeros((2, 62), dtype=np.uint8)
    mask[1, :31] = 1

    feats = build_features(values, mask, dates)
    assert not feats.isna().any().any()
    assert feats.loc[1, "mean"] == pytest.approx(1.0)
    assert feats.loc[1, "miss_fraction"] == pytest.approx(0.5)
    assert feats.loc[1, "miss_fraction_pre2016"] == 1.0 and feats.loc[1, "miss_fraction_2016"] == 0.0
    assert feats.loc[1, "miss_first_observed_day"] == 31
    assert set(missingness_columns(feats)) == {c for c in feats.columns if c.startswith("miss_")}
    assert {"month_2015-12", "month_2016-01"} <= set(feats.columns)


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    threshold = select_threshold(y, scores)
    assert threshold == pytest.approx(0.7)

    m = compute_metrics(y, scores, threshold)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["fpr"] == 0.0
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}
    assert compute_metrics(y, scores, 0.15)["fpr"] == pytest.approx(2 / 3)
