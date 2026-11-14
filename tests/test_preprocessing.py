"""Tests for src/preprocessing/pipeline.py on tiny hand-written fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import (
    Preprocessor,
    PreprocessorConfig,
    _missing_run_lengths,
    interpolate_short_gaps,
    load_sgcc,
    stratified_split,
)


@pytest.fixture
def sgcc_csv(tmp_path):
    # String-ordered date columns with 2020/1/3 absent, plus one all-missing customer.
    df = pd.DataFrame(
        {
            "CONS_NO": ["A", "B", "C"],
            "FLAG": [0, 1, 0],
            "2020/1/1": [1.0, 2.0, np.nan],
            "2020/1/10": [3.0, 4.0, np.nan],
            "2020/1/2": [5.0, np.nan, np.nan],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_sorts_dates_fills_calendar_and_drops_empty_customers(sgcc_csv):
    data = load_sgcc(sgcc_csv)
    assert list(data.ids) == ["A", "B"]
    assert len(data.dates) == 10 and data.dates[0] == pd.Timestamp("2020-01-01")
    assert data.readings[0, 0] == 1.0 and data.readings[0, 1] == 5.0 and data.readings[0, 9] == 3.0
    assert np.isnan(data.readings[0, 2])  # inserted missing day


def test_split_is_disjoint_complete_and_stratified():
    labels = np.array([0] * 90 + [1] * 10)
    splits = stratified_split(labels, val_frac=0.2, test_frac=0.2, seed=0)
    all_idx = np.concatenate(list(splits.values()))
    assert len(all_idx) == len(np.unique(all_idx)) == 100
    assert labels[splits["test"]].sum() == 2 and labels[splits["val"]].sum() == 2
    assert np.array_equal(splits["test"], stratified_split(labels, 0.2, 0.2, seed=0)["test"])


def test_only_short_interior_gaps_are_interpolated():
    nan = np.nan
    x = np.array([[nan, 1.0, nan, 3.0, nan, nan, nan, 7.0, nan]])
    out = interpolate_short_gaps(x, max_gap=2)
    assert out[0, 2] == 2.0                      # gap of 1: filled
    assert np.isnan(out[0, 4:7]).all()           # gap of 3 > max_gap: kept
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 8])  # leading/trailing: kept
    assert list(_missing_run_lengths(x)[0]) == [1, 0, 1, 0, 3, 3, 3, 0, 1]


def test_preprocessor_fits_on_given_data_only_and_round_trips(tmp_path):
    train = np.array([[0.0, 1.0, 2.0, 1000.0]])
    pre = Preprocessor(PreprocessorConfig(cap_percentile=75, max_interp_gap=1)).fit(train)
    assert pre.cap_value == pytest.approx(np.percentile(train, 75))

    values, mask = pre.transform(np.array([[np.nan, 5000.0, 1.0, np.nan]]))
    assert values.dtype == np.float32 and mask.tolist() == [[1, 0, 0, 1]]
    assert np.isfinite(values).all()

    pre.save(tmp_path / "p.json")
    loaded = Preprocessor.load(tmp_path / "p.json")
    assert np.array_equal(loaded.transform(train)[0], pre.transform(train)[0])
