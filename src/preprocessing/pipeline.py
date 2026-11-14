"""Preprocessing for the SGCC wide-format dataset.

Decisions (justified by reports/audit/dataset_audit.md):

* Labels are per customer, so each customer is ONE sample: the full daily
  series (2014-01-01 .. 2016-10-31, 1035 days). No windowing.
* Date columns are stored in string order (2014/1/1, 2014/1/10, ...), so they
  are parsed and sorted, and the missing calendar day (2016-09-18) is
  inserted as a missing reading.
* Customers whose readings are all missing are dropped before the split
  (this uses no label or cross-customer information).
* The customer-level, label-stratified split happens BEFORE anything is
  fitted. The only fitted statistics (outlier cap, scaling mean/std) come
  from the training split.

Per-customer transform (the same for training and inference):

    raw -> cap at train percentile -> interpolate short gaps (own series only)
        -> log1p -> standardise with train mean/std -> remaining gaps = 0

A missingness mask (1 = reading was originally missing) is returned
alongside the values, because the audit shows missingness differs by class.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "CONS_NO"
LABEL_COL = "FLAG"
DATE_FORMAT = "%Y/%m/%d"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class RawDataset:
    ids: np.ndarray        # (n,) customer IDs
    labels: np.ndarray     # (n,) int8, 1 = theft-labelled
    readings: np.ndarray   # (n, T) float64, NaN = missing
    dates: pd.DatetimeIndex


def load_sgcc(path: Path) -> RawDataset:
    df = pd.read_csv(path, low_memory=False)
    date_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    dates = pd.to_datetime(pd.Index(date_cols), format=DATE_FORMAT)

    # Chronological order with every calendar day present.
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    readings = df[date_cols].apply(pd.to_numeric, errors="coerce")
    readings.columns = dates
    readings = readings.T.groupby(level=0).first().T  # guards against duplicate date columns
    readings = readings.reindex(columns=full_range)

    values = readings.to_numpy(dtype=float)
    keep = ~np.isnan(values).all(axis=1)
    return RawDataset(
        ids=df[ID_COL].to_numpy()[keep],
        labels=df[LABEL_COL].to_numpy(dtype=np.int8)[keep],
        readings=values[keep],
        dates=full_range,
    )


# ---------------------------------------------------------------------------
# Customer-level split
# ---------------------------------------------------------------------------

def stratified_split(labels: np.ndarray, val_frac: float, test_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Return row indices for train/val/test, stratified by label. Each row lands in exactly one split."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for value in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == value))
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        splits["test"].append(idx[:n_test])
        splits["val"].append(idx[n_test:n_test + n_val])
        splits["train"].append(idx[n_test + n_val:])
    return {name: np.sort(np.concatenate(parts)) for name, parts in splits.items()}


# ---------------------------------------------------------------------------
# Fitted preprocessing
# ---------------------------------------------------------------------------

def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate interior gaps of at most `max_gap` days, per customer.

    Only each customer's own readings are used, so this leaks nothing across
    customers or splits. Leading/trailing gaps and longer gaps stay NaN.
    """
    filled = pd.DataFrame(values.T).interpolate(method="linear", limit_area="inside", axis=0).to_numpy().T
    # pandas' `limit` would partly fill long gaps; only fill gaps that are short as a whole.
    short_gap = np.isnan(values) & (_missing_run_lengths(values) <= max_gap)
    return np.where(short_gap, filled, values)


def _missing_run_lengths(values: np.ndarray) -> np.ndarray:
    """For every missing cell, the length of the run of consecutive missing cells it belongs to (0 elsewhere)."""
    missing = np.isnan(values)
    lengths = np.zeros(values.shape, dtype=np.int32)
    for row in range(values.shape[0]):
        m = missing[row]
        if not m.any():
            continue
        edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            lengths[row, s:e] = e - s
    return lengths


@dataclass
class PreprocessorConfig:
    cap_percentile: float = 99.9
    max_interp_gap: int = 7


@dataclass
class Preprocessor:
    config: PreprocessorConfig
    cap_value: float | None = None
    mean: float | None = None
    std: float | None = None

    def fit(self, train_readings: np.ndarray) -> "Preprocessor":
        finite = train_readings[np.isfinite(train_readings)]
        self.cap_value = float(np.percentile(finite, self.config.cap_percentile))
        logged = np.log1p(np.clip(finite, 0, self.cap_value))
        self.mean, self.std = float(logged.mean()), float(logged.std())
        return self

    def transform(self, readings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (values float32 (n, T), mask uint8 (n, T) with 1 = originally missing)."""
        if self.cap_value is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        mask = np.isnan(readings).astype(np.uint8)
        x = np.clip(readings, 0, self.cap_value)
        x = interpolate_short_gaps(x, self.config.max_interp_gap)
        x = (np.log1p(x) - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0).astype(np.float32), mask

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"config": asdict(self.config), "cap_value": self.cap_value, "mean": self.mean, "std": self.std}
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls(PreprocessorConfig(**state["config"]), state["cap_value"], state["mean"], state["std"])
