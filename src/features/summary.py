"""Hand-crafted per-customer features for the classical baselines.

Features are computed from the PREPROCESSED series (capped, interpolated,
log-scaled and standardised with training statistics) plus the missingness
mask, so they inherit the leak-free preprocessing and can be computed the
same way at inference time.

Groups:
    level / spread   overall distribution of the series
    shape / trend    changes over time (sudden long drops are a typical theft pattern)
    monthly          mean per calendar month (seasonality and level shifts)
    missingness      how much is missing and when - kept SEPARATE so the
                     missingness-only baseline can measure how much of the
                     signal comes from data-collection artefacts alone
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSINGNESS_PREFIX = "miss_"


def build_features(values: np.ndarray, mask: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """values: (n, T) preprocessed series; mask: (n, T) 1 = originally missing; dates: (T,)."""
    observed = mask == 0
    x = np.where(observed, values, np.nan)  # statistics use observed readings only
    with np.errstate(all="ignore"), _quiet():
        feats: dict[str, np.ndarray] = {
            "mean": np.nanmean(x, axis=1),
            "std": np.nanstd(x, axis=1),
            "min": np.nanmin(x, axis=1),
            "max": np.nanmax(x, axis=1),
        }
        for q in (5, 25, 50, 75, 95):
            feats[f"p{q:02d}"] = np.nanpercentile(x, q, axis=1)
        feats["iqr"] = feats["p75"] - feats["p25"]
        feats["mean_abs_diff"] = np.nanmean(np.abs(np.diff(x, axis=1)), axis=1)

        months = dates.to_period("M")
        month_keys = months.unique()
        monthly = np.stack([np.nanmean(x[:, months == m], axis=1) for m in month_keys], axis=1)
        for k, m in enumerate(month_keys):
            feats[f"month_{m}"] = monthly[:, k]
        monthly_diff = np.diff(monthly, axis=1)
        feats["largest_monthly_drop"] = -np.nanmin(monthly_diff, axis=1)
        feats["largest_monthly_rise"] = np.nanmax(monthly_diff, axis=1)
        feats["monthly_std"] = np.nanstd(monthly, axis=1)
        feats["last6_minus_first6"] = np.nanmean(monthly[:, -6:], axis=1) - np.nanmean(monthly[:, :6], axis=1)
        t = np.arange(x.shape[1], dtype=float)
        feats["trend_slope"] = _nan_slope(x, t)

        weekday = dates.dayofweek.to_numpy()
        feats["weekend_minus_weekday"] = np.nanmean(x[:, weekday >= 5], axis=1) - np.nanmean(x[:, weekday < 5], axis=1)

    pre_2016 = dates < pd.Timestamp("2016-01-01")
    first_obs = np.where(observed.any(axis=1), observed.argmax(axis=1), x.shape[1])
    feats[f"{MISSINGNESS_PREFIX}fraction"] = mask.mean(axis=1)
    feats[f"{MISSINGNESS_PREFIX}fraction_pre2016"] = mask[:, pre_2016].mean(axis=1)
    feats[f"{MISSINGNESS_PREFIX}fraction_2016"] = mask[:, ~pre_2016].mean(axis=1)
    feats[f"{MISSINGNESS_PREFIX}first_observed_day"] = first_obs.astype(float)
    feats[f"{MISSINGNESS_PREFIX}n_gaps"] = (np.diff(mask.astype(np.int8), axis=1) == 1).sum(axis=1).astype(float)

    frame = pd.DataFrame(feats)
    # Customers with no observed readings in a month etc. -> 0 (= training mean on the standardised scale).
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def missingness_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(MISSINGNESS_PREFIX)]


def _nan_slope(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least-squares slope per row, ignoring NaNs."""
    ok = ~np.isnan(x)
    n = ok.sum(axis=1)
    tt = np.where(ok, t, 0.0)
    xx = np.where(ok, x, 0.0)
    t_mean = tt.sum(axis=1) / n
    x_mean = xx.sum(axis=1) / n
    cov = (np.where(ok, (t - t_mean[:, None]) * (x - x_mean[:, None]), 0.0)).sum(axis=1)
    var = (np.where(ok, (t - t_mean[:, None]) ** 2, 0.0)).sum(axis=1)
    return cov / var


class _quiet:
    """Silence 'mean of empty slice' warnings for customers with empty months."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
