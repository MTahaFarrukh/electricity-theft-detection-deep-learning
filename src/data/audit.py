"""Dataset audit for electricity-consumption data.

The goal of this module is to *inspect* a dataset before any preprocessing
decisions are made. It deliberately makes no assumptions about column names:
it detects candidate ID / label / date columns and reports what it found, and
the user can override any detection from the command line.

Two layouts are supported:

* wide  - one row per customer/meter, one column per reading date
          (e.g. ID, LABEL, 2014-01-01, 2014-01-02, ...)
* long  - one row per reading (e.g. ID, TIMESTAMP, CONSUMPTION, [LABEL])

Outputs a JSON file (machine-readable) and a Markdown file (human-readable).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAME_HINTS = ("flag", "label", "class", "target", "theft", "fraud", "y")
ID_NAME_HINTS = ("id", "cons", "meter", "customer", "consumer", "user", "account")
TIME_NAME_HINTS = ("date", "time", "timestamp", "datetime", "day")

# A wide-format dataset is assumed when at least this fraction of columns
# have names that parse as dates.
WIDE_DATE_COLUMN_FRACTION = 0.5


@dataclass
class AuditConfig:
    path: Path
    id_col: str | None = None
    label_col: str | None = None
    layout: str = "auto"  # auto | wide | long
    time_col: str | None = None
    value_col: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV/Excel/Parquet file with all columns kept as-is."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, low_memory=False)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def parse_column_names_as_dates(columns: list[str]) -> pd.Series:
    """Try to parse each column name as a date. Returns NaT where it fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas warns about format inference
        return pd.Series(pd.to_datetime(pd.Index(columns, dtype=str), errors="coerce"), index=columns)


def find_binary_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].dropna().nunique() == 2]


def _name_matches(name: str, hints: tuple[str, ...]) -> bool:
    lowered = str(name).lower()
    return any(h == lowered or h in lowered.replace("-", "_").split("_") or lowered.startswith(h) for h in hints)


def detect_label_column(df: pd.DataFrame, exclude: set[str], notes: list[str]) -> str | None:
    candidates = find_binary_columns(df, exclude)
    named = [c for c in candidates if _name_matches(c, LABEL_NAME_HINTS)]
    if len(named) == 1:
        notes.append(f"Label column auto-detected as '{named[0]}' (binary, name matches a label hint).")
        return named[0]
    if len(candidates) == 1:
        notes.append(f"Label column auto-detected as '{candidates[0]}' (only binary column). VERIFY THIS.")
        return candidates[0]
    notes.append(
        f"Label column NOT auto-detected. Binary candidates: {candidates[:10]}. "
        "Pass --label-col explicitly."
    )
    return None


def detect_id_column(df: pd.DataFrame, exclude: set[str], notes: list[str]) -> str | None:
    named = [c for c in df.columns if c not in exclude and _name_matches(c, ID_NAME_HINTS)]
    if named:
        notes.append(f"ID column auto-detected as '{named[0]}' (name matches an ID hint).")
        return named[0]
    # Fall back to a non-numeric column with mostly unique values.
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object and df[c].nunique(dropna=True) >= 0.9 * len(df):
            notes.append(f"ID column guessed as '{c}' (mostly unique text). VERIFY THIS.")
            return c
    notes.append("ID column NOT detected. Pass --id-col explicitly.")
    return None


def detect_layout(df: pd.DataFrame, notes: list[str]) -> str:
    parsed = parse_column_names_as_dates(list(df.columns))
    fraction = parsed.notna().mean()
    if fraction >= WIDE_DATE_COLUMN_FRACTION:
        notes.append(f"Layout detected as WIDE: {parsed.notna().sum()} of {len(df.columns)} column names parse as dates.")
        return "wide"
    notes.append(f"Layout detected as LONG: only {parsed.notna().sum()} column names parse as dates.")
    return "long"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def numeric_summary(values: np.ndarray) -> dict:
    """Summary statistics of a flat array of readings (NaNs ignored)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    pct = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99, 99.9])
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "p01": float(pct[0]),
        "p05": float(pct[1]),
        "p25": float(pct[2]),
        "median": float(pct[3]),
        "p75": float(pct[4]),
        "p95": float(pct[5]),
        "p99": float(pct[6]),
        "p99_9": float(pct[7]),
        "max": float(finite.max()),
        "negative_count": int((finite < 0).sum()),
        "zero_count": int((finite == 0).sum()),
        "zero_fraction": float((finite == 0).mean()),
    }


def class_distribution(labels: pd.Series) -> dict:
    counts = labels.value_counts(dropna=False)
    total = int(counts.sum())
    return {
        str(k): {"count": int(v), "fraction": float(v / total)} for k, v in counts.items()
    }


def to_numeric_counting_invalid(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce to numeric; count entries that were present but not numbers."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    invalid = int((numeric.isna() & frame.notna()).sum().sum())
    return numeric, invalid


# ---------------------------------------------------------------------------
# Audits per layout
# ---------------------------------------------------------------------------

def audit_wide(df: pd.DataFrame, cfg: AuditConfig) -> dict:
    notes = cfg.notes
    parsed_dates = parse_column_names_as_dates(list(df.columns))
    date_cols = [c for c in df.columns if pd.notna(parsed_dates[c])]
    meta_cols = [c for c in df.columns if c not in date_cols]

    label_col = cfg.label_col or detect_label_column(df[meta_cols], set(), notes)
    id_col = cfg.id_col or detect_id_column(df[meta_cols], {label_col} if label_col else set(), notes)

    dates = pd.Series(parsed_dates[date_cols].values, index=date_cols)
    sorted_dates = dates.sort_values()
    is_sorted = bool((dates.values == sorted_dates.values).all())
    diffs = sorted_dates.diff().dropna()
    full_range = pd.date_range(sorted_dates.min(), sorted_dates.max(), freq="D")
    missing_dates = full_range.difference(pd.DatetimeIndex(sorted_dates.unique()))

    readings, invalid_count = to_numeric_counting_invalid(df[date_cols])
    values = readings.to_numpy(dtype=float)
    missing_mask = np.isnan(values)
    per_customer_missing = missing_mask.mean(axis=1)
    per_customer_valid = (~missing_mask).sum(axis=1)

    report: dict = {
        "layout": "wide",
        "id_column": id_col,
        "label_column": label_col,
        "metadata_columns": meta_cols,
        "n_date_columns": len(date_cols),
        "date_columns": {
            "first": str(sorted_dates.min().date()),
            "last": str(sorted_dates.max().date()),
            "columns_in_chronological_order": is_sorted,
            "duplicate_date_columns": int(dates.duplicated().sum()),
            "most_common_step": str(diffs.mode().iloc[0]) if len(diffs) else None,
            "calendar_days_in_range": len(full_range),
            "calendar_days_missing_as_columns": len(missing_dates),
            "example_original_column_names": date_cols[:3],
        },
        "invalid_non_numeric_readings": invalid_count,
        "missing_readings": {
            "total_cells": int(values.size),
            "missing_cells": int(missing_mask.sum()),
            "missing_fraction": float(missing_mask.mean()),
            "customers_with_no_missing": int((per_customer_missing == 0).sum()),
            "customers_all_missing": int((per_customer_missing == 1).sum()),
            "per_customer_missing_fraction": numeric_summary(per_customer_missing),
        },
        "readings_per_customer": numeric_summary(per_customer_valid.astype(float)),
        "consumption": numeric_summary(values.ravel()),
    }

    if id_col:
        ids = df[id_col]
        report["n_customers"] = int(ids.nunique(dropna=True))
        report["duplicate_ids"] = int(ids.duplicated().sum())
        report["missing_ids"] = int(ids.isna().sum())
        dup_ids = ids[ids.duplicated(keep=False)]
        if len(dup_ids):
            # Are duplicate-ID rows exact copies, or conflicting records?
            dup_rows = df[ids.isin(dup_ids)]
            report["duplicate_id_rows_that_are_exact_copies"] = int(dup_rows.duplicated(keep=False).sum())
            if label_col:
                conflicting = dup_rows.groupby(id_col)[label_col].nunique()
                report["duplicate_ids_with_conflicting_labels"] = int((conflicting > 1).sum())

    if label_col:
        labels = df[label_col]
        report["class_distribution"] = class_distribution(labels)
        # Per-class missingness / consumption - an early hint at whether
        # missingness itself is informative (important for preprocessing).
        per_class = {}
        for value in labels.dropna().unique():
            mask = (labels == value).to_numpy()
            per_class[str(value)] = {
                "n_customers": int(mask.sum()),
                "mean_missing_fraction": float(per_customer_missing[mask].mean()),
                "consumption": numeric_summary(values[mask].ravel()),
            }
        report["per_class"] = per_class

    return report


def audit_long(df: pd.DataFrame, cfg: AuditConfig) -> dict:
    notes = cfg.notes
    time_col = cfg.time_col
    if time_col is None:
        for c in df.columns:
            if _name_matches(c, TIME_NAME_HINTS):
                time_col = c
                notes.append(f"Time column auto-detected as '{c}'.")
                break
    if time_col is None:
        notes.append("Time column NOT detected. Pass --time-col explicitly.")

    exclude = {time_col} if time_col else set()
    label_col = cfg.label_col or detect_label_column(df, exclude, notes)
    if label_col:
        exclude.add(label_col)
    id_col = cfg.id_col or detect_id_column(df, exclude, notes)
    if id_col:
        exclude.add(id_col)

    value_col = cfg.value_col
    if value_col is None:
        numeric_left = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_left) == 1:
            value_col = numeric_left[0]
            notes.append(f"Consumption column auto-detected as '{value_col}' (only remaining numeric column).")
        else:
            notes.append(f"Consumption column NOT detected; numeric candidates {numeric_left}. Pass --value-col.")

    report: dict = {
        "layout": "long",
        "id_column": id_col,
        "label_column": label_col,
        "time_column": time_col,
        "value_column": value_col,
    }

    if time_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(df[time_col], errors="coerce")
        report["time"] = {
            "unparseable_timestamps": int((ts.isna() & df[time_col].notna()).sum()),
            "missing_timestamps": int(df[time_col].isna().sum()),
            "first": str(ts.min()),
            "last": str(ts.max()),
            "example_raw_values": df[time_col].dropna().astype(str).head(3).tolist(),
        }
        if id_col:
            report["duplicate_id_time_pairs"] = int(df.duplicated(subset=[id_col, time_col]).sum())
            ordered = df.assign(_ts=ts).groupby(id_col)["_ts"].apply(lambda s: s.is_monotonic_increasing)
            report["customers_with_time_ordered_rows"] = int(ordered.sum())

    if value_col:
        numeric, invalid = to_numeric_counting_invalid(df[[value_col]])
        report["invalid_non_numeric_readings"] = invalid
        report["missing_readings"] = int(numeric[value_col].isna().sum())
        report["consumption"] = numeric_summary(numeric[value_col].to_numpy(dtype=float))

    if id_col:
        report["n_customers"] = int(df[id_col].nunique(dropna=True))
        report["readings_per_customer"] = numeric_summary(df.groupby(id_col).size().to_numpy(dtype=float))

    if label_col:
        if id_col:
            # Is the label constant per customer (customer-level) or per reading?
            labels_per_customer = df.groupby(id_col)[label_col].nunique()
            report["label_is_constant_per_customer"] = bool((labels_per_customer <= 1).all())
            report["class_distribution_customers"] = class_distribution(df.groupby(id_col)[label_col].first())
        report["class_distribution_rows"] = class_distribution(df[label_col])

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_audit(cfg: AuditConfig) -> dict:
    df = load_table(cfg.path)

    layout = cfg.layout if cfg.layout != "auto" else detect_layout(df, cfg.notes)

    general = {
        "file": str(cfg.path),
        "file_size_mb": round(cfg.path.stat().st_size / 1e6, 2),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "first_columns": [str(c) for c in df.columns[:10]],
        "last_columns": [str(c) for c in df.columns[-5:]],
    }

    detail = audit_wide(df, cfg) if layout == "wide" else audit_long(df, cfg)
    return {"general": general, **detail, "notes": cfg.notes}


def report_to_markdown(report: dict) -> str:
    """Render the audit report as readable Markdown."""
    g = report["general"]
    lines = [
        "# Dataset Audit Report",
        "",
        "_Generated automatically by `scripts/audit_dataset.py`. All numbers are computed from the file; nothing is assumed._",
        "",
        "## General",
        "",
        "| Property | Value |",
        "|---|---|",
    ]
    for key in ("file", "file_size_mb", "n_rows", "n_columns", "memory_mb", "exact_duplicate_rows"):
        lines.append(f"| {key} | {g[key]} |")
    lines.append(f"| dtype_counts | {g['dtype_counts']} |")
    lines.append(f"| first_columns | {g['first_columns']} |")
    lines += ["", "## Detection notes", ""]
    lines += [f"- {n}" for n in report["notes"]]
    lines += ["", "## Details", "", "```json"]
    detail = {k: v for k, v in report.items() if k not in ("general", "notes")}
    lines.append(json.dumps(detail, indent=2))
    lines += ["```", ""]
    return "\n".join(lines)


def save_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_audit.json"
    md_path = output_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path
