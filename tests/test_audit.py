"""Tests for src/data/audit.py.

The tables below are tiny hand-written fixtures used ONLY to check the audit
logic. They are not a substitute for the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.audit import AuditConfig, run_audit, save_report


@pytest.fixture
def wide_csv(tmp_path):
    # Date columns deliberately out of order, with one calendar day absent (01-04).
    df = pd.DataFrame(
        {
            "METER_ID": ["A", "B", "C", "C"],
            "FLAG": [0, 1, 0, 0],
            "2020-01-02": [1.0, 2.0, np.nan, np.nan],
            "2020-01-01": [1.5, "bad", 3.0, 3.0],
            "2020-01-03": [np.nan, -1.0, 0.0, 0.0],
            "2020-01-05": [2.0, 2.0, 2.0, 2.0],
        }
    )
    path = tmp_path / "wide.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def long_csv(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B", "B", "B"],
            "timestamp": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-01", "2020-01-02", "not-a-date"],
            "kwh": [1.0, 2.0, 2.0, 0.5, np.nan, 0.7],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    path = tmp_path / "long.csv"
    df.to_csv(path, index=False)
    return path


def test_wide_layout_detection_and_counts(wide_csv):
    report = run_audit(AuditConfig(path=wide_csv))

    assert report["layout"] == "wide"
    assert report["id_column"] == "METER_ID"
    assert report["label_column"] == "FLAG"
    assert report["n_date_columns"] == 4
    assert report["general"]["exact_duplicate_rows"] == 1
    assert report["duplicate_ids"] == 1
    assert report["n_customers"] == 3

    dates = report["date_columns"]
    assert dates["first"] == "2020-01-01"
    assert dates["last"] == "2020-01-05"
    assert dates["columns_in_chronological_order"] is False
    assert dates["calendar_days_missing_as_columns"] == 1

    assert report["invalid_non_numeric_readings"] == 1  # the "bad" cell
    assert report["consumption"]["negative_count"] == 1
    assert report["class_distribution"]["0"]["count"] == 3
    assert report["class_distribution"]["1"]["count"] == 1


def test_wide_missing_counts_include_invalid_cells(wide_csv):
    report = run_audit(AuditConfig(path=wide_csv))
    # NaNs: A(1) + B("bad"->1) + C(1) + C(1) = 4 of 16 cells
    assert report["missing_readings"]["missing_cells"] == 4
    assert report["missing_readings"]["total_cells"] == 16


def test_long_layout_detection(long_csv):
    report = run_audit(AuditConfig(path=long_csv))

    assert report["layout"] == "long"
    assert report["id_column"] == "customer_id"
    assert report["time_column"] == "timestamp"
    assert report["value_column"] == "kwh"
    assert report["label_column"] == "label"
    assert report["n_customers"] == 2
    assert report["duplicate_id_time_pairs"] == 1
    assert report["time"]["unparseable_timestamps"] == 1
    assert report["label_is_constant_per_customer"] is True


def test_explicit_columns_override_detection(wide_csv):
    report = run_audit(AuditConfig(path=wide_csv, id_col="METER_ID", label_col="FLAG", layout="wide"))
    assert report["label_column"] == "FLAG"
    assert not any("auto-detected" in n and "Label" in n for n in report["notes"])


def test_save_report_writes_files(wide_csv, tmp_path):
    report = run_audit(AuditConfig(path=wide_csv))
    json_path, md_path = save_report(report, tmp_path / "out")
    assert json_path.exists() and md_path.exists()
    assert "Dataset Audit Report" in md_path.read_text(encoding="utf-8")
