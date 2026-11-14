# Dataset Audit Report

_Generated automatically by `scripts/audit_dataset.py`. All numbers are computed from the file; nothing is assumed._

## General

| Property | Value |
|---|---|
| file | data\raw\data.csv |
| file_size_mb | 175.19 |
| n_rows | 42372 |
| n_columns | 1036 |
| memory_mb | 354.27 |
| exact_duplicate_rows | 0 |
| dtype_counts | {'float64': 1034, 'object': 1, 'int64': 1} |
| first_columns | ['CONS_NO', 'FLAG', '2014/1/1', '2014/1/10', '2014/1/11', '2014/1/12', '2014/1/13', '2014/1/14', '2014/1/15', '2014/1/16'] |

## Detection notes

- Layout detected as WIDE: 1034 of 1036 column names parse as dates.
- Label column auto-detected as 'FLAG' (binary, name matches a label hint).
- ID column auto-detected as 'CONS_NO' (name matches an ID hint).

## Details

```json
{
  "layout": "wide",
  "id_column": "CONS_NO",
  "label_column": "FLAG",
  "metadata_columns": [
    "CONS_NO",
    "FLAG"
  ],
  "n_date_columns": 1034,
  "date_columns": {
    "first": "2014-01-01",
    "last": "2016-10-31",
    "columns_in_chronological_order": false,
    "duplicate_date_columns": 0,
    "most_common_step": "1 days 00:00:00",
    "calendar_days_in_range": 1035,
    "calendar_days_missing_as_columns": 1,
    "example_original_column_names": [
      "2014/1/1",
      "2014/1/10",
      "2014/1/11"
    ]
  },
  "invalid_non_numeric_readings": 0,
  "missing_readings": {
    "total_cells": 43812648,
    "missing_cells": 11233528,
    "missing_fraction": 0.25639920234905683,
    "customers_with_no_missing": 0,
    "customers_all_missing": 5,
    "per_customer_missing_fraction": {
      "count": 42372,
      "mean": 0.2563992023490569,
      "std": 0.28825487352140844,
      "min": 0.0009671179883945841,
      "p01": 0.0009671179883945841,
      "p05": 0.0029013539651837525,
      "p25": 0.012572533849129593,
      "median": 0.10638297872340426,
      "p75": 0.620164410058027,
      "p95": 0.7108317214700194,
      "p99": 0.7475822050290135,
      "p99_9": 0.9468085106382979,
      "max": 1.0,
      "negative_count": 0,
      "zero_count": 0,
      "zero_fraction": 0.0
    }
  },
  "readings_per_customer": {
    "count": 42372,
    "mean": 768.8832247710752,
    "std": 298.0555392211363,
    "min": 0.0,
    "p01": 261.0,
    "p05": 299.0,
    "p25": 392.75,
    "median": 924.0,
    "p75": 1021.0,
    "p95": 1031.0,
    "p99": 1033.0,
    "p99_9": 1033.0,
    "max": 1033.0,
    "negative_count": 0,
    "zero_count": 5,
    "zero_fraction": 0.00011800245445105258
  },
  "consumption": {
    "count": 32579120,
    "mean": 9.258860091423593,
    "std": 273.22410350863214,
    "min": 0.0,
    "p01": 0.0,
    "p05": 0.0,
    "p25": 0.73,
    "median": 4.59,
    "p75": 9.74,
    "p95": 25.24,
    "p99": 69.79,
    "p99_9": 371.85881000004713,
    "max": 800003.32,
    "negative_count": 0,
    "zero_count": 5788603,
    "zero_fraction": 0.17767831052526895
  },
  "n_customers": 42372,
  "duplicate_ids": 0,
  "missing_ids": 0,
  "class_distribution": {
    "0": {
      "count": 38757,
      "fraction": 0.914684225431889
    },
    "1": {
      "count": 3615,
      "fraction": 0.08531577456811101
    }
  },
  "per_class": {
    "1": {
      "n_customers": 3615,
      "mean_missing_fraction": 0.3146873520229219,
      "consumption": {
        "count": 2561637,
        "mean": 27.53089734985869,
        "std": 493.3715080030054,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.0,
        "p25": 2.2,
        "median": 7.01,
        "p75": 16.01,
        "p95": 73.73,
        "p99": 333.89,
        "p99_9": 2108.8368000000714,
        "max": 514991.78,
        "negative_count": 0,
        "zero_count": 310660,
        "zero_fraction": 0.12127401345311611
      }
    },
    "0": {
      "n_customers": 38757,
      "mean_missing_fraction": 0.2509624641837958,
      "consumption": {
        "count": 30017483,
        "mean": 7.699557910538326,
        "std": 245.3945277043484,
        "min": 0.0,
        "p01": 0.0,
        "p05": 0.0,
        "p25": 0.65,
        "median": 4.43,
        "p75": 9.38,
        "p95": 23.05,
        "p99": 53.34,
        "p99_9": 211.1,
        "max": 800003.32,
        "negative_count": 0,
        "zero_count": 5477943,
        "zero_fraction": 0.18249174989122172
      }
    }
  }
}
```
