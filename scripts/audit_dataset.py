"""Command-line entry point for the dataset audit.

Examples (run from the project root):

    python scripts/audit_dataset.py data/raw/data.csv
    python scripts/audit_dataset.py data/raw/data.csv --id-col CONS_NO --label-col FLAG
    python scripts/audit_dataset.py readings.csv --layout long --time-col ts --value-col kwh
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import AuditConfig, run_audit, save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an electricity-consumption dataset.")
    parser.add_argument("path", type=Path, help="Dataset file (.csv, .xlsx, .parquet)")
    parser.add_argument("--id-col", help="Customer/meter ID column (auto-detected if omitted)")
    parser.add_argument("--label-col", help="Label column (auto-detected if omitted)")
    parser.add_argument("--layout", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--time-col", help="Timestamp column (long layout only)")
    parser.add_argument("--value-col", help="Consumption column (long layout only)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "audit")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: dataset not found at {args.path}. See data/README.md for how to obtain it.")
        return 1

    cfg = AuditConfig(
        path=args.path,
        id_col=args.id_col,
        label_col=args.label_col,
        layout=args.layout,
        time_col=args.time_col,
        value_col=args.value_col,
    )
    report = run_audit(cfg)
    json_path, md_path = save_report(report, args.output_dir)

    print(json.dumps(report["general"], indent=2))
    print("\nNotes:")
    for note in report["notes"]:
        print(f"  - {note}")
    print(f"\nFull report written to:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
