"""Split the SGCC dataset by customer, fit preprocessing on train, and save the processed splits.

    python scripts/preprocess.py                                  # data/raw/data.csv -> data/processed/
    python scripts/preprocess.py --val-frac 0.15 --test-frac 0.15 --seed 42

Outputs:
    data/processed/{train,val,test}.npz   values (n, T) float32, mask (n, T) uint8, y (n,), ids (n,)
    data/processed/splits.csv             CONS_NO, FLAG, split
    artifacts/scalers/preprocessor.json   fitted cap / mean / std (train only)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import (  # noqa: E402
    ID_COL,
    LABEL_COL,
    Preprocessor,
    PreprocessorConfig,
    load_sgcc,
    stratified_split,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "data.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", 42)))
    parser.add_argument("--cap-percentile", type=float, default=99.9)
    parser.add_argument("--max-interp-gap", type=int, default=7)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: dataset not found at {args.input}. Run scripts/extract_sgcc.py first.")
        return 1

    data = load_sgcc(args.input)
    print(f"Loaded {len(data.ids)} customers x {len(data.dates)} days "
          f"({data.dates[0].date()} .. {data.dates[-1].date()})")

    split_idx = stratified_split(data.labels, args.val_frac, args.test_frac, args.seed)

    config = PreprocessorConfig(cap_percentile=args.cap_percentile, max_interp_gap=args.max_interp_gap)
    preprocessor = Preprocessor(config).fit(data.readings[split_idx["train"]])
    preprocessor.save(args.artifacts_dir / "scalers" / "preprocessor.json")
    print(f"Fitted on train: cap={preprocessor.cap_value:.2f}, "
          f"log-mean={preprocessor.mean:.4f}, log-std={preprocessor.std:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "n_days": len(data.dates), "first_day": str(data.dates[0].date()), "splits": {}}
    split_rows = []
    for name, idx in split_idx.items():
        values, mask = preprocessor.transform(data.readings[idx])
        np.savez(args.out_dir / f"{name}.npz", values=values, mask=mask, y=data.labels[idx], ids=data.ids[idx])
        n_pos = int(data.labels[idx].sum())
        summary["splits"][name] = {"n": len(idx), "n_theft_labelled": n_pos, "positive_rate": n_pos / len(idx)}
        split_rows.append(pd.DataFrame({ID_COL: data.ids[idx], LABEL_COL: data.labels[idx], "split": name}))
        print(f"  {name:5s}: {len(idx):6d} customers, {n_pos:5d} theft-labelled ({n_pos / len(idx):.2%})")

    pd.concat(split_rows).to_csv(args.out_dir / "splits.csv", index=False)
    (args.artifacts_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "metadata" / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved processed splits to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
