"""Train the classical baselines on hand-crafted features and evaluate them on VALIDATION data.

    python scripts/train_baselines.py

Models:
    missingness_only_lr   Logistic Regression on missingness features only (heuristic baseline:
                          how much can be predicted from data-collection artefacts alone?)
    logistic_regression   Logistic Regression on all features (C tuned on validation PR-AUC)
    random_forest         Random Forest on all features

All use class weights for the imbalance. The test split is NOT touched here;
it is used once, in the final evaluation (Phase 9).

Outputs:
    models/baselines/<name>.joblib            fitted pipeline + feature columns + threshold
    reports/results/baselines_validation.json validation metrics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics, select_threshold  # noqa: E402
from src.features.summary import build_features, missingness_columns  # noqa: E402


def load_split(processed_dir: Path, name: str, dates: pd.DatetimeIndex):
    data = np.load(processed_dir / f"{name}.npz", allow_pickle=True)
    return build_features(data["values"], data["mask"], dates), data["y"].astype(int)


def make_lr(c: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "baselines")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "reports" / "results")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", 42)))
    args = parser.parse_args()

    summary_path = PROJECT_ROOT / "artifacts" / "metadata" / "preprocessing_summary.json"
    meta = json.loads(summary_path.read_text(encoding="utf-8"))
    dates = pd.date_range(meta["first_day"], periods=meta["n_days"], freq="D")

    x_train, y_train = load_split(args.processed_dir, "train", dates)
    x_val, y_val = load_split(args.processed_dir, "val", dates)
    miss_cols = missingness_columns(x_train)
    print(f"Features: {x_train.shape[1]} ({len(miss_cols)} missingness). "
          f"Train {len(y_train)}, val {len(y_val)}")

    candidates = {
        "missingness_only_lr": (miss_cols, [make_lr(1.0, args.seed)]),
        "logistic_regression": (list(x_train.columns), [make_lr(c, args.seed) for c in (0.01, 0.1, 1.0)]),
        "random_forest": (list(x_train.columns), [RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=args.seed)]),
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (cols, models) in candidates.items():
        start = time.time()
        best, best_ap = None, -1.0
        for model in models:  # hyper-parameter choice uses validation PR-AUC only
            model.fit(x_train[cols], y_train)
            ap = average_precision_score(y_val, model.predict_proba(x_val[cols])[:, 1])
            if ap > best_ap:
                best, best_ap = model, ap
        scores = best.predict_proba(x_val[cols])[:, 1]
        threshold = select_threshold(y_val, scores)
        metrics = compute_metrics(y_val, scores, threshold)
        metrics["train_seconds"] = round(time.time() - start, 1)
        if name == "logistic_regression":
            metrics["chosen_C"] = float(best[-1].C)
        results[name] = metrics
        joblib.dump({"model": best, "columns": cols, "threshold": threshold}, args.models_dir / f"{name}.joblib")
        print(f"{name:22s} PR-AUC {metrics['pr_auc']:.3f}  ROC-AUC {metrics['roc_auc']:.3f}  "
              f"F1 {metrics['f1']:.3f}  P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
              f"FPR {metrics['fpr']:.3f}  ({metrics['train_seconds']}s)")

    rf = joblib.load(args.models_dir / "random_forest.joblib")["model"]
    importance = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    results["random_forest"]["top_features"] = {k: round(float(v), 4) for k, v in importance.head(15).items()}

    out = {"split": "validation", "positive_rate": float(y_val.mean()), "models": results}
    (args.results_dir / "baselines_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nRandom forest top features:\n" + importance.head(10).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
