"""Phase 9: final, ONE-TIME evaluation on the held-out test split.

    python scripts/evaluate_test.py

Order of operations (fixed in advance; see README section 11):
1. Select the deployed model using VALIDATION results only:
   architecture = highest MEAN validation PR-AUC over its training runs (seeds);
   weights      = that architecture's run with the highest validation PR-AUC.
   Written to artifacts/metadata/model.json BEFORE any test score is computed.
2. Check that src/inference (raw readings -> Predictor) reproduces the scores of
   the evaluation path on validation customers.
3. Score every trained model once on test, using its validation-chosen threshold.
4. Robustness: test metrics on customers with little missing data (missingness artefact check).
5. Error analysis (false positives / negatives) for the deployed model.

The script refuses to run again once test results exist (use --force only to
regenerate identical outputs, e.g. after a crash; never to re-select models).
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.evaluation.metrics import compute_metrics, select_threshold  # noqa: E402

RESULTS = PROJECT_ROOT / "reports" / "results"
FIGURES = PROJECT_ROOT / "reports" / "figures"
METADATA = PROJECT_ROOT / "artifacts" / "metadata"
BASELINES = ["missingness_only_lr", "logistic_regression", "random_forest"]
DEEP = ["cnn", "lstm", "cnn_lstm"]
DISPLAY = {"missingness_only_lr": "Missingness-only LR", "logistic_regression": "Logistic Regression",
           "random_forest": "Random Forest", "cnn": "1D-CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}
LOW_MISSING = 0.05


def model_path(run: str) -> Path:
    arch, _, name = run.partition("@")
    if arch in BASELINES:
        return Path("models") / "baselines" / f"{arch}.joblib"
    base = Path("models") / arch
    return (base / "runs" / name if name else base) / f"{arch}.keras"


def load_runs() -> dict[str, dict]:
    """All trained runs with their stored validation metrics."""
    runs = {}
    for file in ("baselines_validation.json", "deep_validation.json"):
        runs.update(json.loads((RESULTS / file).read_text(encoding="utf-8"))["models"])
    return {k: v for k, v in runs.items() if (PROJECT_ROOT / model_path(k)).exists()}


def select_deployed(runs: dict[str, dict]) -> tuple[str, pd.DataFrame]:
    table = pd.DataFrame([{"run": k, "arch": k.split("@")[0], "val_pr_auc": v["pr_auc"]} for k, v in runs.items()])
    by_arch = table.groupby("arch")["val_pr_auc"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    best_arch = by_arch.index[0]
    best_run = table[table.arch == best_arch].sort_values("val_pr_auc", ascending=False).run.iloc[0]
    return best_run, by_arch


def make_scorer(run: str, dates: pd.DatetimeIndex):
    arch = run.split("@")[0]
    path = PROJECT_ROOT / model_path(run)
    if arch in DEEP:
        from tensorflow import keras

        from src.models.deep import to_model_input
        model = keras.models.load_model(path)
        return lambda v, m: model.predict(to_model_input(v, m), batch_size=512, verbose=0).ravel()
    import joblib

    from src.features.summary import build_features
    bundle = joblib.load(path)
    return lambda v, m: bundle["model"].predict_proba(build_features(v, m, dates)[bundle["columns"]])[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Regenerate outputs (never to re-select models)")
    parser.add_argument("--inference-check-n", type=int, default=500)
    args = parser.parse_args()

    out_path = RESULTS / "test_results.json"
    if out_path.exists() and not args.force:
        print(f"Test results already exist at {out_path}. The test set is used once; refusing to run again.")
        return 1

    meta = json.loads((METADATA / "preprocessing_summary.json").read_text(encoding="utf-8"))
    dates = pd.date_range(meta["first_day"], periods=meta["n_days"], freq="D")
    val = np.load(PROJECT_ROOT / "data" / "processed" / "val.npz", allow_pickle=True)
    test = np.load(PROJECT_ROOT / "data" / "processed" / "test.npz", allow_pickle=True)

    # 1. Selection from validation results only.
    runs = load_runs()
    deployed, by_arch = select_deployed(runs)
    print("Validation PR-AUC by architecture (mean over runs):\n" + by_arch.round(4).to_string())
    print(f"Deployed model (chosen on validation): {deployed}")

    scorers, val_scores, thresholds = {}, {}, {}
    for run in runs:
        scorers[run] = make_scorer(run, dates)
        val_scores[run] = scorers[run](val["values"], val["mask"])
        thresholds[run] = select_threshold(val["y"], val_scores[run])
        stored = runs[run]["pr_auc"]
        recomputed = compute_metrics(val["y"], val_scores[run], thresholds[run])["pr_auc"]
        assert abs(stored - recomputed) < 1e-3, f"{run}: saved model does not reproduce validation PR-AUC"

    model_meta = {
        "model": deployed.split("@")[0],
        "run": deployed,
        "model_path": model_path(deployed).as_posix(),
        "preprocessor_path": "artifacts/scalers/preprocessor.json",
        "threshold": thresholds[deployed],
        "threshold_rule": "maximises F1 on the validation split",
        "selection_rule": "architecture with highest mean validation PR-AUC over runs; best run of that architecture",
        "validation_metrics": compute_metrics(val["y"], val_scores[deployed], thresholds[deployed]),
        "n_days": meta["n_days"],
        "first_day": meta["first_day"],
    }
    (METADATA / "model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    # 2. Inference module must reproduce the evaluation path (raw readings -> Predictor).
    from src.inference.predictor import Predictor
    from src.preprocessing.pipeline import load_sgcc
    raw = load_sgcc(PROJECT_ROOT / "data" / "raw" / "data.csv")
    index = {cid: i for i, cid in enumerate(raw.ids)}
    n = min(args.inference_check_n, len(val["ids"]))
    rows = [index[cid] for cid in val["ids"][:n]]
    predictor = Predictor(METADATA / "model.json")
    diff = np.abs(predictor.score(raw.readings[rows]) - val_scores[deployed][:n]).max()
    assert diff < 1e-4, f"Inference module disagrees with evaluation path (max diff {diff})"
    print(f"Inference check passed on {n} validation customers (max score diff {diff:.2e})")

    # 3. One-time test evaluation.
    low_missing = test["mask"].mean(axis=1) <= LOW_MISSING
    results = {"deployed": deployed, "n_test": int(len(test["y"])), "test_positive_rate": float(test["y"].mean()),
               "low_missing_subset": {"max_missing_fraction": LOW_MISSING, "n": int(low_missing.sum()),
                                      "positive_rate": float(test["y"][low_missing].mean())},
               "runs": {}}
    test_scores = {}
    for run in runs:
        test_scores[run] = scorers[run](test["values"], test["mask"])
        results["runs"][run] = {
            "test": compute_metrics(test["y"], test_scores[run], thresholds[run]),
            "test_low_missing": compute_metrics(test["y"][low_missing], test_scores[run][low_missing], thresholds[run]),
            "validation_pr_auc": runs[run]["pr_auc"],
        }
        t = results["runs"][run]["test"]
        print(f"{run:22s} test PR-AUC {t['pr_auc']:.3f}  ROC-AUC {t['roc_auc']:.3f}  F1 {t['f1']:.3f}  "
              f"P {t['precision']:.3f}  R {t['recall']:.3f}  FPR {t['fpr']:.3f}")

    summary = []
    for arch in BASELINES + DEEP:
        arch_runs = [r for r in runs if r.split("@")[0] == arch]
        if not arch_runs:
            continue
        for key in ("pr_auc", "roc_auc", "f1"):
            vals = [results["runs"][r]["test"][key] for r in arch_runs]
            summary.append({"arch": arch, "metric": key, "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n_runs": len(vals)})
    results["by_architecture"] = summary

    # 5. Error analysis for the deployed model.
    scores = test_scores[deployed]
    pred = scores >= thresholds[deployed]
    y = test["y"].astype(bool)
    groups = {"TP": pred & y, "FP": pred & ~y, "FN": ~pred & y, "TN": ~pred & ~y}
    observed = test["mask"] == 0
    with np.errstate(all="ignore"):
        cust_mean = np.where(observed, test["values"], np.nan)
        cust_mean = np.nanmean(cust_mean, axis=1)
    results["error_analysis"] = {
        g: {"n": int(m.sum()), "median_missing_fraction": float(np.median(test["mask"][m].mean(axis=1))) if m.any() else None,
            "median_scaled_consumption": float(np.nanmedian(cust_mean[m])) if m.any() else None,
            "median_risk_score": float(np.median(scores[m])) if m.any() else None}
        for g, m in groups.items()}
    pd.DataFrame({"CONS_NO": test["ids"], "label": test["y"], "risk_score": scores,
                  "group": np.select(list(groups.values()), list(groups.keys()), default="")}) \
        .sort_values("risk_score", ascending=False).to_csv(RESULTS / "test_predictions_deployed.csv", index=False)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_figures(runs, test, test_scores, thresholds, deployed)
    print(f"\nWrote {out_path}, {METADATA / 'model.json'} and figures in {FIGURES}")
    return 0


def plot_figures(runs, test, test_scores, thresholds, deployed) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    # Fixed categorical order from the validated reference palette (slot 1..6).
    colors = dict(zip(BASELINES + DEEP, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
    surface, ink2, muted, grid = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink2, "xtick.color": muted,
                         "ytick.color": muted, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                         "axes.titleweight": "bold", "legend.frameon": False})

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in BASELINES + DEEP:
        if arch not in runs:
            continue
        p, r, _ = precision_recall_curve(test["y"], test_scores[arch])
        ax.plot(r, p, color=colors[arch], linewidth=2 if arch == deployed.split("@")[0] else 1.4,
                label=DISPLAY[arch])
    ax.axhline(test["y"].mean(), color=muted, linewidth=1, linestyle="--", label="Random classifier")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall on the test set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "test_pr_curves.png", dpi=150)
    plt.close(fig)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test["y"], test_scores[deployed] >= thresholds[deployed])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="#ffffff" if v > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["NORMAL", "SUSPICIOUS"])
    ax.set_yticks([0, 1], ["Normal", "Theft-labelled"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")
    ax.set_title(f"Confusion matrix: {DISPLAY[deployed.split('@')[0]]} (test)")
    fig.tight_layout()
    fig.savefig(FIGURES / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
