"""Train a deep classifier (cnn | lstm | cnn_lstm) and evaluate it on VALIDATION data.

    python scripts/train_deep.py --model cnn
    python scripts/train_deep.py --model cnn --epochs 40 --batch-size 128

Training: binary cross-entropy, Adam, balanced class weights (no SMOTE),
early stopping on validation PR-AUC with the best weights restored, fixed seed.
The test split is NOT touched here (Phase 9).

Outputs:
    models/<model>/<model>.keras                  best model
    models/<model>/history.json                   per-epoch metrics
    reports/results/deep_validation.json          validation metrics (one entry per model)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras  # noqa: E402

from src.evaluation.metrics import compute_metrics, select_threshold  # noqa: E402
from src.models.deep import BUILDERS, to_model_input  # noqa: E402


def load_split(processed_dir: Path, name: str):
    data = np.load(processed_dir / f"{name}.npz", allow_pickle=True)
    return to_model_input(data["values"], data["mask"]), data["y"].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", 42)))
    parser.add_argument("--run-name", help="Extra run (e.g. another seed): saved to models/<model>/runs/<name>/ "
                                           "and stored under '<model>@<name>' instead of replacing the main run")
    args = parser.parse_args()
    run_key = f"{args.model}@{args.run_name}" if args.run_name else args.model

    keras.utils.set_random_seed(args.seed)
    x_train, y_train = load_split(args.processed_dir, "train")
    x_val, y_val = load_split(args.processed_dir, "val")
    print(f"Train {x_train.shape}, val {x_val.shape}")

    pos = y_train.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}  # "balanced"

    model = BUILDERS[args.model](x_train.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), keras.metrics.AUC(name="roc_auc")],
    )
    model.summary(print_fn=lambda line: print(line) if "params" in line.lower() else None)

    out_dir = PROJECT_ROOT / "models" / args.model
    if args.run_name:
        out_dir = out_dir / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.model}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=args.patience,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_pr_auc", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3),
    ]

    start = time.time()
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, class_weight=class_weight, callbacks=callbacks, verbose=2,
    )
    seconds = round(time.time() - start, 1)

    scores = model.predict(x_val, batch_size=512, verbose=0).ravel()
    threshold = select_threshold(y_val, scores)
    metrics = compute_metrics(y_val.astype(int), scores, threshold)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    metrics.update({
        "train_seconds": seconds,
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmax(hist["val_pr_auc"])) + 1,
        "params": int(model.count_params()),
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("model", "processed_dir")},
    })
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    results_path = PROJECT_ROOT / "reports" / "results" / "deep_validation.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {
        "split": "validation", "positive_rate": float(y_val.mean()), "models": {}}
    results["models"][run_key] = metrics
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{run_key}: PR-AUC {metrics['pr_auc']:.3f}  ROC-AUC {metrics['roc_auc']:.3f}  "
          f"F1 {metrics['f1']:.3f}  P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
          f"FPR {metrics['fpr']:.3f}  best epoch {metrics['best_epoch']}/{metrics['epochs_run']}  ({seconds}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
