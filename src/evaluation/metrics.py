"""Shared evaluation: imbalance-aware metrics and validation-set threshold selection.

Every model (baselines and deep models) is evaluated with these functions so
that the comparison table in the README is consistent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def select_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Threshold that maximises F1 on the VALIDATION set. Never call this with test data."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall point has no threshold.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
