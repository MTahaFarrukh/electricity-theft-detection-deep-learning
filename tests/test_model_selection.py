"""Tests for the pre-registered model-selection rule in scripts/evaluate_test.py."""

from pathlib import Path

from scripts.evaluate_test import model_path, select_deployed


def test_selection_uses_mean_over_runs_then_best_run():
    runs = {
        "random_forest": {"pr_auc": 0.47},           # one lucky run, but a single run
        "cnn": {"pr_auc": 0.45},
        "cnn@seed1": {"pr_auc": 0.46},
        "cnn_lstm": {"pr_auc": 0.48},
        "cnn_lstm@seed1": {"pr_auc": 0.40},          # mean 0.44 < cnn mean 0.455
    }
    deployed, by_arch = select_deployed(runs)
    assert by_arch.index[0] == "random_forest"
    runs["random_forest"]["pr_auc"] = 0.43
    deployed, by_arch = select_deployed(runs)
    assert by_arch.index[0] == "cnn" and deployed == "cnn@seed1"


def test_model_paths():
    assert model_path("random_forest") == Path("models/baselines/random_forest.joblib")
    assert model_path("cnn_lstm") == Path("models/cnn_lstm/cnn_lstm.keras")
    assert model_path("cnn@seed2") == Path("models/cnn/runs/seed2/cnn.keras")
