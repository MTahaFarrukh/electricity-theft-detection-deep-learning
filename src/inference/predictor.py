"""The single inference module, shared by the API, the evaluation scripts and the tests.

It applies exactly the preprocessing used in training (the fitted
`Preprocessor` from artifacts/scalers/preprocessor.json) and scores raw daily
readings with the deployed model described in artifacts/metadata/model.json.

A positive result means SUSPICIOUS (investigation recommended), never proof of theft.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import Preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "artifacts" / "metadata" / "model.json"
DEEP_MODELS = {"cnn", "lstm", "cnn_lstm"}

RECOMMENDATIONS = {
    "SUSPICIOUS": "Investigation recommended: high-risk consumption pattern.",
    "NORMAL": "No action needed: consumption pattern within the normal range.",
}


@dataclass
class Prediction:
    risk_score: float
    prediction: str          # "SUSPICIOUS" | "NORMAL"
    threshold: float
    recommendation: str


class Predictor:
    def __init__(self, metadata_path: Path = DEFAULT_METADATA):
        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.metadata = meta
        self.model_name: str = meta["model"]
        self.threshold: float = float(meta["threshold"])
        self.n_days: int = int(meta["n_days"])
        self.dates = pd.date_range(meta["first_day"], periods=self.n_days, freq="D")
        self.preprocessor = Preprocessor.load(PROJECT_ROOT / meta["preprocessor_path"])
        model_path = PROJECT_ROOT / meta["model_path"]
        if self.model_name in DEEP_MODELS:
            from tensorflow import keras  # imported lazily: baselines don't need TensorFlow
            self._model = keras.models.load_model(model_path)
        else:
            import joblib
            bundle = joblib.load(model_path)
            self._model, self._columns = bundle["model"], bundle["columns"]

    def score(self, readings: np.ndarray) -> np.ndarray:
        """Risk scores (0-1) for raw daily readings of shape (n, n_days); NaN = missing reading."""
        readings = np.asarray(readings, dtype=float)
        if readings.ndim == 1:
            readings = readings[None, :]
        if readings.shape[1] != self.n_days:
            raise ValueError(f"Expected {self.n_days} daily readings per meter, got {readings.shape[1]}")
        values, mask = self.preprocessor.transform(readings)
        return self.score_preprocessed(values, mask)

    def score_preprocessed(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Risk scores for already-preprocessed (values, mask), e.g. data/processed/*.npz."""
        if self.model_name in DEEP_MODELS:
            from src.models.deep import to_model_input
            return self._model.predict(to_model_input(values, mask), batch_size=512, verbose=0).ravel()
        from src.features.summary import build_features
        feats = build_features(values, mask, self.dates)[self._columns]
        return self._model.predict_proba(feats)[:, 1]

    def predict(self, readings: np.ndarray) -> list[Prediction]:
        return [self._to_prediction(s) for s in self.score(readings)]

    def _to_prediction(self, score: float) -> Prediction:
        label = "SUSPICIOUS" if score >= self.threshold else "NORMAL"
        return Prediction(float(score), label, self.threshold, RECOMMENDATIONS[label])
