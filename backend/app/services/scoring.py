"""Model access for the API (wraps src/inference) and storage of predictions and alerts."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.db import Alert, Meter, Prediction, Reading
from src.inference.predictor import RECOMMENDATIONS


class ModelUnavailable(RuntimeError):
    pass


class ModelService:
    """Loads the deployed Predictor once (lazily) and shares it across requests. Never trains."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = Path(metadata_path)
        self._predictor = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def predictor(self):
        with self._lock:
            if self._predictor is None:
                if not self.metadata_path.exists():
                    self._error = f"{self.metadata_path} not found: run scripts/evaluate_test.py first"
                    raise ModelUnavailable(self._error)
                from src.inference.predictor import Predictor
                self._predictor = Predictor(self.metadata_path)
            return self._predictor

    def status(self) -> dict:
        try:
            p = self.predictor
            return {"model_loaded": True, "model": p.metadata.get("run", p.model_name), "threshold": p.threshold,
                    "n_days": p.n_days, "first_day": str(p.dates[0].date())}
        except Exception as exc:  # noqa: BLE001 - reported, not raised, by /health
            return {"model_loaded": False, "error": str(exc)}


def series_hash(readings: list[float | None]) -> str:
    return hashlib.sha256(json.dumps(readings, separators=(",", ":")).encode()).hexdigest()


def score_and_store(session: Session, predictor, meter_id: str | None, readings: list[float | None]) -> dict:
    """Score one series; if meter_id is given, store reading/prediction/alert without duplicating resubmissions."""
    raw = np.array([np.nan if v is None else v for v in readings], dtype=float)
    if meter_id is not None:
        existing = _existing_prediction(session, meter_id, series_hash(readings))
        if existing is not None:
            alert = session.scalar(select(Alert).where(Alert.prediction_id == existing.id))
            return _response(meter_id, existing.risk_score, existing.threshold, alert is not None, duplicate=True)

    score = float(predictor.score(raw[None, :])[0])
    threshold = predictor.threshold
    label = "SUSPICIOUS" if score >= threshold else "NORMAL"
    if meter_id is None:
        return _response(None, score, threshold, False)

    meter = session.scalar(select(Meter).where(Meter.meter_id == meter_id)) or Meter(meter_id=meter_id)
    session.add(meter)
    session.flush()
    reading = Reading(meter_pk=meter.id, first_day=str(predictor.dates[0].date()), values=readings,
                      series_hash=series_hash(readings))
    session.add(reading)
    session.flush()
    pred = Prediction(meter_pk=meter.id, reading_id=reading.id, risk_score=score, prediction=label,
                      threshold=threshold, model_run=predictor.metadata.get("run", predictor.model_name))
    session.add(pred)
    session.flush()
    alert_raised = label == "SUSPICIOUS"
    if alert_raised:
        session.add(Alert(meter_pk=meter.id, prediction_id=pred.id, risk_score=score))
    session.commit()
    return _response(meter_id, score, threshold, alert_raised)


def _existing_prediction(session: Session, meter_id: str, digest: str) -> Prediction | None:
    return session.scalar(
        select(Prediction).join(Reading, Prediction.reading_id == Reading.id).join(Meter, Prediction.meter_pk == Meter.id)
        .where(Meter.meter_id == meter_id, Reading.series_hash == digest)
    )


def _response(meter_id, score, threshold, alert_raised, duplicate=False) -> dict:
    label = "SUSPICIOUS" if score >= threshold else "NORMAL"
    return {"meter_id": meter_id, "prediction": label, "risk_score": score, "threshold": threshold,
            "recommendation": RECOMMENDATIONS[label], "alert_raised": alert_raised, "duplicate": duplicate}
