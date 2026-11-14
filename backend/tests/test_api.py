"""API tests with a fake predictor and a temporary SQLite file (no trained model needed)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

N_DAYS = 14


class FakePredictor:
    """Risk score = mean of observed readings / 10, so tests control the outcome."""
    threshold = 0.5
    n_days = N_DAYS
    model_name = "fake"
    metadata = {"run": "fake@test"}
    dates = pd.date_range("2014-01-01", periods=N_DAYS, freq="D")

    def score(self, readings):
        return np.clip(np.nanmean(readings, axis=1) / 10, 0, 1)


class FakeModelService:
    predictor = FakePredictor()

    def status(self):
        return {"model_loaded": True, "model": "fake@test", "threshold": 0.5}


@pytest.fixture
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}", results_dir=tmp_path,
                        model_metadata_path=tmp_path / "missing.json")
    return TestClient(create_app(settings, FakeModelService()))


def series(value, missing_days=0):
    return [None] * missing_days + [value] * (N_DAYS - missing_days)


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_without_meter_is_not_stored(client):
    r = client.post("/predict", json={"readings": series(8.0)}).json()
    assert r["prediction"] == "SUSPICIOUS" and r["risk_score"] == pytest.approx(0.8)
    assert "Investigation recommended" in r["recommendation"]
    assert client.get("/meters").json() == []


def test_wrong_length_is_rejected(client):
    assert client.post("/predict", json={"readings": [1.0] * 3}).status_code == 422


def test_alert_raised_once_per_series_and_meter_history(client):
    first = client.post("/predict", json={"meter_id": "M1", "readings": series(9.0, missing_days=2)}).json()
    again = client.post("/predict", json={"meter_id": "M1", "readings": series(9.0, missing_days=2)}).json()
    assert first["alert_raised"] and not first["duplicate"]
    assert again["duplicate"] and again["alert_raised"]
    assert len(client.get("/alerts").json()) == 1

    client.post("/predict", json={"meter_id": "M1", "readings": series(1.0)})    # new series, now normal
    client.post("/predict", json={"meter_id": "M2", "readings": series(2.0)})
    meters = client.get("/meters").json()
    assert [m["meter_id"] for m in meters] == ["M2", "M1"]               # sorted by latest risk
    assert meters[1]["latest_prediction"] == "NORMAL" and meters[1]["open_alerts"] == 1

    detail = client.get("/meters/M1").json()
    assert len(detail["predictions"]) == 2 and detail["readings"] == series(1.0)
    assert client.get("/meters/NOPE").status_code == 404


def test_alert_status_update_and_summary(client):
    client.post("/predict/batch", json={"items": [
        {"meter_id": "A", "readings": series(7.0)},
        {"meter_id": "B", "readings": series(1.0)},
    ]})
    alert = client.get("/alerts").json()[0]
    assert client.patch(f"/alerts/{alert['id']}", json={"status": "REVIEWED"}).json()["status"] == "REVIEWED"
    assert client.patch(f"/alerts/{alert['id']}", json={"status": "THEFT"}).status_code == 422

    s = client.get("/dashboard/summary").json()
    assert (s["n_meters"], s["n_suspicious"], s["n_normal"]) == (2, 1, 1)
    assert s["alerts_by_status"] == {"OPEN": 0, "REVIEWED": 1, "DISMISSED": 0}
    assert sum(b["count"] for b in s["risk_histogram"]) == 2


def test_model_metrics_reports_missing_files_as_null(client):
    assert client.get("/model/metrics").json()["test"] is None
