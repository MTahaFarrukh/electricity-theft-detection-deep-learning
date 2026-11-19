"""REST endpoints (README section 13)."""

from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.db import Alert, Meter, Prediction, Reading
from backend.app.schemas.api import (
    AlertOut,
    AlertUpdate,
    BatchPredictRequest,
    MeterDetail,
    MeterSummary,
    PredictionOut,
    PredictRequest,
    PredictResponse,
)
from backend.app.services.scoring import ModelUnavailable, score_and_store
from backend.app.services.simulation import simulation_controller

router = APIRouter()


def get_session(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def get_predictor(request: Request):
    try:
        return request.app.state.model_service.predictor
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/health")
def health(request: Request):
    return {"status": "ok", **request.app.state.model_service.status()}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, session: Session = Depends(get_session), predictor=Depends(get_predictor)):
    _check_length(body.readings, predictor.n_days)
    return score_and_store(session, predictor, body.meter_id, body.readings)


@router.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(body: BatchPredictRequest, session: Session = Depends(get_session), predictor=Depends(get_predictor)):
    for item in body.items:
        _check_length(item.readings, predictor.n_days)
    return [score_and_store(session, predictor, item.meter_id, item.readings) for item in body.items]


@router.get("/meters", response_model=list[MeterSummary])
def list_meters(status: str | None = Query(None, pattern="^(SUSPICIOUS|NORMAL)$"),
                limit: int = Query(200, le=2000), session: Session = Depends(get_session)):
    rows = [_meter_summary(session, m) for m in session.scalars(select(Meter))]
    if status:
        rows = [r for r in rows if r.latest_prediction == status]
    rows.sort(key=lambda r: -(r.latest_risk_score or 0))
    return rows[:limit]


@router.get("/meters/{meter_id}", response_model=MeterDetail)
def meter_detail(meter_id: str, session: Session = Depends(get_session)):
    meter = session.scalar(select(Meter).where(Meter.meter_id == meter_id))
    if meter is None:
        raise HTTPException(404, f"Meter {meter_id} not found")
    reading = session.scalar(select(Reading).where(Reading.meter_pk == meter.id).order_by(Reading.id.desc()))
    alerts = session.scalars(select(Alert).where(Alert.meter_pk == meter.id).order_by(Alert.created_at.desc()))
    return MeterDetail(
        **_meter_summary(session, meter).model_dump(),
        first_day=reading.first_day if reading else None,
        readings=reading.values if reading else [],
        predictions=[PredictionOut.model_validate(p, from_attributes=True) for p in meter.predictions],
        alerts=[_alert_out(a) for a in alerts],
    )


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(status: str | None = Query(None, pattern="^(OPEN|REVIEWED|DISMISSED)$"),
                min_risk: float = Query(0.0, ge=0, le=1), limit: int = Query(200, le=2000),
                session: Session = Depends(get_session)):
    q = select(Alert).where(Alert.risk_score >= min_risk).order_by(Alert.created_at.desc()).limit(limit)
    if status:
        q = q.where(Alert.status == status)
    return [_alert_out(a) for a in session.scalars(q)]


@router.patch("/alerts/{alert_id}", response_model=AlertOut)
def update_alert(alert_id: int, body: AlertUpdate, session: Session = Depends(get_session)):
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"Alert {alert_id} not found")
    alert.status = body.status
    session.commit()
    return _alert_out(alert)


@router.get("/dashboard/summary")
def dashboard_summary(request: Request, session: Session = Depends(get_session)):
    latest = _latest_predictions(session)
    scores = np.array([p.risk_score for p in latest])
    counts, edges = np.histogram(scores, bins=10, range=(0, 1))
    alert_status = dict(session.execute(select(Alert.status, func.count()).group_by(Alert.status)).all())
    recent = session.scalars(select(Alert).order_by(Alert.created_at.desc()).limit(10))
    return {
        "n_meters": len(latest),
        "n_suspicious": sum(p.prediction == "SUSPICIOUS" for p in latest),
        "n_normal": sum(p.prediction == "NORMAL" for p in latest),
        "n_predictions": session.scalar(select(func.count()).select_from(Prediction)),
        "alerts_by_status": {s: alert_status.get(s, 0) for s in ("OPEN", "REVIEWED", "DISMISSED")},
        "risk_histogram": [{"from": round(float(edges[i]), 1), "to": round(float(edges[i + 1]), 1), "count": int(c)}
                           for i, c in enumerate(counts)],
        "recent_alerts": [_alert_out(a).model_dump() for a in recent],
        "model": request.app.state.model_service.status(),
    }


@router.get("/model/metrics")
def model_metrics(request: Request):
    """Stored evaluation results only; nothing is computed here."""
    settings = request.app.state.settings
    out = {}
    for key, path in [("deployed_model", settings.model_metadata_path),
                      ("test", settings.results_dir / "test_results.json"),
                      ("validation_baselines", settings.results_dir / "baselines_validation.json"),
                      ("validation_deep", settings.results_dir / "deep_validation.json")]:
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


@router.get("/simulation/status")
def simulation_status():
    return simulation_controller.status()


@router.post("/simulation/start")
def simulation_start(request: Request, limit: int = Query(30, ge=5, le=500), interval: float = Query(1.0, ge=0.1, le=10.0), predictor=Depends(get_predictor)):
    return simulation_controller.start(request.app.state.session_factory, predictor, limit=limit, interval=interval)


@router.post("/simulation/stop")
def simulation_stop():
    return simulation_controller.stop()


@router.post("/simulation/reset")
def simulation_reset(request: Request):
    return simulation_controller.reset_data(request.app.state.session_factory)


# --- helpers -----------------------------------------------------------------

def _check_length(readings: list, n_days: int) -> None:
    if len(readings) != n_days:
        raise HTTPException(422, f"Expected {n_days} daily readings, got {len(readings)}")


def _latest_predictions(session: Session) -> list[Prediction]:
    latest_ids = select(func.max(Prediction.id)).group_by(Prediction.meter_pk)
    return list(session.scalars(select(Prediction).where(Prediction.id.in_(latest_ids))))


def _meter_summary(session: Session, meter: Meter) -> MeterSummary:
    last = meter.predictions[-1] if meter.predictions else None
    open_alerts = session.scalar(select(func.count()).select_from(Alert)
                                 .where(Alert.meter_pk == meter.id, Alert.status == "OPEN"))
    return MeterSummary(meter_id=meter.meter_id, latest_prediction=last.prediction if last else None,
                        latest_risk_score=last.risk_score if last else None,
                        last_scored_at=last.created_at if last else None, open_alerts=open_alerts)


def _alert_out(alert: Alert) -> AlertOut:
    return AlertOut(id=alert.id, meter_id=alert.meter.meter_id, risk_score=alert.risk_score,
                    status=alert.status, created_at=alert.created_at)
