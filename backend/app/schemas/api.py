"""Request / response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    meter_id: str | None = Field(None, max_length=64, description="Store the result under this meter (optional)")
    readings: list[float | None] = Field(..., description="Daily kWh from the model's first day; null = missing")


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class PredictResponse(BaseModel):
    meter_id: str | None
    prediction: str
    risk_score: float
    threshold: float
    recommendation: str
    alert_raised: bool = False
    duplicate: bool = False


class PredictionOut(BaseModel):
    id: int
    risk_score: float
    prediction: str
    threshold: float
    model_run: str
    created_at: datetime


class AlertOut(BaseModel):
    id: int
    meter_id: str
    risk_score: float
    status: str
    created_at: datetime


class AlertUpdate(BaseModel):
    status: str = Field(..., pattern="^(OPEN|REVIEWED|DISMISSED)$")


class MeterSummary(BaseModel):
    meter_id: str
    latest_prediction: str | None
    latest_risk_score: float | None
    last_scored_at: datetime | None
    open_alerts: int


class MeterDetail(MeterSummary):
    first_day: str | None
    readings: list[float | None]
    predictions: list[PredictionOut]
    alerts: list[AlertOut]
