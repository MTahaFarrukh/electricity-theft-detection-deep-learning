"""SQLite schema: meters, readings, predictions and alerts (one file, SQLAlchemy 2.0)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Meter(Base):
    __tablename__ = "meters"
    id: Mapped[int] = mapped_column(primary_key=True)
    meter_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="meter", order_by="Prediction.created_at")


class Reading(Base):
    """One submitted daily series for a meter. `series_hash` identifies identical resubmissions."""
    __tablename__ = "readings"
    __table_args__ = (UniqueConstraint("meter_pk", "series_hash"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    meter_pk: Mapped[int] = mapped_column(ForeignKey("meters.id"), index=True)
    first_day: Mapped[str] = mapped_column(String(10))
    values: Mapped[list] = mapped_column(JSON)          # kWh per day, null = missing
    series_hash: Mapped[str] = mapped_column(String(64))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    meter_pk: Mapped[int] = mapped_column(ForeignKey("meters.id"), index=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("readings.id"), unique=True)
    risk_score: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String(16))   # SUSPICIOUS | NORMAL
    threshold: Mapped[float] = mapped_column(Float)
    model_run: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    meter: Mapped[Meter] = relationship(back_populates="predictions")


class Alert(Base):
    """Raised when risk_score >= threshold; at most one per prediction (so one per submitted series)."""
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    meter_pk: Mapped[int] = mapped_column(ForeignKey("meters.id"), index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    risk_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN | REVIEWED | DISMISSED
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    meter: Mapped[Meter] = relationship()


def make_session_factory(database_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
