"""In-memory simulation controller allowing real-time meter inspection demos from the UI."""

from __future__ import annotations

import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import delete

from backend.app.models.db import Alert, Meter, Prediction, Reading
from backend.app.services.scoring import score_and_store
from src.preprocessing.pipeline import ID_COL, load_sgcc

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SimulationController:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.state = {
            "running": False,
            "current": 0,
            "total": 0,
            "alerts_raised": 0,
            "last_meter_id": None,
            "message": "Idle",
        }

    def status(self) -> dict:
        with self._lock:
            return dict(self.state)

    def stop(self) -> dict:
        self._stop_event.set()
        with self._lock:
            self.state["running"] = False
            self.state["message"] = "Stopped"
            return dict(self.state)

    def reset_data(self, session_factory) -> dict:
        self.stop()
        with session_factory() as session:
            session.execute(delete(Alert))
            session.execute(delete(Prediction))
            session.execute(delete(Reading))
            session.execute(delete(Meter))
            session.commit()
        with self._lock:
            self.state = {
                "running": False,
                "current": 0,
                "total": 0,
                "alerts_raised": 0,
                "last_meter_id": None,
                "message": "Demo data reset successfully",
            }
            return dict(self.state)

    def start(self, session_factory, predictor, limit: int = 50, interval: float = 1.0, seed: int = 42) -> dict:
        with self._lock:
            if self.state["running"]:
                return dict(self.state)
            self._stop_event.clear()
            self.state = {
                "running": True,
                "current": 0,
                "total": limit,
                "alerts_raised": 0,
                "last_meter_id": None,
                "message": "Starting live inspection...",
            }

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(session_factory, predictor, limit, interval, seed),
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def _run_loop(self, session_factory, predictor, limit: int, interval: float, seed: int):
        raw_path = PROJECT_ROOT / "data" / "raw" / "data.csv"
        splits_path = PROJECT_ROOT / "data" / "processed" / "splits.csv"
        n_days = predictor.n_days

        use_csv = splits_path.exists() and raw_path.exists()
        if use_csv:
            splits = pd.read_csv(splits_path, dtype={ID_COL: str})
            test_ids = splits.loc[splits["split"] == "test", ID_COL].to_numpy()
            raw = load_sgcc(raw_path)
            index = {str(cid): i for i, cid in enumerate(raw.ids)}
            order = np.random.default_rng(seed).permutation(test_ids)[:limit]
            get_readings = lambda cid: [None if np.isnan(v) else round(float(v), 4) for v in raw.readings[index[cid]]]
        else:
            rng = np.random.default_rng(seed)
            order = [f"METER_{i+1:04d}" for i in range(limit)]

            def generate_synthetic(cid):
                idx = int(cid.split("_")[1])
                is_theft_drop = (idx % 10 == 3)
                is_theft_tamper = (idx % 10 == 7)
                base = rng.uniform(4.0, 9.0, size=n_days) + np.sin(np.linspace(0, 20, n_days)) * 1.5
                if is_theft_drop:
                    base[550:] = base[550:] * rng.uniform(0.05, 0.20)
                    mask = rng.random(n_days) < 0.20
                elif is_theft_tamper:
                    base[450:] = base[450:] * rng.uniform(0.10, 0.35)
                    mask = rng.random(n_days) < 0.28
                else:
                    mask = rng.random(n_days) < 0.03
                return [None if m else round(float(max(0.05, v)), 4) for v, m in zip(base, mask)]

            get_readings = generate_synthetic

        for k, meter_id in enumerate(order, 1):
            if self._stop_event.is_set():
                break

            readings = get_readings(meter_id)
            with session_factory() as session:
                res = score_and_store(session, predictor, meter_id, readings)

            with self._lock:
                self.state["current"] = k
                self.state["last_meter_id"] = meter_id
                if res.get("alert_raised") and not res.get("duplicate"):
                    self.state["alerts_raised"] += 1
                self.state["message"] = f"Inspecting meter {meter_id} ({k}/{limit})..."

            time.sleep(interval)

        with self._lock:
            self.state["running"] = False
            self.state["message"] = f"Completed inspection of {self.state['current']} meters. Alerts raised: {self.state['alerts_raised']}."


simulation_controller = SimulationController()
