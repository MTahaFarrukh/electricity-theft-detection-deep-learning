"""Simulated Real-Time Monitoring: replay HELD-OUT TEST customers to the API, one meter per tick.

This is a simulation; it is not connected to real meters. Only test-split
customers are replayed (never training or validation data), and their labels
are never sent to the API.

    uvicorn backend.app.main:app                     # terminal 1
    python scripts/simulate_monitoring.py            # terminal 2
    python scripts/simulate_monitoring.py --interval 0.2 --limit 300
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import ID_COL, load_sgcc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default=os.environ.get("SIMULATOR_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--interval", type=float, default=float(os.environ.get("SIMULATOR_INTERVAL_SECONDS", 2)))
    parser.add_argument("--limit", type=int, default=200, help="Number of test meters to replay (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="Order in which test meters are replayed")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.api_url, timeout=60)
    try:
        health = client.get("/health").json()
    except httpx.HTTPError as exc:
        print(f"ERROR: API not reachable at {args.api_url} ({exc}). Start it with: uvicorn backend.app.main:app")
        return 1
    if not health.get("model_loaded"):
        print(f"ERROR: API has no model loaded: {health.get('error')}")
        return 1

    splits = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "splits.csv", dtype={ID_COL: str})
    test_ids = splits.loc[splits["split"] == "test", ID_COL].to_numpy()
    print("Loading raw readings for held-out test meters ...")
    raw = load_sgcc(PROJECT_ROOT / "data" / "raw" / "data.csv")
    if str(raw.dates[0].date()) != health["first_day"] or len(raw.dates) != health["n_days"]:
        print("ERROR: dataset dates do not match the deployed model's expected input")
        return 1
    index = {str(cid): i for i, cid in enumerate(raw.ids)}
    order = np.random.default_rng(args.seed).permutation(test_ids)
    if args.limit:
        order = order[:args.limit]

    print(f"Simulated Real-Time Monitoring: {len(order)} test meters, one every {args.interval}s. Ctrl+C to stop.")
    n_alerts = 0
    try:
        for k, meter_id in enumerate(order, 1):
            values = raw.readings[index[meter_id]]
            readings = [None if np.isnan(v) else round(float(v), 4) for v in values]
            r = client.post("/predict", json={"meter_id": meter_id, "readings": readings})
            r.raise_for_status()
            res = r.json()
            n_alerts += res["alert_raised"] and not res["duplicate"]
            flag = "  -> ALERT (investigation recommended)" if res["alert_raised"] else ""
            print(f"[{k:4d}/{len(order)}] meter {meter_id}: {res['prediction']:10s} risk {res['risk_score']:.3f}{flag}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")
    print(f"New alerts raised: {n_alerts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
