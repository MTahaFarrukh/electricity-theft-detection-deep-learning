# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Final Year Project: electricity-theft detection on smart-meter data (SGCC dataset) with a hybrid
CNN-LSTM, compared against Logistic Regression, Random Forest, 1D-CNN and LSTM. Results are served
through a FastAPI backend with SQLite and shown on a React + Vite + Recharts dashboard. Work
proceeds in numbered phases; the phase table at the end of README.md is the source of truth for
what is done. Update that table and the `[TO BE FILLED ...]` placeholders as phases complete.

Implemented: the offline pipeline below, `src/inference/predictor.py`, the FastAPI backend, the React
dashboard and the simulator, all with tests. `notebooks/` and `src/training/` are unused placeholders
(training lives in `scripts/`). The one-time test evaluation is `scripts/evaluate_test.py`. It refuses
to run twice; don't use `--force` to re-select models.

Data pipeline: `extract_sgcc.py` → `audit_dataset.py data/raw/data.csv` → `preprocess.py` →
`eda.py` (training split only; figures use the colour-blind-safe pair blue `#2a78d6` = normal,
orange `#eb6834` = theft-labelled — keep these for later figures) → `train_baselines.py` →
`train_deep.py --model cnn|lstm|cnn_lstm` (shared trainer; add new architectures to `BUILDERS` in
`src/models/deep.py`; input via `to_model_input`, shape (1036, 2); ~30 min per model on CPU, so run it
in the background and log to `models/<model>/train.log`).

Every model is scored with `src/evaluation/metrics.py` (`select_threshold` on validation only,
`compute_metrics`). Baseline features come from `src/features/summary.py`; `miss_*` columns are kept
separate so the missingness-only baseline can measure how much signal comes from missing-data artefacts.
Validation results go in README section 12 "Validation results"; the "Final test results" table is
filled only in Phase 9. Processed
splits are `data/processed/{train,val,test}.npz` with `values` (n, 1035) float32, `mask` (n, 1035)
uint8 (1 = originally missing), `y`, `ids`. One sample = one customer's full daily series (labels are
per customer). The fitted `Preprocessor` (`artifacts/scalers/preprocessor.json`) must also be used
by `src/inference/` so serving matches training.

## Commands

Windows / PowerShell. Python 3.12 venv at `.venv` (TensorFlow does not support newer Pythons; a
Python 3.15 is also installed on this machine, so do not use the bare `py` launcher).

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

pytest                                   # all tests (tests/ and backend/tests/)
pytest tests/test_audit.py::test_long_layout_detection   # single test

python scripts/extract_sgcc.py           # joins data/raw/sgcc_zip/data.z01+z02+zip -> data/raw/
python scripts/audit_dataset.py data/raw/<file>.csv [--id-col X --label-col Y --layout wide|long]
```

`pytest.ini` sets `pythonpath = .`, so imports are `from src.... import ...`. Scripts in `scripts/`
insert the project root into `sys.path` themselves and must be run from the project root.

```powershell
uvicorn backend.app.main:app --reload             # needs artifacts/metadata/model.json (from evaluate_test.py)
python scripts/simulate_monitoring.py             # replays test meters to the API
cd frontend; npm install; npm run dev             # http://localhost:5173 ; `npm run build` to check it compiles
```
Backend tests use a fake predictor via `create_app(settings, model_service)`. For a smoke test against a real
model without touching `model.json`, point the `MODEL_METADATA_PATH`, `DATABASE_URL` and `RESULTS_DIR` env vars
at the scratchpad.

## Non-negotiable methodology rules

These are what the viva will probe; violating them invalidates the results.

- **Audit before assuming.** Never hard-code SGCC column names, date ranges or label semantics.
  Preprocessing decisions are made from `reports/audit/dataset_audit.md`. The audit code detects
  columns heuristically and records every detection in `notes`; keep that pattern.
- **Split customers into train/val/test (stratified by label) before any windowing, scaling,
  imputation or outlier capping.** No customer appears in more than one split. Every fitted
  statistic (scaler, percentile caps) is fitted on train only and saved to `artifacts/`.
- **Test set is used exactly once**, for final evaluation. Model selection, early stopping
  (on validation PR-AUC) and the decision threshold all use validation data. The chosen threshold
  goes to `artifacts/metadata/*.json` (committed); the API reads it from there.
- Imbalance is handled with class weights, not SMOTE on raw sequences. Fixed random seed
  (`RANDOM_SEED` in `.env`).
- Decided after the audit: SGCC labels are per customer, so each sample is one customer's full
  series (no windowing). Don't reintroduce windows without aggregating back to one prediction per customer.
- Output wording: a positive prediction is "SUSPICIOUS" / "Investigation Recommended", never
  "theft" or "theft confirmed". The real-time feature must be labelled "Simulated Real-Time
  Monitoring" and replays held-out test sequences only.
- Results tables in README must stay as placeholders until real numbers exist. Never invent metrics.

## Architecture

- Offline: `src/{data,preprocessing,features,models,evaluation}` pipeline driven by CLI scripts in `scripts/`.
  Trained models go to `models/<baselines|cnn|lstm|cnn_lstm>/`, fitted preprocessing to
  `artifacts/`. Training never happens inside an API request.
- `src/inference/` is the **single** inference module, shared by the API, tests and evaluation
  scripts, so preprocessing at serving time is identical to training.
- Online: one FastAPI process (`backend/app/{api,schemas,services,models,core}`), one SQLite file
  (`DATABASE_URL`) with meters, readings, predictions and alerts. Alerts are raised when
  `risk_score >= threshold`, without duplicates for the same reading. No microservices.
  Planned endpoints are listed in README section 13.
- Frontend shows only numbers that come from the backend.

Config lives in `.env` (see `.env.example`). `data/raw`, `data/processed`, model binaries,
`.joblib`/`.npy` artifacts and `*.db` are gitignored.
