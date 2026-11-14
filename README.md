# AI-Based Electricity Theft Detection System Using CNN-LSTM

> Final Year Project. A decision-support system that flags **suspicious electricity-consumption
> patterns** in smart-meter data for investigation. A prediction is **not** proof of theft.

**Project status:** All phases done. The final, one-time test results are in section 12. The deployed model is the
CNN-LSTM (seed 1): test PR-AUC 0.487, recall 0.500, FPR 0.053.

---

## 1. Project overview
The system analyses time series of smart-meter consumption and gives each meter a **risk score**
(0â€“1) and a classification of **NORMAL** or **SUSPICIOUS**. The proposed model is a hybrid
**CNN-LSTM**. It is compared against Logistic Regression, Random Forest, a stand-alone 1D-CNN and a
stand-alone LSTM. The results are served through a FastAPI backend and a React dashboard, which
includes a clearly labelled **simulated** real-time monitoring mode.

## 2. Problem statement
Non-technical losses, including electricity theft, cost utilities a lot of revenue. Manual
inspection is expensive, so utilities need a way to prioritise which meters to inspect. Labelled
theft data is scarce and highly imbalanced, consumption is noisy and has missing readings, and a
false alert can cause an unjustified investigation of an honest customer.

## 3. Objectives
1. Audit and preprocess a public labelled smart-meter dataset without data leakage.
2. Establish baselines (Logistic Regression, Random Forest, plus one simple heuristic or linear baseline).
3. Implement and compare 1D-CNN, LSTM and CNN-LSTM classifiers.
4. Evaluate with imbalance-aware metrics and choose a decision threshold on the validation set.
5. Serve the selected model through a REST API that stores predictions and alerts.
6. Build a dashboard for monitoring, meter drill-down, alerts and model performance.
7. Document the methodology so that it can be explained in the viva.

## 4. System architecture (proposed)

```
            OFFLINE (training)                                   ONLINE (serving)
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ data/raw â”€â–º audit â”€â–º preprocessing   â”‚        â”‚  Simulator (scripts/) â”€â”€HTTPâ”€â”€â–º FastAPI       â”‚
â”‚   (customer-level split FIRST)       â”‚        â”‚                                  â”‚            â”‚
â”‚        â”‚                             â”‚        â”‚     inference service â—„â”€â”€ models/ + artifacts/â”‚
â”‚        â–¼                             â”‚ saved  â”‚            â”‚                                  â”‚
â”‚  EDA â”€â–º baselines â”€â–º CNN â”€â–º LSTM â”€â–º  â”‚ â”€â”€â”€â”€â”€â–º â”‚            â–¼                                  â”‚
â”‚        CNN-LSTM â”€â–º evaluation        â”‚ models â”‚     SQLite (meters, readings, predictions,    â”‚
â”‚        + threshold selection (val)   â”‚  +     â”‚             alerts)                           â”‚
â”‚                                      â”‚ scaler â”‚            â”‚                                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜        â”‚            â–¼  REST/JSON                       â”‚
                                                â”‚     React + Vite dashboard (Recharts)         â”‚
                                                â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Design principles: the stack is a **single backend process** with no microservices, **one SQLite
file** and **one inference module**, which the API, the tests and the evaluation scripts all share.
Training happens offline and **never** during API requests.

## 5. Dataset
See [data/README.md](data/README.md). Dataset: **SGCC** (State Grid Corporation of China).
Full audit: `reports/audit/dataset_audit.md`. Verified facts:

- 42,372 customers (rows), wide layout: `CONS_NO`, `FLAG`, then one column per day.
- 2014-01-01 to 2016-10-31: 1,034 date columns; **2016-09-18 is absent**. Columns are stored in string order, not chronological order.
- Labels are **per customer**: 3,615 theft-labelled (8.53%), 38,757 normal.
- No duplicate IDs, no non-numeric readings. 5 customers have no readings at all.
- 25.6% of readings are missing, mostly because ~16k customers' series start late. After a customer's first reading the median missing fraction is 1.4%. Mean missing fraction: 31.5% (theft) vs 25.1% (normal).
- Extreme outliers: max 800,003 kWh against a 99.9th percentile of 372.

## 6. Data preprocessing
Code: `src/preprocessing/pipeline.py`, run with `scripts/preprocess.py`.
1. Parse date columns, sort chronologically, and insert the missing day (2016-09-18) as missing â†’ 1,035 days.
2. Coerce invalid readings to missing; drop the 5 customers with no readings (uses no label information).
3. **Split customers 70/15/15 into train/validation/test, stratified by label, BEFORE anything is fitted.**
   Train 29,657 / val 6,355 / test 6,355 customers, each with an 8.5% positive rate. Seed from `RANDOM_SEED`.
4. Cap readings at the 99.9th percentile of the **training** readings (365.0).
5. Linearly interpolate interior gaps of â‰¤ 7 days, using only that customer's own series. Longer, leading and trailing gaps stay missing.
6. `log1p`, then standardise with mean/std fitted **on training readings only**. Remaining gaps are set to 0.
7. A **missingness mask** (1 = originally missing) is kept as a second channel, since missingness differs by class.
8. **Sequence design:** labels are per customer, so each customer is **one sample: the full 1,035-day series**.
   No windowing, so every sample's label really belongs to it.

Outputs: `data/processed/{train,val,test}.npz` (`values`, `mask`, `y`, `ids`), `data/processed/splits.csv`,
`artifacts/scalers/preprocessor.json` (fitted statistics) and `artifacts/metadata/preprocessing_summary.json`.

### EDA findings (training split only)
Run with `python scripts/eda.py`. Figures: `reports/figures/eda_*.png`, table: `reports/results/eda_summary.md`.
Validation and test customers are excluded, so EDA cannot leak held-out data into modelling decisions.

- **Theft-labelled customers consume more, not less**: median per-customer mean of 8.8 vs 5.5 kWh/day.
  Their consumption is also more variable (std 1.96Ã—, coefficient of variation 1.21Ã—). Theft shows up as
  irregular patterns, such as sudden long drops, rather than simply low usage.
- **Missingness is strongly associated with the label**: theft-labelled customers have 2.2Ã— the median
  missing fraction (0.21 vs 0.10). Before 2016 about 35â€“43% of readings are missing; from 2016-01-01 it is
  under 4% for both classes. This justifies the mask channel, and it is a **risk**: a model could partly
  learn data-collection artefacts. This is analysed in Phase 9 and listed under limitations.
- **Data-quality days**: 2014-10-03 is about 99% missing and 2016-09-18 is absent from the file.
- **Weekly profile**: weekend consumption is 1â€“2% above the weekly mean for both classes, a weak weekly cycle.
  This supports a weekly (7-day) structure in the CNN design.
- **Label noise is visible**: some normal-labelled customers read 0 kWh on every day.

## 7. CNN architecture
Code: `src/models/deep.py` (`build_cnn`). Input: **(1036, 2)**, i.e. 1,035 days padded with one missing day to make
148 whole weeks. Channel 0 is preprocessed consumption, channel 1 is the missingness mask.

```
Conv1D(32, k=7) â†’ BN â†’ ReLU â†’ MaxPool(2)      # first kernel spans one week (weekly cycle from EDA)
Conv1D(64, k=5) â†’ BN â†’ ReLU â†’ MaxPool(2)
Conv1D(64, k=3) â†’ BN â†’ ReLU â†’ MaxPool(2)
[GlobalAvgPool â€– GlobalMaxPool] â†’ Dropout(0.3) â†’ Dense(32, ReLU) â†’ Dropout(0.3) â†’ Dense(1, sigmoid)
```
27,777 parameters. Trained with `python scripts/train_deep.py --model cnn` (shared trainer for all deep
models: BCE loss, Adam 1e-3, batch 128, balanced class weights, ReduceLROnPlateau, early stopping on validation
PR-AUC with patience 6, seed 42). The best validation PR-AUC came at epoch 39 of 40
(28 min on CPU). The model was still improving slowly when it hit the 40-epoch limit.

## 8. LSTM architecture
Code: `src/models/deep.py` (`build_lstm`). It takes the same (1036, 2) input, which is reshaped **inside the model** to
**148 weekly steps Ã— 14 features** (7 days Ã— [value, mask]). Running an LSTM over 1,036 daily steps is very slow on a
CPU, and recurrent nets struggle with dependencies that long. Weekly steps keep the weekly structure found in the EDA.

```
Reshape(148, 14) â†’ LSTM(64, return_sequences) â†’ LSTM(32) â†’ Dropout(0.3) â†’ Dense(32, ReLU) â†’ Dropout(0.3) â†’ Dense(1, sigmoid)
```
33,729 parameters. Same trainer and settings as the CNN (`python scripts/train_deep.py --model lstm`).
The best validation PR-AUC came at epoch 38 of 40 (23 min on CPU).
The stand-alone LSTM is **weaker than both the CNN and the Random Forest** (validation PR-AUC 0.359).
Without convolutional feature extraction it has to learn local patterns, such as sudden drops, from raw weekly
inputs. This motivates the hybrid model.

## 9. CNN-LSTM architecture (proposed model)
Code: `src/models/deep.py` (`build_cnn_lstm`). It uses the same convolutional blocks as the CNN. These extract local
patterns and shorten the daily sequence 8Ã— (1,036 â†’ 129 steps of 64 learned features). LSTM layers then model how
those patterns evolve over the whole period, in place of the CNN's order-blind global pooling.

```
Conv1D(32, k=7) â†’ BN â†’ ReLU â†’ MaxPool(2)
Conv1D(64, k=5) â†’ BN â†’ ReLU â†’ MaxPool(2)
Conv1D(64, k=3) â†’ BN â†’ ReLU â†’ MaxPool(2)      # (129, 64)
LSTM(64, return_sequences) â†’ LSTM(32) â†’ Dropout(0.3) â†’ Dense(32, ReLU) â†’ Dropout(0.3) â†’ Dense(1, sigmoid)
```
70,145 parameters. Same trainer and settings as the other deep models (`python scripts/train_deep.py --model cnn_lstm`).
Early stopping ended training at epoch 36; the best epoch was 30 (83 min on CPU).

**Validation comparison with the CNN (honest reading):** the CNN-LSTM has the **highest PR-AUC** of all models
(0.478 vs 0.453). PR-AUC is the primary, imbalance-aware selection metric. It also has the highest recall
(0.548 vs 0.520). The CNN, however, is better on ROC-AUC (0.849 vs 0.832), F1
(0.504 vs 0.481) and FPR (0.051 vs 0.068). Each figure comes from one training run with one seed,
so the gap may lie within run-to-run variation. This is checked in Phase 9 before any claim is made on the test set.

## 10. Training procedure
Binary cross-entropy, the Adam optimiser, **class weights** for imbalance (no SMOTE on raw
sequences), early stopping on validation PR-AUC, model checkpointing and a fixed random seed.
All deep models share one trainer (`scripts/train_deep.py`). Settings: batch 128, Adam learning rate 1e-3, at most 40
epochs, early-stopping patience 6 with the best weights restored, ReduceLROnPlateau (factor 0.5, patience 3),
balanced class weights {0: 0.55, 1: 5.87} and seed 42. Extra seeds are added with `--seed N --run-name seedN`.
Baselines: Logistic Regression (C in {0.01, 0.1, 1}, chosen on validation PR-AUC) and Random Forest (400 trees,
min_samples_leaf 5, balanced_subsample), both on standardised hand-crafted features (`scripts/train_baselines.py`).
Training happens offline only, on a CPU (Windows). Per-model times are given in sections 7â€“9.

## 11. Evaluation methodology
- Metrics: accuracy, precision, recall, F1, PR-AUC, ROC-AUC, false-positive rate and the confusion matrix.
- The **test set is used once**, for the final evaluation. Model selection and thresholds use validation data only.
- The threshold is chosen on the validation set from the precision-recall trade-off and stored in `artifacts/metadata/`.
- False-positive and false-negative analysis for the deployed model.

**Model-selection rule (fixed before the test set was used).** `scripts/evaluate_test.py` applies this rule:
1. The CNN and CNN-LSTM, the two strongest models on validation, are each trained with **three seeds** (42, 1, 2).
   One seed cannot separate a real gap from run-to-run noise. Baselines are deterministic given the seed and are trained once.
2. **Deployed architecture** = the highest **mean validation PR-AUC** across its runs.
   **Deployed weights** = that architecture's run with the highest validation PR-AUC. Its validation-chosen threshold
   is written to `artifacts/metadata/model.json` *before* any test score is computed.
3. Every trained run is then scored **once** on test with its own validation threshold. Architectures are reported as
   mean Â± std over seeds. Nothing is retrained or re-selected after the test scores are seen, and the script refuses to run twice.
4. **Missingness robustness:** test metrics are also reported for customers with â‰¤ 5% missing readings. This checks
   whether performance depends on missing-data artefacts (see the EDA).
5. Before testing, the script checks that `src/inference/` (raw readings â†’ `Predictor`) reproduces the
   evaluation scores on validation customers, so the API serves exactly the evaluated model.

## 12. Results

### Validation results (model development; NOT the final test results)
Baselines use 56 hand-crafted per-customer features (`src/features/summary.py`) and class weights. Thresholds
maximise F1 on validation. The validation positive rate is 8.5%, so a random classifier has a PR-AUC of about 0.085.
Sources: `reports/results/baselines_validation.json`, `reports/results/deep_validation.json`.

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Missingness-only LR (heuristic) | 0.803 | 0.186 | 0.387 | 0.251 | 0.170 | 0.665 | 0.158 |
| Logistic Regression | 0.865 | 0.298 | 0.426 | 0.351 | 0.292 | 0.773 | 0.094 |
| Random Forest | 0.904 | 0.440 | 0.448 | 0.444 | 0.418 | 0.828 | 0.053 |
| 1D-CNN | 0.913 | 0.488 | 0.520 | 0.504 | 0.453 | 0.849 | 0.051 |
| LSTM | 0.885 | 0.361 | 0.450 | 0.401 | 0.359 | 0.795 | 0.074 |
| **CNN-LSTM** | 0.899 | 0.428 | 0.548 | 0.481 | 0.478 | 0.832 | 0.068 |

The missingness-only model (PR-AUC 0.170, twice the random level) shows that **missing-data patterns alone
carry some label signal**. The full-feature models are well above it, so they are learning more than
data-collection artefacts. The Random Forest's top features are `max`, `monthly_std`, `trend_slope` and `std`:
variability and trend, which matches the EDA.

**Extra seeds (validation PR-AUC).** 1D-CNN: 0.453 / 0.448 / 0.444 (seeds 42 / 1 / 2), mean 0.448 Â± 0.004.
CNN-LSTM: 0.478 / 0.495 / 0.461, mean 0.478 Â± 0.017. Under the pre-registered rule, the **CNN-LSTM** was selected
and its best run (**seed 1**, validation PR-AUC 0.495, threshold 0.764) was deployed. This was recorded in
`artifacts/metadata/model.json` before any test score was computed.

### Final test results
Test split: 6,355 customers, 8.5% positive (random PR-AUC â‰ˆ 0.085). Each model uses its own validation threshold.
CNN and CNN-LSTM are shown as **mean Â± std over 3 seeds**; the other models were trained once.
Source: `reports/results/test_results.json`; figures `reports/figures/test_pr_curves.png`, `test_confusion_matrix.png`.

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Missingness-only LR (heuristic) | 0.795 | 0.171 | 0.365 | 0.233 | 0.180 | 0.677 | 0.165 |
| Logistic Regression | 0.861 | 0.274 | 0.382 | 0.319 | 0.284 | 0.774 | 0.094 |
| Random Forest | 0.900 | 0.411 | 0.391 | 0.401 | 0.394 | 0.828 | 0.052 |
| 1D-CNN (3 seeds) | 0.901 Â± 0.007 | 0.432 Â± 0.030 | 0.488 Â± 0.013 | 0.458 Â± 0.012 | 0.459 Â± 0.011 | **0.847 Â± 0.007** | **0.060 Â± 0.009** |
| LSTM | 0.882 | 0.335 | 0.395 | 0.362 | 0.337 | 0.795 | 0.073 |
| **CNN-LSTM (3 seeds)** | 0.899 Â± 0.009 | 0.427 Â± 0.037 | **0.516 Â± 0.015** | **0.466 Â± 0.017** | **0.475 Â± 0.016** | 0.834 Â± 0.005 | 0.065 Â± 0.011 |
| *Deployed: CNN-LSTM seed 1* | 0.909 | 0.469 | 0.500 | 0.484 | 0.487 | 0.838 | 0.053 |

Deployed model on test: TP 271, FP 307, FN 271, TN 5,506. About half of the theft-labelled customers are flagged,
and roughly 1 in 2 flagged customers is theft-labelled (versus 1 in 12 at random).

**Interpretation.**
- The deep models clearly beat the baselines: the CNN-LSTM reaches a PR-AUC of 0.475, about 5.6Ã— the random level,
  compared with 0.394 for the Random Forest and 0.284 for Logistic Regression.
- **CNN-LSTM vs 1D-CNN:** the CNN-LSTM has the higher mean PR-AUC, recall and F1. The CNN has the higher ROC-AUC and
  the lower FPR. The PR-AUC gap (0.016) is about one standard deviation across seeds, so with three seeds **the two
  architectures are not clearly separated**. We do not claim the hybrid is significantly better. Its practical
  advantage is higher recall (more suspicious customers found) at a similar false-positive rate.
- The stand-alone LSTM is the weakest deep model, which supports using convolutional feature extraction before the LSTM.
- Validation and test numbers agree closely, which suggests no overfitting to the validation split.

**Missingness robustness: customers with â‰¤ 5% missing readings** (n = 2,756, 5.8% positive, random PR-AUC â‰ˆ 0.058).

| Model | PR-AUC | ROC-AUC | Recall | FPR |
|---|---|---|---|---|
| Missingness-only LR | 0.132 | 0.679 | 0.138 | 0.017 |
| Logistic Regression | 0.199 | 0.785 | 0.319 | 0.083 |
| Random Forest | 0.294 | 0.837 | 0.306 | 0.040 |
| 1D-CNN (3 seeds) | 0.442 Â± 0.005 | 0.875 Â± 0.011 | 0.458 Â± 0.019 | 0.045 Â± 0.008 |
| LSTM | 0.315 | 0.809 | 0.369 | 0.057 |
| CNN-LSTM (3 seeds) | 0.440 Â± 0.016 | 0.840 Â± 0.007 | 0.525 Â± 0.027 | 0.053 Â± 0.008 |

On nearly complete series, the deep models keep a PR-AUC of about 0.44, roughly 7.6Ã— the random level of this subset,
and their ROC-AUC rises. The missingness-only model falls to 0.132. **So the deep models' performance does not rely
mainly on missing-data artefacts.** The Random Forest loses more on this subset (0.394 â†’ 0.294), which suggests its
hand-crafted features depend more on missingness.

**Error analysis (deployed model, test).** Medians per group (`reports/results/test_predictions_deployed.csv`):

| Group | n | Missing fraction | Mean consumption (log-standardised) | Risk score |
|---|---|---|---|---|
| True positives | 271 | 0.224 | 0.505 | 0.963 |
| False positives | 307 | 0.177 | 0.274 | 0.885 |
| False negatives | 271 | 0.243 | 0.154 | 0.290 |
| True negatives | 5,506 | 0.088 | 0.066 | 0.054 |

False alarms have twice the missing data of true negatives and higher consumption: the model partly treats "large and
gappy" series as risky. Missed cases consume less than the detected ones, so low-consumption theft is harder to find.
Some false positives may also be undetected theft, since the labels come from inspections (label noise).

*Process note:* the first run of `evaluate_test.py` computed all test metrics, then crashed on a dtype bug in the
error-analysis CSV step before saving (log kept as `reports/results/evaluate_test_crashed_run.log`). After a one-line
fix, the script was run again with the same models, thresholds and selection. Every number is identical to the crashed run.
No model, threshold or selection was changed.

## 13. Backend API
FastAPI (`backend/app/`): one process, one SQLite file (`DATABASE_URL`), with tables `meters`, `readings`, `predictions` and
`alerts`. All scoring goes through `src/inference/predictor.py`, which loads the deployed model and threshold from
`artifacts/metadata/model.json` and applies the training-fitted preprocessing. The API never trains.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check; reports whether the model is loaded |
| POST | `/predict` | Score one consumption sequence |
| POST | `/predict/batch` | Score several sequences |
| GET | `/meters` | List monitored meters with their latest status |
| GET | `/meters/{meter_id}` | Meter detail: readings, prediction history, alerts |
| GET | `/alerts` | Alert history (filter: `status`, `min_risk`) |
| PATCH | `/alerts/{id}` | Set status: OPEN / REVIEWED / DISMISSED |
| GET | `/dashboard/summary` | Counts and distributions for the dashboard |
| GET | `/model/metrics` | Stored evaluation results for the Model Performance page |

`POST /predict` takes `{"meter_id": "optional", "readings": [kWh or null, ...]}`, i.e. 1,035 daily values starting on
2014-01-01, the model's input window. With a `meter_id`, the reading, prediction and (if `risk_score >= threshold`) an
alert are stored. **Resubmitting an identical series returns the stored result with `"duplicate": true` and raises no
new alert.** Response:
```json
{"meter_id": "M1", "prediction": "SUSPICIOUS", "risk_score": 0.76, "threshold": 0.31,
 "recommendation": "Investigation recommended: high-risk consumption pattern.", "alert_raised": true, "duplicate": false}
```
(The threshold value is illustrative; the real one comes from validation.) Interactive docs: http://127.0.0.1:8000/docs.
Tests: `backend/tests/test_api.py` (fake model, temporary database).

## 14. Frontend
React + Vite + Recharts (`frontend/`). Pages: Dashboard, Meters, Meter Details, Alerts, Model Performance and
About/Methodology. Every number shown comes from the backend. Suspicious meters display
"Investigation Recommended", never "theft confirmed". Status is shown with an icon and a label, never colour alone.
Light and dark mode are supported. The dashboard refreshes every 5 s while the simulator runs.

## 15. Simulated real-time monitoring
`scripts/simulate_monitoring.py` replays **held-out test-set** sequences to `/predict` at a configurable interval. The
backend stores each prediction and raises an alert when the risk score is at or above the
validated threshold, without creating duplicates for the same reading. The UI labels this
feature **"Simulated Real-Time Monitoring"**. It is not connected to real meters. Only test-split customers are replayed,
and their labels are never sent to the API.

## 16. Installation
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                  # macOS/Linux: cp .env.example .env
```
Frontend: Node.js 18+ (`cd frontend && npm install`).
Python 3.11 or 3.12 is required for TensorFlow. On native Windows, TensorFlow runs on the CPU
only. For GPU training, use Google Colab or WSL2.

## 17. How to train
Full offline pipeline, in order (total about 6â€“7 h on a CPU):
```bash
python scripts/extract_sgcc.py                 # data/raw/sgcc_zip/* -> data/raw/data.csv
python scripts/audit_dataset.py data/raw/data.csv
python scripts/preprocess.py                   # -> data/processed/, artifacts/
python scripts/eda.py                          # -> reports/figures/, reports/results/eda_summary.md
python scripts/train_baselines.py              # -> models/baselines/, reports/results/baselines_validation.json
python scripts/train_deep.py --model cnn        # -> models/cnn/, reports/results/deep_validation.json (~30 min CPU)
python scripts/train_deep.py --model lstm       # -> models/lstm/ (~25 min CPU)
python scripts/train_deep.py --model cnn_lstm   # -> models/cnn_lstm/ (~85 min CPU)
python scripts/train_deep.py --model cnn --seed 1 --run-name seed1        # extra seeds (also seed 2,
python scripts/train_deep.py --model cnn_lstm --seed 1 --run-name seed1   #   for both models)
python scripts/evaluate_test.py                # ONE-TIME test evaluation -> artifacts/metadata/model.json,
                                               #   reports/results/test_results.json, reports/figures/test_*.png
```

## 18. How to run the backend
Requires `artifacts/metadata/model.json`, written by `scripts/evaluate_test.py`.
```bash
uvicorn backend.app.main:app --reload              # from the project root -> http://127.0.0.1:8000/docs
python scripts/simulate_monitoring.py              # second terminal: Simulated Real-Time Monitoring
python scripts/simulate_monitoring.py --interval 0.2 --limit 300   # faster / more meters
```

## 19. How to run the frontend
```bash
cd frontend
npm install
npm run dev                                        # -> http://localhost:5173 (expects the API on :8000)
```
To use another API address, set `VITE_API_BASE_URL` in `frontend/.env` (see `frontend/.env.example`).

## 20. Limitations
- Labels come from the utility's own inspections. They may be noisy, and undetected theft may be labelled normal.
- A model trained on one region or utility may not generalise to others.
- The real-time component is a simulation.
- The output is a risk indicator for prioritising inspections, not evidence of wrongdoing.
- Missing-data patterns are associated with the label (EDA; missingness-only baseline PR-AUC 0.170 vs 0.085 random).
  Part of any model's signal may reflect data collection rather than consumption behaviour. Section 12 reports
  test performance on low-missingness customers to measure this.
- The model needs the full 1,035-day window (2014-01-01 to 2016-10-31) to score a meter. It cannot score a new
  meter with a short history, or a later period, without retraining.
- Deep models were trained on a CPU with a small hyper-parameter search, so better configurations may exist.

## 21. Future work
- Score rolling windows (e.g. the last 6â€“12 months) so new meters and current periods can be monitored, retraining periodically.
- A broader hyper-parameter search and attention layers, trained on a GPU.
- Probability calibration, and choosing the threshold from inspection cost and capacity rather than F1.
- Explanations for each alert (e.g. which weeks drove the score) to help inspectors.
- Validation on data from another utility; feedback from inspection outcomes (alert status) as new labels.

---

## Project structure
```
â”œâ”€â”€ data/            raw/ (not committed), processed/, README.md
â”œâ”€â”€ src/             data/, preprocessing/, features/, models/, training/, evaluation/, inference/
â”œâ”€â”€ models/          baselines/, cnn/, lstm/, cnn_lstm/   (trained model files, not committed)
â”œâ”€â”€ artifacts/       scalers/, encoders/, metadata/        (preprocessing + threshold metadata)
â”œâ”€â”€ reports/         audit/, figures/, results/
â”œâ”€â”€ backend/         app/{api,schemas,services,models,core}, tests/
â”œâ”€â”€ frontend/        React + Vite + Recharts dashboard
â”œâ”€â”€ scripts/         CLI entry points (audit, training, simulator)
â”œâ”€â”€ tests/           tests for the ML pipeline
â””â”€â”€ docs/            viva Q&A
```

## Development phases
| # | Phase | Status |
|---|---|---|
| 1 | Inspect workspace and dataset | Done (workspace was empty; no dataset) |
| 2 | Project structure and dependencies | Done |
| 3 | Data audit and preprocessing | Done |
| 4 | EDA | Done |
| 5 | Baseline models | Done |
| 6 | CNN | Done |
| 7 | LSTM | Done |
| 8 | CNN-LSTM | Done |
| 9 | Evaluation and comparison | Done |
| 10 | Inference pipeline | Done |
| 11 | FastAPI backend | Done |
| 12 | Database and alerts | Done |
| 13 | React dashboard | Done |
| 14 | Simulated real-time monitoring | Done |
| 15 | Testing, documentation | Done |
