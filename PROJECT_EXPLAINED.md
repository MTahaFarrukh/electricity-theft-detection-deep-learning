# ⚡ Electricity Theft Detection System — Complete Project Guide
> **Who this is for:** Someone who has never heard of deep learning, neural networks, or machine learning before. No prior knowledge required.

---

## Table of Contents
1. [What is the problem?](#1-what-is-the-problem)
2. [What does this system do?](#2-what-does-this-system-do)
3. [The Dataset — What data do we use?](#3-the-dataset--what-data-do-we-use)
4. [Data Cleaning — Making raw data usable](#4-data-cleaning--making-raw-data-usable)
5. [Understanding the AI Models](#5-understanding-the-ai-models)
   - [What is a "model"?](#what-is-a-model)
   - [Baseline models (the simple ones)](#baseline-models-the-simple-ones)
   - [1D-CNN — the pattern spotter](#1d-cnn--the-pattern-spotter)
   - [LSTM — the memory keeper](#lstm--the-memory-keeper)
   - [CNN-LSTM — the best of both worlds](#cnn-lstm--the-best-of-both-worlds)
6. [How we train the models](#6-how-we-train-the-models)
7. [How we evaluate performance](#7-how-we-evaluate-performance)
8. [The Results](#8-the-results)
9. [The Backend API](#9-the-backend-api)
10. [The Database](#10-the-database)
11. [The React Dashboard](#11-the-react-dashboard)
12. [The Simulated Real-Time Monitor](#12-the-simulated-real-time-monitor)
13. [How Everything Connects](#13-how-everything-connects)
14. [Project Folder Structure](#14-project-folder-structure)
15. [Limitations and Honest Caveats](#15-limitations-and-honest-caveats)

---

## 1. What is the problem?

Electricity companies (utilities) supply power to millions of homes and businesses through **smart meters** — digital devices that record how much electricity each customer uses every day.

**Electricity theft** happens when someone tampers with their meter or illegally taps into power lines to avoid paying their full bill. This is called a "non-technical loss" and costs utilities enormous amounts of money globally.

The old way to find thieves: send inspectors to check meters one by one. This is:
- **Very slow** — there are tens of thousands of meters
- **Very expensive** — each physical inspection costs money
- **Largely random** — inspectors don't know which meters to check first

**The goal of this project:** Use AI to automatically analyze daily electricity usage patterns and produce a **risk score** (0 to 1) for each meter. High score = more likely to be suspicious. This turns a random inspection schedule into a **prioritized queue** — inspectors check the highest-risk meters first.

> **IMPORTANT:** A high risk score means **"this meter is worth inspecting"**, NOT **"this person is definitely stealing electricity"**. The system is a decision-support tool, not a judge.

---

## 2. What does this system do?

```
Smart Meter Data ──► AI Model ──► Risk Score (0.0 to 1.0) ──► NORMAL or SUSPICIOUS
                                                              └──► If SUSPICIOUS: raise ALERT
```

**In plain English:**
1. We have ~3 years of daily electricity readings for 42,372 customers
2. Some customers are **labeled as theft** (found by past inspections)
3. We train an AI to learn the **pattern difference** between normal and theft customers
4. We deploy that AI as a web service
5. When a meter's readings are submitted, the AI returns a score and a recommendation
6. A dashboard lets utility staff monitor all meters, see alerts, and mark them reviewed or dismissed

---

## 3. The Dataset — What data do we use?

**Dataset:** SGCC (State Grid Corporation of China) — a publicly available, real-world dataset.

| Property | Value |
|---|---|
| Total customers | 42,372 |
| Date range | 2014-01-01 to 2016-10-31 |
| Days of data per customer | 1,035 days |
| Theft-labeled customers | 3,615 (8.53%) |
| Normal customers | 38,757 (91.47%) |

**What the raw data looks like:**

Imagine a giant spreadsheet:
- Each **row** = one customer
- **Columns 1 & 2** = Customer ID, Label (0=normal, 1=theft)
- **Columns 3 onwards** = one column per day, containing kWh (kilowatt-hours) used that day

```
CustomerID | Label | 2014/1/1 | 2014/1/2 | 2014/1/3 | ...
-----------|-------|----------|----------|----------| ---
   C0001   |   0   |   5.2    |   4.8    |   6.1    | ...  ← Normal
   C0002   |   1   |  12.3    |   0.0    |   0.0    | ...  ← Suspicious (sudden drop)
   C0003   |   0   |   3.1    |   NaN    |   3.5    | ...  ← Normal (one missing day)
```

**Key observations from exploring the data (EDA):**
- **Theft customers actually consume MORE on average** (8.8 vs 5.5 kWh/day) — theft often shows up as irregular spikes and drops, not just low consumption
- **Missing readings are strongly associated with theft** — theft-labeled customers have 2.2× more missing readings. This makes sense: a tampered meter may not transmit properly
- One day (2016-09-18) is completely absent from the dataset — a data-collection gap
- Some extreme outliers exist (e.g., 800,003 kWh in one day — clearly an error)

---

## 4. Data Cleaning — Making raw data usable

Raw data is messy. We clean it in several carefully ordered steps (`src/preprocessing/pipeline.py`):

### Step 1: Fix the date columns
The date columns are stored in alphabetical (string) order: `2014/1/1, 2014/1/10, 2014/1/11...` instead of chronological order. We re-sort them and insert the missing day (2016-09-18) as a "missing" reading. Result: **1,035 properly ordered days**.

### Step 2: Drop useless rows
5 customers have **zero readings** across all 1,035 days. We drop them — they carry no information.

### Step 3: Split the data FIRST ⚠️
> **CRITICAL STEP:** We split all 42,372 customers into three groups **before doing any statistics or fitting**:
> - **Training set (70%):** 29,657 customers — used to teach the AI
> - **Validation set (15%):** 6,355 customers — used to tune the AI and pick a decision threshold
> - **Test set (15%):** 6,355 customers — used ONCE at the very end to measure final performance
>
> Why? If we computed statistics on all data first and then split, the model would "see" the test data during training — this is called **data leakage** and would make our results artificially good and dishonest.

### Step 4: Cap extreme outliers
Using only training data: find the 99.9th percentile value (365 kWh/day). Any reading above this is capped at 365. This prevents extreme errors from distorting the model.

### Step 5: Fill short gaps
If a customer has 1–7 consecutive missing days **in the middle** of their series, we fill them with a straight line between the surrounding values (linear interpolation). Long gaps and gaps at the start/end stay missing.

### Step 6: Log-transform and standardize
- Apply `log(1 + x)` to every reading — this squishes the big numbers and makes the distribution more uniform
- Subtract the training mean and divide by the training standard deviation — this centers all values around 0

### Step 7: Create a "missingness mask"
For every customer, we create a **second series** of 1s and 0s:
- `1` = this day's reading was originally missing
- `0` = this day had a real reading

This becomes a second input **channel** to the AI — because missing patterns differ between normal and theft customers.

### Step 8: Save the processed data
Output files:
- `data/processed/train.npz`, `val.npz`, `test.npz` — the cleaned data arrays
- `artifacts/scalers/preprocessor.json` — the fitted statistics (cap value, mean, std) saved for later use

---

## 5. Understanding the AI Models

### What is a "model"?

A **model** is a mathematical function that takes in numbers (the daily electricity readings) and outputs a single number (a **risk score** between 0 and 1).

Think of it like this: you're trying to teach a robot to distinguish between normal and suspicious electricity usage patterns. You show it thousands of examples (training data) where you already know the answer, and it gradually learns the difference. Once trained, it can analyze new, unseen data.

**How it learns:** The model starts with random guesses. For every example it sees, it checks if its guess was wrong, and adjusts itself slightly to do better next time. After seeing tens of thousands of examples, it gets quite good.

---

### Baseline Models (the simple ones)

Before building complex AI, we establish **baselines** — simpler models that serve as a "minimum bar" the complex ones must beat.

#### Missingness-Only Logistic Regression
Only uses the missing-data features (how much is missing, when it starts, etc.). Scores PR-AUC = 0.170 (vs random = 0.085). **This tells us:** missing patterns alone carry some signal, but we need more.

#### Logistic Regression (full features)
Uses 56 hand-crafted features computed per customer:
- Mean, standard deviation, min, max of their consumption
- Monthly averages for each of 34 months
- Trend slope (is usage going up or down?)
- Largest monthly drop (sudden drops are a theft sign)
- Weekend vs weekday difference

Think of it as: for each customer, boil 1,035 days of data down to 56 summary numbers, then fit a simple straight-line decision boundary.

#### Random Forest
Instead of one decision rule, trains **400 decision trees** on random subsets of the data, then votes. Each tree learns different aspects of the data. Random Forest is better than Logistic Regression at capturing non-linear patterns.

The most important features the Random Forest found: **max consumption, monthly variability (std), trend slope** — matching what we saw in the EDA.

---

### 1D-CNN — The Pattern Spotter

**CNN = Convolutional Neural Network.** Originally designed for images, here adapted for time series.

**The key idea:** A **convolution** is a sliding window that scans across the data looking for a specific pattern. If the pattern is found, it activates strongly.

Imagine scanning across a consumption series with a 7-day (one week) window:
```
Day:  Mon Tue Wed Thu Fri Sat Sun | Mon Tue Wed ...
kWh:   5   6   5   6   5   8   9  |  5   6   5  ...
                                   ↑ The window slides and looks for weekly cycles
```

**Our CNN architecture** (`src/models/deep.py`):

```
Input: 1,036 days × 2 channels (consumption + missingness mask)
│
├─ Conv1D(32 filters, window=7) → BatchNorm → ReLU → MaxPool(÷2)
│   ↑ 32 different "patterns to look for", each spanning 1 week
│
├─ Conv1D(64 filters, window=5) → BatchNorm → ReLU → MaxPool(÷2)
│   ↑ 64 patterns at a wider scale (after the first pool)
│
├─ Conv1D(64 filters, window=3) → BatchNorm → ReLU → MaxPool(÷2)
│   ↑ 64 more patterns at an even wider scale
│
├─ GlobalAveragePool + GlobalMaxPool → Dropout(30%)
│   ↑ "Where do the patterns occur? How strongly?"
│
├─ Dense(32 neurons) → ReLU → Dropout(30%)
│
└─ Dense(1 neuron) → Sigmoid → Risk Score (0–1)
```

**Glossary:**
- **BatchNorm:** Normalizes the output of each layer to stabilize training
- **ReLU:** A simple activation: `max(0, x)` — passes positive signals, blocks negatives
- **MaxPool:** Takes every 2 values, keeps only the larger one — halves the sequence length
- **Dropout:** Randomly "turns off" 30% of neurons during training — prevents over-memorizing
- **Sigmoid:** Squishes any number to between 0 and 1 — perfect for a probability/risk score

Total parameters: **27,777** (a relatively small model).

---

### LSTM — The Memory Keeper

**LSTM = Long Short-Term Memory.** A type of **Recurrent Neural Network (RNN)** — designed for sequential data where order and time matter.

**The key idea:** Unlike a CNN that scans with fixed windows, an LSTM processes the sequence **step by step** and maintains an internal "memory" of what it has seen so far. It can "remember" something that happened months ago and use it to influence today's output.

**The challenge:** With 1,035 daily steps, an LSTM would be very slow and struggle to remember things from 2 years ago. **Solution:** We reshape the input from 1,035 daily steps into **148 weekly steps** (1,035 ÷ 7 ≈ 148). Now the LSTM processes one week at a time and models how consumption evolves week to week.

```
Input: 1,036 days × 2 channels
│
├─ Reshape → 148 weeks × 14 features (7 days × 2 channels)
│
├─ LSTM(64 units, return_sequences=True)   ← processes all 148 weeks, outputs all
│
├─ LSTM(32 units)                           ← condenses into one final state
│
├─ Dropout(30%) → Dense(32) → ReLU → Dropout(30%)
│
└─ Dense(1) → Sigmoid → Risk Score (0–1)
```

**Result:** The stand-alone LSTM is actually **weaker** than the CNN and even the Random Forest (validation PR-AUC = 0.359). Why? Without CNN's local pattern extraction first, the LSTM has to learn patterns like "sudden drops" directly from raw weekly data, which is harder. This motivates the hybrid model.

---

### CNN-LSTM — The Best of Both Worlds

**The proposed/deployed model.** Combines CNN's ability to detect local patterns with LSTM's ability to model how those patterns evolve over time.

```
Input: 1,036 days × 2 channels
│
├─ CNN block (same as above):
│   Conv1D(32, 7) → BN → ReLU → MaxPool(÷2)
│   Conv1D(64, 5) → BN → ReLU → MaxPool(÷2)
│   Conv1D(64, 3) → BN → ReLU → MaxPool(÷2)
│   Result: 129 steps × 64 learned feature maps
│           ↑ The 1,036 daily steps are compressed 8× into 129 richer steps
│
├─ LSTM(64 units, return_sequences=True)
│   ↑ Now processes 129 "pattern snapshots" in sequence — much more manageable!
│
├─ LSTM(32 units) → Dropout(30%) → Dense(32) → ReLU → Dropout(30%)
│
└─ Dense(1) → Sigmoid → Risk Score (0–1)
```

**Why this is better:**
- The CNN extracts **local patterns** (weekly cycles, sudden drops, long gaps)
- The CNN also **reduces sequence length** from 1,036 to 129 — making the LSTM's job much easier
- The LSTM then models **how these patterns evolve over 3 years** — did sudden drops increase over time? Did consumption trend down?
- The CNN's global pooling is **order-blind** (doesn't know if patterns happened early or late). The LSTM **replaces** that pooling and preserves temporal order.

Total parameters: **70,145** — still a relatively small model.

---

## 6. How we train the models

Training is done **offline** (on a laptop CPU), not during API requests.

**Training settings (all deep models):**

| Setting | Value | Why |
|---|---|---|
| Loss function | Binary Cross-Entropy | Standard for yes/no classification |
| Optimizer | Adam (lr = 0.001) | Adapts learning rate automatically |
| Batch size | 128 customers | Processes 128 at a time |
| Max epochs | 40 | Upper limit on training rounds |
| Early stopping | Patience 6 on val PR-AUC | Stops if no improvement for 6 epochs |
| Class weights | {Normal: 0.55, Theft: 5.87} | Theft is 8.5% — weight it ~11× higher |
| ReduceLROnPlateau | Factor 0.5, patience 3 | Halves learning rate if stuck |
| Random seed | 42 (main run) | Reproducibility |

**Why class weights?** If 91.5% of customers are normal, a dumb model that always says "NORMAL" gets 91.5% accuracy. That's useless. We penalize wrong predictions on theft customers much more heavily.

**Training times on a CPU:**
- 1D-CNN: ~28 minutes
- LSTM: ~23 minutes
- CNN-LSTM: ~83 minutes

---

## 7. How we evaluate performance

We use **imbalance-aware metrics** because only 8.5% of customers are theft-labeled.

### The confusion matrix
After picking a decision threshold, every prediction is one of:

| | Predicted NORMAL | Predicted SUSPICIOUS |
|---|---|---|
| **Actually NORMAL** | ✅ True Negative (TN) | ❌ False Positive (FP) — false alarm |
| **Actually THEFT** | ❌ False Negative (FN) — missed theft | ✅ True Positive (TP) — caught theft |

### Key metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | % correct. **Misleading on imbalanced data!** |
| **Precision** | TP / (TP+FP) | Of flagged meters, what fraction is real theft? |
| **Recall** | TP / (TP+FN) | Of actual theft meters, what fraction did we catch? |
| **F1** | 2×P×R / (P+R) | Harmonic mean of precision and recall |
| **PR-AUC** | Area under Precision-Recall curve | **Primary metric** — not fooled by imbalance |
| **ROC-AUC** | Area under ROC curve | Overall ranking ability |
| **FPR** | FP / (FP+TN) | False alarm rate |

> **Why PR-AUC is the primary metric:** A random classifier on our 8.5% positive dataset scores PR-AUC ≈ 0.085. A perfect classifier scores 1.0. We want to be as far above 0.085 as possible. Regular accuracy is useless here — a model that always predicts NORMAL gets 91.5% accuracy but catches zero thieves.

### Decision threshold
The model outputs a continuous score (0.0 to 1.0). We pick a **threshold** on the **validation set** (never the test set): any meter scoring ≥ threshold is flagged SUSPICIOUS.

We choose the threshold that **maximizes F1** on the validation set. For the deployed CNN-LSTM (seed 1), the threshold is **0.764**.

---

## 8. The Results

### Model comparison (validation set — used for development)

| Model | Accuracy | Recall | PR-AUC | FPR |
|---|---|---|---|---|
| Missingness-only LR | 0.803 | 0.387 | 0.170 | 0.158 |
| Logistic Regression | 0.865 | 0.426 | 0.292 | 0.094 |
| Random Forest | 0.904 | 0.448 | 0.418 | 0.053 |
| 1D-CNN | 0.913 | 0.520 | 0.453 | 0.051 |
| LSTM | 0.885 | 0.450 | 0.359 | 0.074 |
| **CNN-LSTM ✓ Deployed** | **0.899** | **0.548** | **0.478** | 0.068 |

### Final test results (used ONCE — the honest numbers)

| Model | Recall | PR-AUC | FPR |
|---|---|---|---|
| Logistic Regression | 0.382 | 0.284 | 0.094 |
| Random Forest | 0.391 | 0.394 | 0.052 |
| 1D-CNN (3 seeds avg) | 0.488 | 0.459 | 0.060 |
| LSTM | 0.395 | 0.337 | 0.073 |
| **CNN-LSTM (3 seeds avg)** | **0.516** | **0.475** | 0.065 |
| *Deployed: CNN-LSTM seed 1* | *0.500* | *0.487* | *0.053* |

### What the numbers mean in plain English

The deployed CNN-LSTM (seed 1) on the **6,355 test customers**:
- **Caught 271 out of 542 actual theft customers** (50% recall) — found half the thieves
- **Raised 578 alerts in total** — 271 correct (TP) + 307 false alarms (FP)
- **False alarm rate: 5.3%** — only 1 in 19 normal customers gets falsely flagged
- **Precision: 46.9%** — almost 1 in 2 flagged customers is actually theft-labeled
- **PR-AUC = 0.487** — about 5.7× better than a random classifier (0.085)

**Why this is valuable:** Without AI, an inspector checking random meters would find theft in 1 of every 12 customers checked (8.5%). With this system, they find theft in almost 1 of every 2 flagged customers — a **~5.5× improvement in inspection efficiency**.

---

## 9. The Backend API

The backend is a **FastAPI** web server (`backend/app/`) — a Python framework that handles HTTP requests.

**What is an API?** An Application Programming Interface — a way for different software to talk to each other. The frontend dashboard talks to the backend through the API.

The server runs at: **http://localhost:8000**

### Available endpoints (routes)

| Method | URL | What it does |
|---|---|---|
| GET | `/health` | Is the server alive? Is the model loaded? |
| POST | `/predict` | Submit one meter's readings → get a risk score |
| POST | `/predict/batch` | Submit many meters at once |
| GET | `/meters` | List all monitored meters with their latest status |
| GET | `/meters/{id}` | Full detail for one meter (readings, history, alerts) |
| GET | `/alerts` | List alerts (filter by OPEN/REVIEWED/DISMISSED) |
| PATCH | `/alerts/{id}` | Mark an alert as reviewed or dismissed |
| GET | `/dashboard/summary` | Overview stats for the main dashboard page |
| GET | `/model/metrics` | Stored training/evaluation results |

### Example: Submitting a meter for scoring

**Request:**
```json
POST /predict
{
  "meter_id": "C00001",
  "readings": [5.2, 4.8, null, 6.1, ...]
}
```

**Response:**
```json
{
  "meter_id": "C00001",
  "prediction": "SUSPICIOUS",
  "risk_score": 0.876,
  "threshold": 0.764,
  "recommendation": "Investigation recommended: high-risk consumption pattern.",
  "alert_raised": true,
  "duplicate": false
}
```

The API never re-trains the model. Training is offline-only. The model is loaded once at startup and shared across all requests.

> Interactive API documentation: **http://localhost:8000/docs** — test all endpoints directly in the browser.

---

## 10. The Database

A **SQLite** database (`backend/etd.db`) stores everything persistently — it's a single file, no separate database server needed.

### Tables

#### `meters` — one row per unique customer ID
#### `readings` — the raw submitted daily values (JSON array), deduplicated by hash
#### `predictions` — the model's output for each reading (risk score, label, threshold)
#### `alerts` — raised automatically when risk score >= threshold

An alert has three states:
- **OPEN** — just raised, needs attention
- **REVIEWED** — an inspector looked at this meter
- **DISMISSED** — decided it's not worth investigating

---

## 11. The React Dashboard

The frontend is a **React** app (`frontend/`) built with **Vite** (a fast build tool) and **Recharts** (for charts). It runs at: **http://localhost:5173**

React is a JavaScript library that builds interactive web pages. Every piece of data shown comes from the backend API — the frontend has no logic, only display.

### Pages

#### 🏠 Dashboard (Overview)
- Big stat tiles: total meters, suspicious count, predictions count, alert counts
- **Risk score histogram:** distribution of all meters' latest scores, with the alert threshold marked
- **Highest-risk meters table:** top 8 meters by risk score, with live-updating risk bars
- **Recent alerts:** last 10 alerts, with new ones flashing when they arrive during the session
- Auto-refreshes every 5 seconds while the simulator runs

#### 📊 Meters
- List all monitored meters sorted by risk score
- Filter by NORMAL or SUSPICIOUS, search by meter ID
- Click any meter to go to its detail page

#### 🔍 Meter Detail
- Full consumption chart: all 1,035 days plotted as a line, with missing-data gaps highlighted
- A zoom brush to focus on specific time periods
- Prediction history and alert history for this meter

#### 🔔 Alerts
- Filterable list: OPEN / REVIEWED / DISMISSED / All
- Each alert shows: meter ID, risk score bar, status, timestamp
- Action buttons: **Mark reviewed** | **Dismiss**

#### 📈 Model Performance
- Validation results table (all 6 models side by side)
- Final test results table with best values bolded
- **PR-AUC bar chart** comparing all models on validation and test
- **Confusion matrix** for the deployed model

#### ℹ️ About / Methodology
- Explains the system's purpose and limitations in plain language

### UI Features
- **Risk meter widget:** each score shown as a 0–1 bar with a tick at the threshold
- **Themes:** Light / Dark / System — toggle in the sidebar
- **Quick search:** Ctrl+K (or `/`) opens a search box for pages and meter IDs
- **Responsive layout:** works on phones and tablets
- **No internet required:** fonts are bundled locally

---

## 12. The Simulated Real-Time Monitor

`scripts/simulate_monitoring.py` is a demo script that shows what the system looks like in action.

**What it does:** Takes the 6,355 held-out test customers and submits them to the API one by one, with a 2-second pause between each. This simulates meters checking in over time.

```bash
python scripts/simulate_monitoring.py
# Output:
# [   1/ 200] meter C12345: NORMAL     risk 0.042
# [   2/ 200] meter C67890: SUSPICIOUS risk 0.891  -> ALERT (investigation recommended)
```

The dashboard refreshes every 5 seconds and you can watch meters and alerts appear in real time.

> **NOTE:** This is a **simulation** using historical held-out test data. It is NOT connected to real electricity meters. The labels are never sent to the API — the model genuinely "doesn't know" who is theft-labeled when it scores them.

---

## 13. How Everything Connects

```
OFFLINE PHASE (done once, on a laptop):
─────────────────────────────────────────
Raw CSV Data
    ↓ (preprocessing pipeline)
Cleaned numpy arrays (train / val / test)
    ↓ (model training scripts)
Trained model files (models/cnn_lstm/)
    ↓ (evaluate_test.py)
artifacts/metadata/model.json  ← threshold + model path

ONLINE PHASE (live, two servers running):
─────────────────────────────────────────
                    ┌──────────────────────────────┐
Simulator script ──►│                              │
  or               │  FastAPI Backend :8000        │◄──► SQLite DB (etd.db)
Real meter data ──►│  (loads model.json once)      │
                    │                              │
                    └──────────────┬───────────────┘
                                   │ REST API (JSON)
                                   ↓
                    ┌──────────────────────────────┐
                    │  React Dashboard :5173        │◄──── Utility Staff
                    │  (polls API every 5s)         │       view & manage alerts
                    └──────────────────────────────┘
```

---

## 14. Project Folder Structure

```
PAI/
│
├── data/
│   ├── raw/              ← Original SGCC CSV file (not in Git — too large)
│   └── processed/        ← Cleaned numpy arrays: train.npz, val.npz, test.npz
│
├── src/                  ← Core Python library (reusable code)
│   ├── preprocessing/    ← pipeline.py: load, clean, split, transform
│   ├── models/           ← deep.py: CNN, LSTM, CNN-LSTM model definitions
│   ├── features/         ← summary.py: hand-crafted features for baselines
│   ├── evaluation/       ← metrics.py: PR-AUC, threshold selection
│   └── inference/        ← predictor.py: load model + run predictions (used by API)
│
├── scripts/              ← Command-line entry points
│   ├── preprocess.py          ← Run the full preprocessing pipeline
│   ├── eda.py                 ← Exploratory data analysis (makes figures)
│   ├── train_baselines.py     ← Train LR and Random Forest
│   ├── train_deep.py          ← Train CNN / LSTM / CNN-LSTM
│   ├── evaluate_test.py       ← ONE-TIME final test evaluation
│   └── simulate_monitoring.py ← Demo real-time simulation
│
├── backend/              ← FastAPI web server
│   └── app/
│       ├── main.py            ← App entry point
│       ├── api/routes.py      ← All HTTP endpoints
│       ├── models/db.py       ← Database tables
│       ├── schemas/api.py     ← Request/response data shapes
│       └── services/scoring.py← Model loading + score-and-store
│
├── frontend/             ← React dashboard
│   └── src/
│       ├── pages/             ← Dashboard, Meters, Alerts, ModelPerformance, About
│       ├── components/        ← Reusable UI pieces (cards, risk meter widget)
│       └── api.js             ← Functions that call the backend API
│
├── models/               ← Saved trained model files (not in Git)
│   └── cnn_lstm/              ← CNN-LSTM weights (deployed)
│
├── artifacts/            ← Metadata files kept in Git
│   ├── scalers/preprocessor.json  ← Fitted cap/mean/std values
│   └── metadata/model.json        ← Deployed model info + threshold
│
└── reports/
    ├── figures/           ← EDA charts, PR curves, confusion matrices
    └── results/           ← JSON files with all metric numbers
```

---

## 15. Limitations and Honest Caveats

| Limitation | Why it matters |
|---|---|
| **Labels may be noisy** | Labels come from past inspections. Some "normal" customers may be undetected thieves |
| **Fixed time window** | The model needs exactly 2014-01-01 to 2016-10-31 (1,035 days). It cannot score a newer meter without retraining |
| **Simulation only** | The "real-time" monitoring feature is a replay of historical test data — not connected to real meters |
| **Region-specific** | Trained on Chinese utility data. Patterns might differ for utilities in other countries |
| **Missing data signal** | Part of the model's performance may come from missing-data patterns rather than actual consumption behaviour |
| **Decision support only** | A SUSPICIOUS label is a recommendation to inspect, never evidence of theft |
| **CPU training only** | Models were trained on a laptop CPU. A GPU and more search could improve results |

> ⚠️ **This system must never be used as direct evidence in legal or criminal proceedings. It is a prioritization tool for inspectors, not a proof of wrongdoing.**

---

*This document was written to explain the project to someone with no prior knowledge of AI, machine learning, or deep learning. For precise implementation details, refer to the technical [README.md](README.md).*
