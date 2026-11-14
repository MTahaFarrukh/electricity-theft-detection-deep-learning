# Viva preparation: likely questions and answers

Every number below comes from this project's own outputs (`reports/`). Final test numbers are in README section 12.

## Data and problem

**Q: What dataset did you use, and what does a label mean?**
SGCC (State Grid Corporation of China): 42,372 customers, daily kWh from 2014-01-01 to 2016-10-31, 8.53% theft-labelled.
Labels are **per customer**. They come from the utility's inspections, so they are noisy: undetected theft is labelled
normal, and the audit even found normal-labelled customers who read 0 kWh on every day.

**Q: Why not use accuracy?**
Only 8.5% of customers are positive, so a model that always says "normal" scores about 91.5% accuracy and catches nothing.
We use **PR-AUC** as the primary metric (random level ≈ 0.085) and also report recall, precision, F1, ROC-AUC and the
false-positive rate. FPR matters because every false alarm sends an inspector to an honest customer.

**Q: What did the data audit find that changed your design?**
(1) Date columns were stored in string order (2014/1/1, 2014/1/10, …) and had to be sorted. (2) One day (2016-09-18)
is absent. (3) 25.6% of readings are missing, mostly because ~16k series start late. (4) The maximum reading is 800,003 kWh
against a 99.9th percentile of 372, so outlier capping is needed. (5) Labels are per customer, so we do not window the series.

## Leakage and methodology

**Q: How did you prevent data leakage?**
- Customers were split 70/15/15 (stratified by label) **before** anything was fitted. No customer is in two splits.
- The outlier cap, log-scaling mean and std were fitted on **training readings only** (`artifacts/scalers/preprocessor.json`).
- Interpolation uses only each customer's own series, so no information crosses customers.
- EDA was done on the training split only.
- Early stopping, hyper-parameters and the decision threshold use **validation** data only.
- The test set was used **once**. The script that evaluates it refuses to run a second time.

**Q: Why one full series per customer instead of sliding windows?**
The label belongs to the customer, not to a period. A window from a "theft" customer may contain no theft at all,
so windowing would create many wrongly labelled samples and leak one customer's behaviour across many samples.

**Q: How did you handle class imbalance? Why not SMOTE?**
Balanced class weights in the loss (positives weighted ≈ 10× more). SMOTE interpolates between samples. On raw
time series, that creates unrealistic synthetic consumption curves, and if it is applied before the split it leaks.

**Q: How was the threshold chosen?**
For each model, the threshold that maximises F1 on the validation set. It is stored with the model in
`artifacts/metadata/model.json` and the API reads it from there. A utility could instead choose a threshold that
matches its inspection capacity (future work).

**Q: How did you choose the final model? Didn't you pick it by looking at test results?**
No. The rule was written in the README before the test set was used. The deployed architecture is the one with the
highest **mean validation PR-AUC over three seeds**, and the deployed weights are its best validation run.
`evaluate_test.py` writes this choice to `model.json` before it computes any test score.

**Q: Why three seeds?**
On single runs the CNN-LSTM had the higher validation PR-AUC (0.478 vs 0.453), while the CNN was better on ROC-AUC, F1
and FPR. A gap that size can be run-to-run noise, so we trained each model three times and compare means and spreads
before making any claim.

**Q: What are the final results?**
On test (6,355 customers, used once), the CNN-LSTM reaches PR-AUC 0.475 ± 0.016 over 3 seeds vs 0.459 ± 0.011 for the 1D-CNN,
0.394 for Random Forest, 0.337 for LSTM and 0.284 for Logistic Regression (random ≈ 0.085). The deployed run (seed 1) scores
PR-AUC 0.487, recall 0.500, precision 0.469 and FPR 0.053: about half of the theft-labelled customers are flagged, and about
1 in 2 flagged customers is theft-labelled.

**Q: So is the CNN-LSTM better than the CNN?**
It has higher mean PR-AUC, recall and F1. The CNN has higher ROC-AUC and lower FPR. The PR-AUC gap is about one standard
deviation across seeds, so we say the hybrid is **competitive with, not significantly better than**, the CNN. Its practical
advantage is recall. Both clearly beat the baselines. Honest reporting here is a deliberate choice.

**Q: Did you look at the test set more than once?**
The first run of the evaluation script crashed on a dtype bug while writing the error-analysis CSV, after the metrics had
been computed but before they were saved. We fixed that one line and re-ran it. The model, threshold and selection were
already fixed in `model.json`, and every number was identical (the crashed log is kept). Nothing was tuned on test.

## Models

**Q: Describe the CNN-LSTM.**
Three Conv1D blocks (kernels 7, 5, 3; 32/64/64 filters; batch norm, ReLU, max-pool ×2 each) turn the 1,036-day input
(value + missingness mask) into 129 steps of 64 learned features. Two LSTM layers (64, 32) model how these local
patterns evolve over time, followed by dense layers and a sigmoid risk score. The first kernel spans 7 days because the
EDA showed a weekly cycle.

**Q: Why a hybrid? What does each part contribute?**
The CNN detects local shapes (sudden drops, flat periods, weekly irregularity). The LSTM models their order and long-term
trend, which the CNN's global pooling discards. The stand-alone LSTM was the weakest deep model (validation PR-AUC 0.359):
without convolutional feature extraction it struggles to learn local patterns from raw inputs.

**Q: Why does the LSTM read weeks instead of days?**
1,036 daily steps is very slow to train on a CPU, and LSTMs struggle with dependencies that long. Reshaping to 148 weekly
steps × 14 features keeps all the information and the weekly structure.

**Q: What is the missingness mask and why include it?**
A second input channel: 1 where the reading was originally missing. Imputed values must not look like real readings, and
the EDA showed missingness differs by class.

**Q: Isn't the model then just learning missing-data artefacts?**
That is the main threat to validity, and we measured it two ways. (1) A **missingness-only** baseline reaches validation
PR-AUC 0.170, about twice the random level, so missingness carries some signal, but the full models reach 0.42–0.48.
(2) On test customers with ≤ 5% missing readings, the deep models still reach a PR-AUC of about 0.44 (random 0.058;
missingness-only 0.132). The Random Forest drops more (0.294), so the deep models rely less on missingness. Error analysis
does show that false alarms have more missing data than true negatives, so the effect is reduced, not zero.

**Q: Why did theft-labelled customers consume more, not less?**
Their median daily consumption is 8.8 vs 5.5 kWh, and their usage is more variable. Theft in this data shows up as
irregular patterns (for example a sudden long drop from a higher level), not simply as low consumption. This is why
variability and trend features dominate the Random Forest.

## System

**Q: How do you guarantee the API uses the same preprocessing as training?**
There is a single inference module, `src/inference/predictor.py`, used by the API, the simulator and the evaluation.
It loads the training-fitted preprocessor. Before testing, `evaluate_test.py` checks that raw readings → Predictor
reproduces the evaluation scores on 500 validation customers (max difference < 1e-4).

**Q: Is this real-time?**
No, and the UI says so. "Simulated Real-Time Monitoring" replays held-out **test** customers to the API at a fixed
interval. Labels are never sent. An identical resubmission is detected by a hash and does not create a duplicate alert.

**Q: What does "SUSPICIOUS" mean legally or operationally?**
A prioritisation signal: "investigation recommended". It is never proof of theft, and the UI never says "theft".

**Q: Main limitations?**
Label noise, the missingness association, a single utility and region, a fixed 1,035-day input window (it cannot score
new meters or later periods without retraining), and limited CPU-only hyper-parameter tuning.
