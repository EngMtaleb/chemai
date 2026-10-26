# Declared before the run — Part C: where should a closed-circuit RO batch end?

Written on 2026-10-27, after the development work in `stepC1_ccro_dev.py` and **before any model
of this part forecast a test batch**. Committed before `stepC2_ccro_test.py` runs; the script refuses
to run otherwise and records the commit. Nothing above the closing line is edited after the run.

## The question and the method to beat

Project 5's declared test T4: from the live cost between minutes 5 and 20 of a batch, the
square-root law (a straight line m = m0 + s V, then V* = √(2 F_eff / s)) forecasts where the batch
should end. On the test batches it was within 4–7 % of each batch's own best for b ≤ 3, and **11 %
short at b = 10** — it failed there. The live cost bends; a straight line cannot.

Here every method forecasts the live cost after minute 20, m(r)/m20 with r = V/V20, and the same
economics picks the end: J(V) = (F_b + E20 + ∫ m dV) / V, V* = argmin J over r from 1 to 10, with
the previous batch's drain as the fixed cost, as the rule and T4 use. Only the curve differs.

## Data and split

| Role | Data | Batches (≥ 25 min) |
|---|---|---|
| Development | constant-setpoint batches of dataset 3 (extremum seeking: Exp2, Exp2 continued, Exp3) and dataset 4 (open loop) | 328 + 8 = 336; 13,810 points after minute 20 |
| **Test, scored once** | **dataset 6 (Bayesian optimisation)** — the batches T4 was scored on | 142 |

The extremum-seeking batches stop early: after minute 20 they reach at most r = 2.34. Only the eight
open-loop batches run far (r up to 6.6). The test batches run to r = 6.9, and their best ends lie far
out — **a forecast here must extrapolate beyond almost all of its training data.** Tuning is grouped
by experiment (four groups); the open-loop fold is the one check of that extrapolation and is
reported on its own. T4's batch-level results on dataset 6 are known (Project 5); no model of this
part has forecast a dataset-6 batch.

## The ladder (frozen in `results/stepC1_ccro_dev.json`)

Development MAE of m/m20 after minute 20 (all folds · open-loop fold). The straight line: 0.0143 ·
0.0169; a flat cost: 0.0285.

| Rung | Models | Dev MAE |
|---|---|---|
| Machine learning (plain features + r → m/m20) | ridge, k-NN, RBF SVR, random forest, extra trees, XGBoost, LightGBM | 0.007–0.050; pick **ridge 0.0070 · 0.0228** |
| Ensemble | ridge, extra trees, random forest | 0.0068 · 0.0332 |
| Physics-informed ML — a hybrid: the model learns only the departure from the square-root law's straight line, with physics features (the line at r, the conductivity a steady salt build-up would reach) | the same seven | 0.0045–0.0165; pick **extra trees 0.0045 · 0.0079** |
| Physics-informed ensemble | extra trees, LightGBM, k-NN | 0.0050 · 0.0120 |
| Deep learning | an MLP on the features and r | **0.397 · 0.568** — it fails whenever the held-out experiment differs |
| Physics-informed deep learning | **PowerLawNet**: the cost keeps the slope measured at minute 20, and the network learns only how that slope decays, m/m20 = 1 + k·slope·(r^a − 1)/a (a = 1 is the straight line); every forecast rises smoothly | 0.0083 · 0.0353 |

Every model is refitted on all development points with its frozen settings (the SVR on 4,000
points, k-NN on 12,000, as in tuning); each network trains once, with early stopping on one
development experiment.

## What is reported, and the pass mark

For every method, per b in {0, 1, 3, 10}, on the test batches whose own best lies inside the batch
(Project 5's "inside": 103, 97, 75 and 10 batches): the median absolute error of V* against the own
best (T4's measure), the share within 10 %, the median V*/V_best, the median cost of stopping at V*
above the best, and how many forecasts fall beyond the batch's end. Also: each method's MAE of m/m20
on the test batches' real curves after minute 20.

**A rung beats the square-root law** if, at b = 0, 1 and 3 (the brine prices with at least 75 inside
batches), its pick's median absolute error is lower than T4's with the 95 % interval of the
difference below zero (resampling batches). At b = 10 there are 10 inside batches: reported, no
pass mark. Otherwise the result is a tie or a loss.

---
