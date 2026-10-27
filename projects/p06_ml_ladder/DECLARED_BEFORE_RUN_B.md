# Declared before the run — Part B: how far will a UF cycle's resistance climb?

Written on 2026-10-27, after the development work in `stepB1_uf_dev.py` and **before any
2014–2015 cycle was scored by any model of this part**. Committed before `stepB2_uf_test.py` runs;
the script refuses to run otherwise and records the commit. Nothing above the closing line is edited
after the run.

## The question

At minute 5 of a filtration cycle of the seawater UF pilot (Project 4's segments: steady flow, one
set of elements online), how far will the resistance index climb from minute ~4 to minute ~11?

    y = 100 × [ mean ln R over 10–12 min − mean ln R over 3–5 min ]      (percent of R)

What the plant may use: the cycle's first five minutes of ln R, the cycles before it in the same
episode (within 3 h; each known only after it ended), and the conditions it set — flow per element,
temperature, elements online — plus how long the episode has run and the days since the last
chemical clean seen in the record. Feature lists: `PLAIN`, `PHYSICS`, `CLOCK` in
`chemai/features/uf_forecast.py`.

## Data and split

Project 4's declared confirmation split. **Development: the 4,168 cycles before 2014-01-01
(40 episodes). Test, scored once: the 10,830 cycles from 2014-01-01 to 2015-04-01.** Project 4 and 5
looked at the fouling rates of 2014–15 in aggregate (the tide and sun tests, the backwash economics);
no model of this part was fitted or scored on them. Tuning is grouped by episode.

## The methods to beat (development MAE, percentage points of R)

| Method | Dev MAE |
|---|---:|
| No model: the development median climb (1.64 %) | 1.537 |
| Physics, straight line: extrapolate the first five minutes (Hermia intermediate blocking) | 3.605 |
| **Physics, Hermia: the best of the four constant-flux laws fitted to the first five minutes, extrapolated** | 2.974 |
| History: the last cycle's fouling rate (same elements if possible) | 1.730 |

## The ladder (frozen in `results/stepB1_uf_dev.json`)

| Rung | Models | Dev MAE (5-fold, grouped by episode) |
|---|---|---|
| Machine learning (plain features) | ridge, k-NN, RBF SVR, random forest, extra trees, XGBoost, LightGBM, each tuned on a small grid | 1.396–1.473; pick **extra trees 1.396** |
| Ensemble | mean of extra trees, SVR, random forest | 1.389 |
| Physics-informed machine learning | the same seven on plain + physics features (viscosity, the four Hermia constants of the first five minutes, the best law, the physics and history predictions, the flow ratio) | 1.393–1.438; pick **extra trees 1.393** |
| Physics-informed ensemble | extra trees, SVR, XGBoost | 1.377 |
| Deep learning | MLP on the plain features; a GRU over the eight cycles before + the plain features | MLP **1.427**, GRU 1.437 |
| Physics-informed deep learning | **HermiaNet** — a Hermia PINN in the sense of the 2025 literature: the network weights the four blocking mechanisms and gives each a rate constant; the constant-flux laws turn them into the climb, so the output is always a rise some mix of the laws allows | 1.434 |

Every model is refitted on all development cycles with its frozen settings (the SVR on a
3,000-row subsample, as in tuning); each network trains once, with early stopping on a fifth of the
development episodes.

## What is reported, and the pass mark

For every model and method: test MAE, and its difference from the no-model median, from the Hermia
extrapolation and from the history method, each with a 95 % interval from resampling whole episodes.

**A rung beats the no-model baseline** if its pick's test MAE is lower than the no-model median's
with the 95 % interval of the difference below zero; the same for the Hermia extrapolation.
Otherwise it is a tie or a loss.

## Also reported, declared now

- **The clock, by machine learning:** the frozen ML pick refitted with hour-of-day and M2-tide-phase
  features. On development it changed the MAE by 0.0002. Project 4 found no tide and no daylight
  effect; a test MAE that does not improve is the same answer from a different method.
- **Can Hermia's mechanisms be told apart here?** HermiaNet's mean mechanism weights on the test
  cycles, and, for every test cycle, the spread at minute 12 between the four laws fitted to its
  whole 12-minute curve, against the cycle's own point-to-point noise. On development the best law of
  the first five minutes was "complete blocking" or "cake" in 98 % of cycles — the two that bend
  most — which says the noise, not the mechanism, picks the law at this scale.

---
