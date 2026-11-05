# Phase 3 addition — Prophet, an LSTM and a plain RNN, judged by phase 3's own test

Written 2 November 2026, about 00:30 UTC. The addition was declared in `declaration_addendum.md` before any new model ran.
Its SHA-256 was recorded on 1 November 2026 at 23:02:43 UTC (`declaration_addendum.sha256`) and still matches, so the
addendum was not edited after the run. No "Changes after the run" section was needed: no declared setting changed. What
happened during the run is listed under "What went wrong, and doubts".

## Headline

**No new method passed the bar, for either indicator, at any horizon, in either test: 0 of 64 new verdicts.**

- **ΔP:** every new method did worse than *no change* at every horizon, in both tests. The closest was the plain RNN 1 day
  ahead in the rolling-origin test: −1.6 %, better in 2 of 4 blocks. On an unseen train the closest was the LSTM 1 day ahead:
  −3.0 %, better in 0 of 4 blocks. Over the four horizons (rolling origin / unseen train):
  - LSTM: −6.4 to −11.8 % / −3.0 to −32.0 %
  - RNN: −1.6 to −17.5 % / −4.7 to −23.4 %
  - Prophet (as approved): −151.0 to −309.4 %
  - Prophet + level shift: −26.9 to −91.9 %

  Both Prophet forms use only each train's own history, so their numbers are the same in both tests. The RNN 7 days ahead
  (rolling origin) beat *no change* in 3 of the 4 blocks but was −11.8 % overall, because block 1 was −101.9 %.
- **Flux loss:** no pass either. The closest was the declared variant, Prophet + level shift, 14 days ahead: +10.0 % over all
  blocks, which clears the +5 %. But it beat *no change* in only 2 of the 4 blocks (block 1 −1.1 %, block 2 +29.4 %,
  block 3 −1.6 %, block 4 +22.6 %), so it fails the declared rule, in both tests. The approved Prophet form was −3.6 % there.
  Over the four horizons (rolling origin / unseen train):
  - LSTM: −2.5 to −16.9 % / −6.4 to −24.2 %
  - RNN: −4.8 to −11.8 % / −6.4 to −24.7 %
  - Prophet (as approved): −3.6 to −91.4 %

  The only flux pass is still phase 3's own: the smoothed state 14 days ahead (+7.1 %, 3 of 4 blocks, both tests).
- **Phase 3's declared results stay exactly as they are.** All 71 phase-3 files outside `addition/` have the same SHA-256 as
  before this work began. The six methods' rows below are the saved numbers: `score.py` re-scores the saved forecasts and stops
  unless they equal `scores_rolling.csv` and `scores_unseen_train.csv`. The largest difference was 2.8e-14, with the same verdicts.

## What was run (declared in the addendum)

- **Prophet (as approved):** one model per train, indicator and origin day *t*, fitted on that train's records up to day *t*.
  Changepoints sit exactly at the train's ledger cleaning days known at the origin. Weekly seasonality on, yearly and daily off,
  Prophet 1.5.0 defaults otherwise. 2,584 fits (646 origins × 2 indicators × 2 forms); none failed.
- **Prophet + level shift (declared variant):** the same, plus a step regressor at each cleaning before the origin. A Prophet
  changepoint can change the trend's slope but cannot make it drop at a cleaning.
- **LSTM and plain RNN (Elman):**
  - **Inputs:** a 17-day sequence (t − 16 … t) of the four daily indicators, with a record flag and a current-cycle flag; plus
    XGBoost's other 8 inputs and 3 flags for missing values.
  - **Network:** one recurrent layer of 32 units and a dense layer of 16. The target is the change to *t + h*, as for XGBoost.
  - **Training:** per fold, on exactly XGBoost's training rows. Stage 1 stops early on the last 20 % of those rows (by day);
    stage 2 refits on all of them for the chosen number of epochs. Five seeds (0–4) are averaged, as for XGBoost.
- **Same test and same bar:** the folds are built by the same lines as `run.py`, and the scores by `p3lib.scores` and
  `p3lib.verdict`, imported unchanged.

## The check before the run

Before any new method ran, the harness reproduced phase 3's saved numbers (`reproduce_check.txt`):

- **Test pairs:** the same 4,666 test forecasts in each test, with identical pairs, today's values and truths.
- ***No change* MAE:** the same for every indicator and horizon in both tests (for example ΔP 7 days 2.912653 pp, flux 14 days
  4.307318 pp).
- **XGBoost and hybrid:** re-run with phase 3's own code for ΔP 7 days ahead (rolling origin and unseen train) and flux 14 days
  ahead (rolling origin). The forecasts matched the saved ones (largest difference 9.9e-17), and so did the skills (XGBoost:
  ΔP 7 days −9.347 % rolling origin and −28.459 % unseen train; flux 14 days +3.562 %).

## Results

*Skill* is the % less error than *no change* over all four test blocks; negative means worse. *(k/4)* is the number of blocks
with positive skill. *Bar*: skill at least +5 % and positive in at least 3 of the 4 blocks. Rows not marked "new" are phase 3's
six methods, as saved. Blocks by day *t*: 1 = 550–599, 2 = 600–649, 3 = 650–699, 4 = 700–771.

### Overview

#### ΔP (`dp_pct`) — rolling origin

*No change* MAE, percentage points: 1 d 1.76, 3 d 2.83, 7 d 2.91, 14 d 3.49

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -20.8 (1/4) | fail | +2.2 (3/4) | fail | +1.8 (2/4) | fail | +3.6 (4/4) | fail |
| trend | -4.5 (0/4) | fail | -12.5 (0/4) | fail | -26.4 (0/4) | fail | -53.2 (0/4) | fail |
| fouling law | -48.5 (0/4) | fail | -6.3 (1/4) | fail | -16.2 (1/4) | fail | -20.4 (1/4) | fail |
| XGBoost | -15.7 (1/4) | fail | -0.8 (1/4) | fail | -9.3 (2/4) | fail | -14.6 (1/4) | fail |
| hybrid | -50.4 (0/4) | fail | -36.6 (1/4) | fail | -24.9 (1/4) | fail | -32.2 (0/4) | fail |
| **Prophet** (new) | -309.4 (0/4) | fail | -165.7 (0/4) | fail | -171.6 (0/4) | fail | -151.0 (0/4) | fail |
| **Prophet + level shift** (new) | -91.9 (0/4) | fail | -27.6 (1/4) | fail | -31.7 (1/4) | fail | -26.9 (0/4) | fail |
| **LSTM** (new) | -6.4 (1/4) | fail | -11.8 (0/4) | fail | -6.5 (2/4) | fail | -10.9 (1/4) | fail |
| **RNN** (new) | -1.6 (2/4) | fail | -4.9 (1/4) | fail | -11.8 (3/4) | fail | -17.5 (1/4) | fail |

#### flux loss (`flux_pct`) — rolling origin

*No change* MAE, percentage points: 1 d 2.11, 3 d 3.07, 7 d 3.48, 14 d 4.31

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -15.9 (0/4) | fail | +1.4 (2/4) | fail | +2.7 (2/4) | fail | +7.1 (3/4) | **PASS** |
| trend | -3.1 (0/4) | fail | -8.9 (0/4) | fail | -22.1 (0/4) | fail | -27.3 (0/4) | fail |
| fouling law | -44.9 (0/4) | fail | -13.6 (1/4) | fail | -7.9 (2/4) | fail | +1.0 (1/4) | fail |
| XGBoost | -12.1 (1/4) | fail | -2.7 (2/4) | fail | -5.8 (2/4) | fail | +3.6 (2/4) | fail |
| hybrid | -28.7 (0/4) | fail | -16.9 (2/4) | fail | -18.7 (2/4) | fail | -3.3 (2/4) | fail |
| **Prophet** (new) | -91.4 (0/4) | fail | -33.4 (1/4) | fail | -21.8 (1/4) | fail | -3.6 (2/4) | fail |
| **Prophet + level shift** (new) | -54.4 (0/4) | fail | -10.4 (2/4) | fail | -1.8 (2/4) | fail | +10.0 (2/4) | fail |
| **LSTM** (new) | -5.9 (0/4) | fail | -2.5 (2/4) | fail | -10.3 (1/4) | fail | -16.9 (1/4) | fail |
| **RNN** (new) | -5.9 (0/4) | fail | -4.8 (1/4) | fail | -10.7 (1/4) | fail | -11.8 (1/4) | fail |

#### ΔP (`dp_pct`) — unseen train

*No change* MAE, percentage points: 1 d 1.76, 3 d 2.83, 7 d 2.91, 14 d 3.49

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -20.8 (1/4) | fail | +2.2 (3/4) | fail | +1.8 (2/4) | fail | +3.6 (4/4) | fail |
| trend | -4.5 (0/4) | fail | -12.5 (0/4) | fail | -26.4 (0/4) | fail | -53.2 (0/4) | fail |
| fouling law | -48.5 (0/4) | fail | -6.3 (1/4) | fail | -16.2 (1/4) | fail | -20.4 (1/4) | fail |
| XGBoost | -22.9 (0/4) | fail | -12.4 (1/4) | fail | -28.5 (1/4) | fail | -24.6 (1/4) | fail |
| hybrid | -78.0 (0/4) | fail | -58.7 (1/4) | fail | -67.3 (1/4) | fail | -57.9 (0/4) | fail |
| **Prophet** (new) | -309.4 (0/4) | fail | -165.7 (0/4) | fail | -171.6 (0/4) | fail | -151.0 (0/4) | fail |
| **Prophet + level shift** (new) | -91.9 (0/4) | fail | -27.6 (1/4) | fail | -31.7 (1/4) | fail | -26.9 (0/4) | fail |
| **LSTM** (new) | -3.0 (0/4) | fail | -9.7 (0/4) | fail | -32.0 (0/4) | fail | -22.6 (1/4) | fail |
| **RNN** (new) | -4.7 (0/4) | fail | -14.7 (0/4) | fail | -11.4 (1/4) | fail | -23.4 (1/4) | fail |

#### flux loss (`flux_pct`) — unseen train

*No change* MAE, percentage points: 1 d 2.11, 3 d 3.07, 7 d 3.48, 14 d 4.31

| method | 1 d: skill % (blocks > 0) | bar | 3 d: skill % (blocks > 0) | bar | 7 d: skill % (blocks > 0) | bar | 14 d: skill % (blocks > 0) | bar |
|---|---|---|---|---|---|---|---|---|
| smoothed no change | -15.9 (0/4) | fail | +1.4 (2/4) | fail | +2.7 (2/4) | fail | +7.1 (3/4) | **PASS** |
| trend | -3.1 (0/4) | fail | -8.9 (0/4) | fail | -22.1 (0/4) | fail | -27.3 (0/4) | fail |
| fouling law | -44.9 (0/4) | fail | -13.6 (1/4) | fail | -7.9 (2/4) | fail | +1.0 (1/4) | fail |
| XGBoost | -18.3 (0/4) | fail | -13.6 (1/4) | fail | -17.9 (2/4) | fail | +1.7 (2/4) | fail |
| hybrid | -41.0 (0/4) | fail | -27.0 (1/4) | fail | -31.1 (1/4) | fail | -6.3 (1/4) | fail |
| **Prophet** (new) | -91.4 (0/4) | fail | -33.4 (1/4) | fail | -21.8 (1/4) | fail | -3.6 (2/4) | fail |
| **Prophet + level shift** (new) | -54.4 (0/4) | fail | -10.4 (2/4) | fail | -1.8 (2/4) | fail | +10.0 (2/4) | fail |
| **LSTM** (new) | -6.4 (0/4) | fail | -11.6 (1/4) | fail | -23.9 (0/4) | fail | -24.2 (0/4) | fail |
| **RNN** (new) | -6.4 (0/4) | fail | -9.1 (0/4) | fail | -24.7 (0/4) | fail | -20.4 (0/4) | fail |

### Every horizon in detail

**ΔP, 1 d ahead, rolling origin** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 1.76 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.13 | -20.8 | -27.7 | -52.0 | -14.1 | +1.7 | -0.35 | fail |
| trend | 1.84 | -4.5 | -6.6 | -5.2 | -7.1 | -1.1 | -0.09 | fail |
| fouling law | 2.62 | -48.5 | -68.1 | -50.0 | -62.2 | -27.3 | -1.01 | fail |
| XGBoost | 2.04 | -15.7 | -20.6 | -6.8 | -53.0 | +5.8 | -0.44 | fail |
| hybrid | 2.65 | -50.4 | -76.2 | -68.9 | -73.5 | -6.7 | -1.06 | fail |
| **Prophet** (new) | 7.23 | -309.4 | -690.5 | -159.4 | -297.1 | -219.0 | -13.57 | fail |
| **Prophet + level shift** (new) | 3.39 | -91.9 | -120.3 | -79.9 | -148.6 | -46.7 | -2.18 | fail |
| **LSTM** (new) | 1.88 | -6.4 | -5.0 | -4.2 | +2.2 | -14.6 | -0.12 | fail |
| **RNN** (new) | 1.79 | -1.6 | -9.6 | -3.1 | +3.9 | +0.2 | -0.03 | fail |

**ΔP, 3 d ahead, rolling origin** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.83 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.76 | +2.2 | +8.7 | -5.6 | +0.9 | +8.9 | 0.04 | fail |
| trend | 3.18 | -12.5 | -14.2 | -12.3 | -19.3 | -4.4 | -0.26 | fail |
| fouling law | 3.00 | -6.3 | -13.3 | +12.5 | -15.9 | -17.1 | -0.10 | fail |
| XGBoost | 2.85 | -0.8 | -3.8 | +6.4 | -8.5 | -0.6 | -0.01 | fail |
| hybrid | 3.86 | -36.6 | -27.7 | +4.7 | -42.0 | -99.2 | -0.90 | fail |
| **Prophet** (new) | 7.51 | -165.7 | -346.7 | -27.7 | -171.3 | -190.4 | -5.66 | fail |
| **Prophet + level shift** (new) | 3.61 | -27.6 | -36.2 | +9.1 | -67.7 | -32.6 | -0.49 | fail |
| **LSTM** (new) | 3.16 | -11.8 | -13.3 | -0.7 | -29.8 | -8.4 | -0.20 | fail |
| **RNN** (new) | 2.96 | -4.9 | -14.9 | +1.5 | -6.1 | -3.8 | -0.07 | fail |

**ΔP, 7 d ahead, rolling origin** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.91 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.02 | — |
| smoothed no change | 2.86 | +1.8 | -3.8 | +5.0 | -2.8 | +6.0 | 0.07 | fail |
| trend | 3.68 | -26.4 | -45.2 | -29.6 | -31.7 | -4.7 | -0.61 | fail |
| fouling law | 3.39 | -16.2 | -76.5 | +9.7 | -10.9 | -25.9 | -0.17 | fail |
| XGBoost | 3.18 | -9.3 | -42.6 | +2.2 | +1.4 | -19.9 | -0.12 | fail |
| hybrid | 3.64 | -24.9 | -111.6 | +4.5 | -24.2 | -18.8 | -0.32 | fail |
| **Prophet** (new) | 7.91 | -171.6 | -583.9 | -29.7 | -150.6 | -166.6 | -5.31 | fail |
| **Prophet + level shift** (new) | 3.84 | -31.7 | -123.8 | +12.8 | -49.1 | -23.2 | -0.46 | fail |
| **LSTM** (new) | 3.10 | -6.5 | -59.2 | +0.4 | +9.5 | -5.3 | -0.08 | fail |
| **RNN** (new) | 3.26 | -11.8 | -101.9 | +0.6 | +5.7 | +1.3 | -0.14 | fail |

**ΔP, 14 d ahead, rolling origin** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.49 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.07 | — |
| smoothed no change | 3.36 | +3.6 | +9.9 | +1.5 | +2.4 | +4.0 | 0.02 | fail |
| trend | 5.34 | -53.2 | -72.5 | -79.3 | -42.1 | -18.3 | -1.52 | fail |
| fouling law | 4.20 | -20.4 | -91.8 | -6.7 | +5.8 | -34.2 | -0.43 | fail |
| XGBoost | 4.00 | -14.6 | -77.0 | -7.9 | +9.4 | -20.7 | -0.43 | fail |
| hybrid | 4.61 | -32.2 | -93.2 | -32.8 | -7.7 | -28.6 | -1.12 | fail |
| **Prophet** (new) | 8.75 | -151.0 | -404.4 | -57.1 | -134.8 | -131.7 | -5.43 | fail |
| **Prophet + level shift** (new) | 4.43 | -26.9 | -78.6 | -16.1 | -18.1 | -20.1 | -0.58 | fail |
| **LSTM** (new) | 3.87 | -10.9 | -53.0 | -20.2 | +22.9 | -25.3 | -0.21 | fail |
| **RNN** (new) | 4.10 | -17.5 | -72.5 | -18.7 | +20.4 | -41.8 | -0.35 | fail |

**flux loss, 1 d ahead, rolling origin** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.11 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.44 | -15.9 | -24.9 | -42.8 | -8.0 | -8.7 | -0.20 | fail |
| trend | 2.17 | -3.1 | -3.8 | -4.5 | -2.6 | -2.8 | -0.05 | fail |
| fouling law | 3.05 | -44.9 | -34.6 | -79.0 | -45.0 | -29.7 | -0.32 | fail |
| XGBoost | 2.36 | -12.1 | -72.2 | -20.1 | -15.0 | +4.9 | 0.08 | fail |
| hybrid | 2.71 | -28.7 | -110.0 | -41.6 | -26.0 | -11.4 | -0.11 | fail |
| **Prophet** (new) | 4.03 | -91.4 | -274.7 | -68.5 | -58.6 | -103.6 | -0.80 | fail |
| **Prophet + level shift** (new) | 3.25 | -54.4 | -52.1 | -60.1 | -61.7 | -44.9 | -0.52 | fail |
| **LSTM** (new) | 2.23 | -5.9 | -29.6 | -5.4 | -1.9 | -6.0 | 0.05 | fail |
| **RNN** (new) | 2.23 | -5.9 | -47.3 | -5.4 | -2.5 | -2.4 | 0.07 | fail |

**flux loss, 3 d ahead, rolling origin** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.07 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.02 | +1.4 | +10.0 | -7.4 | +8.3 | -0.1 | 0.05 | fail |
| trend | 3.34 | -8.9 | -4.5 | -12.0 | -9.0 | -7.5 | -0.16 | fail |
| fouling law | 3.48 | -13.6 | +3.3 | -15.5 | -26.6 | -3.6 | 0.00 | fail |
| XGBoost | 3.15 | -2.7 | -81.4 | +13.0 | -8.3 | +5.9 | 0.13 | fail |
| hybrid | 3.58 | -16.9 | -100.2 | +1.4 | -37.6 | +4.5 | -0.09 | fail |
| **Prophet** (new) | 4.09 | -33.4 | -144.6 | +11.7 | -35.7 | -42.8 | -0.12 | fail |
| **Prophet + level shift** (new) | 3.39 | -10.4 | +9.3 | +12.9 | -39.2 | -4.5 | 0.01 | fail |
| **LSTM** (new) | 3.14 | -2.5 | -46.8 | +4.5 | -9.7 | +7.3 | 0.12 | fail |
| **RNN** (new) | 3.21 | -4.8 | -60.0 | -1.9 | -6.0 | +4.5 | 0.14 | fail |

**flux loss, 7 d ahead, rolling origin** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.48 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.39 | +2.7 | -13.3 | +8.0 | -5.3 | +7.7 | 0.04 | fail |
| trend | 4.25 | -22.1 | -27.0 | -17.5 | -26.8 | -21.0 | -0.40 | fail |
| fouling law | 3.75 | -7.9 | -28.7 | +1.9 | -35.8 | +13.4 | 0.06 | fail |
| XGBoost | 3.68 | -5.8 | -196.2 | +19.0 | -22.4 | +11.0 | 0.15 | fail |
| hybrid | 4.13 | -18.7 | -237.8 | +20.0 | -60.0 | +13.6 | -0.10 | fail |
| **Prophet** (new) | 4.24 | -21.8 | -250.6 | +22.9 | -39.7 | -17.2 | 0.03 | fail |
| **Prophet + level shift** (new) | 3.54 | -1.8 | -31.1 | +25.7 | -44.5 | +19.1 | 0.15 | fail |
| **LSTM** (new) | 3.84 | -10.3 | -155.9 | +14.3 | -18.8 | -6.6 | 0.12 | fail |
| **RNN** (new) | 3.85 | -10.7 | -162.0 | +1.1 | -7.0 | -6.8 | 0.14 | fail |

**flux loss, 14 d ahead, rolling origin** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 4.31 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 4.00 | +7.1 | -0.1 | +16.4 | +3.5 | +9.3 | 0.11 | **PASS** |
| trend | 5.48 | -27.3 | -25.2 | -50.8 | -17.3 | -28.6 | -0.73 | fail |
| fouling law | 4.26 | +1.0 | -9.9 | -4.4 | -2.5 | +19.2 | 0.09 | fail |
| XGBoost | 4.15 | +3.6 | -23.7 | +27.3 | -2.4 | +6.2 | 0.13 | fail |
| hybrid | 4.45 | -3.3 | -25.3 | +10.4 | -13.2 | +15.7 | -0.01 | fail |
| **Prophet** (new) | 4.46 | -3.6 | -49.4 | +20.9 | +1.0 | -17.8 | 0.15 | fail |
| **Prophet + level shift** (new) | 3.88 | +10.0 | -1.1 | +29.4 | -1.6 | +22.6 | 0.22 | fail |
| **LSTM** (new) | 5.04 | -16.9 | -46.6 | +9.2 | -1.7 | -64.1 | -0.06 | fail |
| **RNN** (new) | 4.82 | -11.8 | -39.5 | +8.9 | -4.0 | -37.6 | -0.03 | fail |

**ΔP, 1 d ahead, unseen train** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 1.76 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.13 | -20.8 | -27.7 | -52.0 | -14.1 | +1.7 | -0.35 | fail |
| trend | 1.84 | -4.5 | -6.6 | -5.2 | -7.1 | -1.1 | -0.09 | fail |
| fouling law | 2.62 | -48.5 | -68.1 | -50.0 | -62.2 | -27.3 | -1.01 | fail |
| XGBoost | 2.17 | -22.9 | -19.5 | -14.2 | -63.0 | -4.2 | -0.64 | fail |
| hybrid | 3.14 | -78.0 | -112.1 | -100.8 | -115.8 | -16.6 | -2.00 | fail |
| **Prophet** (new) | 7.23 | -309.4 | -690.5 | -159.4 | -297.1 | -219.0 | -13.57 | fail |
| **Prophet + level shift** (new) | 3.39 | -91.9 | -120.3 | -79.9 | -148.6 | -46.7 | -2.18 | fail |
| **LSTM** (new) | 1.82 | -3.0 | -3.8 | -5.6 | -2.2 | -1.3 | -0.06 | fail |
| **RNN** (new) | 1.85 | -4.7 | -13.7 | -5.0 | -1.9 | -1.3 | -0.09 | fail |

**ΔP, 3 d ahead, unseen train** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.83 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.76 | +2.2 | +8.7 | -5.6 | +0.9 | +8.9 | 0.04 | fail |
| trend | 3.18 | -12.5 | -14.2 | -12.3 | -19.3 | -4.4 | -0.26 | fail |
| fouling law | 3.00 | -6.3 | -13.3 | +12.5 | -15.9 | -17.1 | -0.10 | fail |
| XGBoost | 3.18 | -12.4 | -22.5 | +0.5 | -14.5 | -19.6 | -0.28 | fail |
| hybrid | 4.48 | -58.7 | -59.0 | +0.2 | -100.5 | -101.6 | -1.66 | fail |
| **Prophet** (new) | 7.51 | -165.7 | -346.7 | -27.7 | -171.3 | -190.4 | -5.66 | fail |
| **Prophet + level shift** (new) | 3.61 | -27.6 | -36.2 | +9.1 | -67.7 | -32.6 | -0.49 | fail |
| **LSTM** (new) | 3.10 | -9.7 | -15.1 | -4.5 | -9.3 | -12.7 | -0.17 | fail |
| **RNN** (new) | 3.24 | -14.7 | -29.9 | -1.3 | -28.6 | -5.7 | -0.38 | fail |

**ΔP, 7 d ahead, unseen train** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.91 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.02 | — |
| smoothed no change | 2.86 | +1.8 | -3.8 | +5.0 | -2.8 | +6.0 | 0.07 | fail |
| trend | 3.68 | -26.4 | -45.2 | -29.6 | -31.7 | -4.7 | -0.61 | fail |
| fouling law | 3.39 | -16.2 | -76.5 | +9.7 | -10.9 | -25.9 | -0.17 | fail |
| XGBoost | 3.74 | -28.5 | -161.4 | +1.5 | -2.6 | -26.4 | -0.73 | fail |
| hybrid | 4.87 | -67.3 | -298.3 | +2.2 | -46.0 | -61.2 | -1.46 | fail |
| **Prophet** (new) | 7.91 | -171.6 | -583.9 | -29.7 | -150.6 | -166.6 | -5.31 | fail |
| **Prophet + level shift** (new) | 3.84 | -31.7 | -123.8 | +12.8 | -49.1 | -23.2 | -0.46 | fail |
| **LSTM** (new) | 3.85 | -32.0 | -215.5 | -0.6 | -2.2 | -7.4 | -1.03 | fail |
| **RNN** (new) | 3.24 | -11.4 | -74.0 | +0.7 | -1.2 | -5.0 | -0.14 | fail |

**ΔP, 14 d ahead, unseen train** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.49 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.07 | — |
| smoothed no change | 3.36 | +3.6 | +9.9 | +1.5 | +2.4 | +4.0 | 0.02 | fail |
| trend | 5.34 | -53.2 | -72.5 | -79.3 | -42.1 | -18.3 | -1.52 | fail |
| fouling law | 4.20 | -20.4 | -91.8 | -6.7 | +5.8 | -34.2 | -0.43 | fail |
| XGBoost | 4.34 | -24.6 | -100.3 | -9.1 | +0.9 | -36.7 | -0.73 | fail |
| hybrid | 5.50 | -57.9 | -143.3 | -26.1 | -43.7 | -67.4 | -1.92 | fail |
| **Prophet** (new) | 8.75 | -151.0 | -404.4 | -57.1 | -134.8 | -131.7 | -5.43 | fail |
| **Prophet + level shift** (new) | 4.43 | -26.9 | -78.6 | -16.1 | -18.1 | -20.1 | -0.58 | fail |
| **LSTM** (new) | 4.27 | -22.6 | -100.5 | -18.4 | +15.4 | -37.9 | -0.51 | fail |
| **RNN** (new) | 4.30 | -23.4 | -110.8 | -18.6 | +15.0 | -33.3 | -0.53 | fail |

**flux loss, 1 d ahead, unseen train** (n = 642)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 2.11 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 2.44 | -15.9 | -24.9 | -42.8 | -8.0 | -8.7 | -0.20 | fail |
| trend | 2.17 | -3.1 | -3.8 | -4.5 | -2.6 | -2.8 | -0.05 | fail |
| fouling law | 3.05 | -44.9 | -34.6 | -79.0 | -45.0 | -29.7 | -0.32 | fail |
| XGBoost | 2.49 | -18.3 | -87.2 | -33.9 | -16.2 | -0.9 | 0.02 | fail |
| hybrid | 2.97 | -41.0 | -120.2 | -69.7 | -33.0 | -21.3 | -0.26 | fail |
| **Prophet** (new) | 4.03 | -91.4 | -274.7 | -68.5 | -58.6 | -103.6 | -0.80 | fail |
| **Prophet + level shift** (new) | 3.25 | -54.4 | -52.1 | -60.1 | -61.7 | -44.9 | -0.52 | fail |
| **LSTM** (new) | 2.24 | -6.4 | -42.1 | -8.3 | -2.0 | -3.7 | 0.06 | fail |
| **RNN** (new) | 2.24 | -6.4 | -47.8 | -7.9 | -2.6 | -2.5 | 0.06 | fail |

**flux loss, 3 d ahead, unseen train** (n = 617)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.07 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.02 | +1.4 | +10.0 | -7.4 | +8.3 | -0.1 | 0.05 | fail |
| trend | 3.34 | -8.9 | -4.5 | -12.0 | -9.0 | -7.5 | -0.16 | fail |
| fouling law | 3.48 | -13.6 | +3.3 | -15.5 | -26.6 | -3.6 | 0.00 | fail |
| XGBoost | 3.48 | -13.6 | -154.9 | +5.4 | -13.1 | -1.5 | 0.06 | fail |
| hybrid | 3.89 | -27.0 | -185.0 | -8.2 | -42.8 | +3.3 | -0.15 | fail |
| **Prophet** (new) | 4.09 | -33.4 | -144.6 | +11.7 | -35.7 | -42.8 | -0.12 | fail |
| **Prophet + level shift** (new) | 3.39 | -10.4 | +9.3 | +12.9 | -39.2 | -4.5 | 0.01 | fail |
| **LSTM** (new) | 3.42 | -11.6 | -48.8 | +1.9 | -25.7 | -1.3 | 0.02 | fail |
| **RNN** (new) | 3.35 | -9.1 | -67.3 | -2.9 | -9.3 | -2.6 | 0.06 | fail |

**flux loss, 7 d ahead, unseen train** (n = 572)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 3.48 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 3.39 | +2.7 | -13.3 | +8.0 | -5.3 | +7.7 | 0.04 | fail |
| trend | 4.25 | -22.1 | -27.0 | -17.5 | -26.8 | -21.0 | -0.40 | fail |
| fouling law | 3.75 | -7.9 | -28.7 | +1.9 | -35.8 | +13.4 | 0.06 | fail |
| XGBoost | 4.10 | -17.9 | -437.6 | +21.9 | -26.8 | +5.2 | 0.02 | fail |
| hybrid | 4.56 | -31.1 | -449.5 | +16.4 | -54.4 | -0.9 | -0.21 | fail |
| **Prophet** (new) | 4.24 | -21.8 | -250.6 | +22.9 | -39.7 | -17.2 | 0.03 | fail |
| **Prophet + level shift** (new) | 3.54 | -1.8 | -31.1 | +25.7 | -44.5 | +19.1 | 0.15 | fail |
| **LSTM** (new) | 4.31 | -23.9 | -180.7 | -2.0 | -41.4 | -7.4 | -0.08 | fail |
| **RNN** (new) | 4.34 | -24.7 | -184.5 | -12.6 | -32.2 | -9.1 | -0.03 | fail |

**flux loss, 14 d ahead, unseen train** (n = 502)

| method | MAE, pp | skill % | block 1 | block 2 | block 3 | block 4 | R² of the change | bar |
|---|---|---|---|---|---|---|---|---|
| no change | 4.31 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.00 | — |
| smoothed no change | 4.00 | +7.1 | -0.1 | +16.4 | +3.5 | +9.3 | 0.11 | **PASS** |
| trend | 5.48 | -27.3 | -25.2 | -50.8 | -17.3 | -28.6 | -0.73 | fail |
| fouling law | 4.26 | +1.0 | -9.9 | -4.4 | -2.5 | +19.2 | 0.09 | fail |
| XGBoost | 4.24 | +1.7 | -24.1 | +21.0 | -2.2 | +3.0 | 0.12 | fail |
| hybrid | 4.58 | -6.3 | -45.1 | +19.7 | -12.4 | -0.5 | -0.04 | fail |
| **Prophet** (new) | 4.46 | -3.6 | -49.4 | +20.9 | +1.0 | -17.8 | 0.15 | fail |
| **Prophet + level shift** (new) | 3.88 | +10.0 | -1.1 | +29.4 | -1.6 | +22.6 | 0.22 | fail |
| **LSTM** (new) | 5.35 | -24.2 | -81.4 | -3.2 | -2.5 | -69.4 | -0.13 | fail |
| **RNN** (new) | 5.19 | -20.4 | -103.4 | -10.3 | -5.5 | -28.3 | -0.06 | fail |

## Run time

- **Machine:** two cores, shared with another job (load average about 4–6 throughout). The new methods ran at lower priority
  (nice 10), so these times are inflated.
- **Prophet:** 938.7 s for 2,584 fits (921.7 s of fitting), 23:32:11 to 23:48:19 UTC.
- **LSTM + RNN:** 64 folds × 2 indicators × 2 networks × 5 seeds = 1,280 seed-fits, each with a stage 1 and a stage 2. Two
  processes ran side by side, taking 2,798 s and 2,393 s (14–223 s per fold), 23:32:14 to 00:18:57 UTC.
- **Totals:** all new methods took 46 min 46 s of wall time; the reproduction check took 170 s.

## What went wrong, and doubts

1. **The first reproduction run stalled.** It ran for about 24 minutes before I stopped it; the saved phase-3 run made about
   eight times as many XGBoost fits in 478 s. The most likely cause: XGBoost's two threads spin-wait while the second core is
   busy with the other job. I re-ran it with `OMP_WAIT_POLICY=PASSIVE`, which changes only how idle threads wait. The re-run
   matched the saved forecasts within 1e-16 in 170 s. No new method had run at that point.
2. **A check in `score.py` failed on its first run; no number was affected.** The comparison with the saved verdicts read
   *no change*'s empty verdict as a mismatch (pandas 3 keeps a missing value missing when converting it to text). I fixed the
   comparison, no number changed, and the check then passed.
3. **Prophet as approved cannot follow a cleaning.** A Prophet changepoint changes the slope of a continuous trend, while the
   indicators drop at a cleaning. This is the likely reason for the approved form's large losses on ΔP (−151 to −309 %). The
   variant, with a step at each cleaning, did far better (−27 to −92 % on ΔP), but it is my declared addition to the approved
   specification, not the specification itself. Its flux 14-day +10.0 % is the only new result above +5 % overall. It still
   fails on the blocks, and it is one of 64 new verdicts.
4. **The networks learned little.**
   - In 332 of the 1,280 seed-fits (25.9 %) the validation error was lowest after the first or second epoch. The median E* was
     9–10 epochs, and 6 seed-fits reached the 150-epoch cap.
   - The bar judges the five-seed average, but single seeds spread widely; for example, ΔP 1 day ahead, rolling origin, LSTM:
     −43.0 to −0.2 %. Of the 160 single-seed results (5 seeds × 2 networks × 2 indicators × 4 horizons × 2 tests), only one beat
     *no change* at all (+0.4 %), and none reached +5 %.
   - With the same information as XGBoost, the LSTM's skill was above XGBoost's in 10 of 16 cells and the RNN's in 7 of 16, but
     all of them stayed below *no change*.
   - At 7 and 14 days, block 1 (days 550–599) was the networks' worst block in 15 of 16 cases.
5. **Details the addendum did not spell out** (fixed in the code before the run; no declared setting is affected):
   - Before a train's first record (only rows with *t* < 16), the sequence repeats the first record.
   - An input with no spread in a fold (standard deviation below 1e-12) is divided by 1 instead.
   - Prophet's day 0 is 1 January 2000.
6. **Not covered:**
   - The 90 % ranges and the early warning were not run for the new methods (as declared).
   - A network on the four indicators alone, without XGBoost's other eight inputs, was not tested.
   - The networks' five-seed means would also move from run to run. How much was not measured: there was one run per seed.
     Phase 3 found up to 2.7 points (rolling origin) and 7.1 points (unseen train) of run-to-run spread for XGBoost from row
     order alone.
7. **Folder timestamp:** the `phase3` folder's own modification time changed because `addition/` was created inside it. No
   file in it changed.

## Files (all in `phase3/addition/`)

- **Declaration:** `declaration_addendum.md` and `declaration_addendum.sha256`, written and checksummed before the run.
- **Harness:** `p3add.py` — the data check, phase 3's tables and folds via `p3lib`, and the networks' inputs and fitting.
- **Reproduction check:** `reproduce.py` → `reproduce_check.txt`.
- **Prophet:** `run_prophet.py` → `pred_prophet.csv`, `prophet_log.json`, `run_prophet.log`.
- **Networks:** `run_nets.py` →
  - `nets_parts/` (per fold: forecasts, per-seed changes, training log);
  - `pred_nets_rolling.csv`, `pred_nets_unseen_train.csv`, `pred_nets_seeds.csv`, `nets_log.csv`;
  - `run_nets_part0.log`, `run_nets_part1.log`, `collect.log`.
- **Scores:** `score.py` → `scores_rolling_addition.csv`, `scores_unseen_train_addition.csv`, `summary_addition.json`,
  `report_tables.md`, `score.log`.
- **Diagnostics:** `diag.py` → `diag.txt`, `seed_spread.csv` (descriptive, not part of the test).
- **Pre-addendum tool checks:** `tool_checks/`, the two checks run before the addendum (made-up and random numbers only) and
  their outputs.

**To re-run:**

1. `python3 -B reproduce.py`
2. `<python with prophet 1.5.0> -B run_prophet.py`
3. `python3 -B run_nets.py` (or `run_nets.py 0 2` and `run_nets.py 1 2` side by side), then `python3 -B run_nets.py collect`
4. `python3 -B score.py`
5. `python3 -B diag.py`

**Versions:** numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1, torch 2.14.1, prophet 1.5.0 (cmdstanpy 1.3.0).
Prophet was installed in a separate virtual environment layered on the system packages (`venv_prophet`), so the
packages that reproduce phase 3 were not changed.
