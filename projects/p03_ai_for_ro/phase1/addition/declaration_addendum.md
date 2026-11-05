# Phase 3 — addition: three more forecasting methods, declared before they are run

Written 2 November 2026, 02:00 (UTC+3) — 1 November 2026, 23:00 UTC — before any of the new models was trained or scored
and before any phase-3 code was run again for this addition. Run before writing it, none of it on the study data: installing
Prophet; a check that Prophet recovers a known straight line with one step, on made-up numbers; a timing check of the two
small networks on random numbers. Nothing here will be changed after seeing results; if something must change, it goes into
a dated section "Changes after the run", with its reason, below the original text.

**Phase 3's declared results stay exactly as they are.** This addition edits no file of phase 3 outside `phase3/addition/`.
The six declared methods' numbers are quoted from `phase3/out/` as saved; the addition re-runs some of them only to check
that it uses the same folds and the same score, and those re-runs replace nothing. The new methods are a declared addition,
judged by exactly the same test and the same bar; whatever they show, the phase-3 verdicts are not changed by them.

## Why

The project owner approved adding Prophet and two deep-learning methods, an LSTM and a plain RNN. Background only: in the
earlier, separate test (`fouling_prediction_two_ways`, 7 days ahead, an LSTM on the last 28 days) the LSTM was +25.5 % (ΔP)
and +28.0 % (flux) better than *no change* on a random split, and −7.0 % and −90.0 % on a future split
(`fouling7/results.json`); the phase-3 declaration dropped the LSTM. This addition puts the deep-learning methods, and
Prophet, through phase 3's own test.

## What stays the same (from `declaration.md` and the code that ran it)

- **Pairs, inputs, targets, horizons, blocks:** exactly phase 3's, built by `build_days` in `phase3/p3lib.py`, imported
  unchanged: (train, day *t*) → the same train on day *t + h*, h = 1, 3, 7, 14; both days must have data and no ledger
  cleaning may fall in (t, t + h]; targets `dp_pct` and `flux_pct`; test blocks by day *t*: 550–599, 600–649, 650–699,
  700–771. Every forecast is clipped to 0–1.
- **Folds:** built by the same lines as `phase3/run.py` — rolling origin: `train = P[P.t + h < start].sort_values(["t", "train"])`,
  test = the block's pairs; unseen train: `train[train.train != k]`, test = train *k*'s pairs in the block.
- **Scores and bar:** `scores` and `verdict` from `phase3/p3lib.py`, imported unchanged, on the same test pairs, against the same
  *no change* forecasts.

The test and the bar, copied from `declaration.md`:

> ## Tests
>
> 1. **Future, rolling origin:** four test blocks by day *t* (table above). Each block's models train on all pairs whose target
>    day *t + h* falls before the block starts — all trains, all earlier days.
> 2. **An unseen train, in the future:** for each held-out train and each block, train on the other two trains (target day before
>    the block) and test on the held-out train in that block.
>
> ## Scores and what counts as success
>
> - **MAE** (percentage points of the indicator) and **skill** = 1 − MAE / MAE(no change); positive skill beats no change.
> - **R² of the change** (t → t + h), not of the level: a level R² looks high for any method that copies today's value.
> - A method **beats no change** at a horizon only if its skill over all test blocks is at least **+5 %** and it is positive in at
>   least **3 of the 4 blocks**. Both tests are reported; the unseen-train test is the stricter one.

## The new methods

### 7. Prophet (as approved)

- **One model per train, indicator and forecast origin** *t*; the four horizons are read from the same fit.
- **Data:** the train's daily records with day ≤ *t*, from its first record — what is known at the origin, the same information
  the other methods have (*no change* uses day *t*'s value). Day 0 is mapped to 1 January 2000; only the spacing matters.
- **Changepoints:** exactly the train's cleaning days known at the origin — the ledger days ≤ *t*, the same cleaning days every
  other method uses — and no automatic changepoints. In Prophet a changepoint changes the trend's slope; the trend stays continuous.
- **Seasonality:** weekly on (Prophet's default for daily data); yearly off — at most origins the record spans less than two years
  (Prophet's own rule would switch it on only from about day 730, changing the model inside block 4) and the first 300 days hold
  only 50–53 records per train; daily off (daily data).
- **Everything else Prophet's defaults** (version 1.5.0): linear growth, changepoint prior scale 0.05, seasonality prior scale 10,
  additive, fitted by maximum a posteriori. Point forecasts only (no uncertainty sampling).
- **Forecast:** Prophet's `yhat` for day *t + h*, clipped to 0–1. If a fit fails, that origin's forecasts fall back to *no change*;
  the number of such cases is reported.
- Prophet uses only the forecast train's own history, so, like the four methods that need no training, its forecasts are the
  same in both tests.

### 7b. Prophet with a level shift at each cleaning (declared variant)

The same as 7, plus one step input for each cleaning before the origin (ledger day *c* < *t*): 0 up to day *c*, 1 after it,
entered as an extra regressor (not standardised; prior scale 10, Prophet's default for regressors). At each cleaning the trend can
then change its slope (the changepoint) and drop in level (the step). Reason: a Prophet changepoint cannot drop the trend, while a
cleaning lowers the ΔP and flux-loss indicators within a day. In the check on made-up numbers (a straight line with one step),
the approved form forecast 0.224 one day ahead against a true 0.260, the variant 0.258. Both are judged by the bar; the approved
form (7) stays the main Prophet result, and the variant is reported as a variant.

### 8. LSTM and 9. plain RNN (Elman)

**Inputs — every XGBoost input reaches the networks, all known on day *t*; no train label:**

- **A sequence of the last 17 days, *t* − 16 … *t*** — the span XGBoost's inputs reach (its 14-day lag, with the 2-day tolerance).
  Six values per day: the four daily indicators (`dp_pct`, `flux_pct`, `sp_pct`, `ndp_pct`), carried forward from the latest
  earlier record on a day without one; a flag "record on this day"; and a flag "day in the current cleaning cycle" (after the last
  ledger cleaning known at *t*; before the first cleaning, from the first record). XGBoost's 20 indicator values (today and 1, 3,
  7 and 14 days back) lie inside this window.
- **XGBoost's other 8 inputs, entered next to the sequence's summary:** days since the last cleaning (missing before the first
  cleaning: days since the first record, plus a flag), the number of 10-min samples behind today's value, the 14-day slopes of ΔP
  and flux (missing: 0, plus a flag), the smoothed ΔP and flux states, and the fouling-law forecasts of ΔP and flux for *t + h*
  (missing: today's value, plus a flag) — 8 values and 3 flags. XGBoost sees the missing values as missing; the flags carry the same
  information.
- Every input is standardised with the mean and standard deviation of the rows the network is fitted on (training rows only).

**Target:** the change of the indicator from *t* to *t + h*, as for XGBoost, standardised with the fitted rows' mean and standard
deviation. Forecast = clip(today's value + the mean of the five seeds' changes, 0, 1), as for XGBoost.

**Networks (PyTorch 2.14.1, CPU, one thread):**

- **LSTM:** one LSTM layer of 32 units; its state after day *t*, joined with the 11 other inputs, goes to a dense layer of 16 units
  (ReLU) and one output — 5,841 weights.
- **RNN:** the same with one plain Elman layer of 32 units (tanh) in place of the LSTM — 2,001 weights.
- No dropout; PyTorch's default starting weights.

**Training, per fold, per seed, per network:**

- **Rows:** exactly the fold's XGBoost training rows.
- **Validation slice, from the training rows only:** with the rows sorted by (t, train) as in `run.py`, the slice starts at the day
  *t* of the row at 80 % of them (position rounded) and holds every training row from that day on. In stage 1 the network is
  fitted on the training rows whose target day *t + h* is before that day, so no fitted target falls inside the slice.
- **Stage 1:** Adam, learning rate 0.001, weight decay 0.0001; batches of 64, reshuffled every epoch; loss: mean squared error of
  the standardised change (XGBoost's objective is squared error too); gradient norm clipped at 1.0. After each epoch, the MAE of the
  level forecast on the validation slice; training stops after 15 epochs without a new lowest MAE, or at 150 epochs. E* = the epoch
  with the lowest validation MAE.
- **Stage 2:** the same network with the same seed (the same starting weights), refitted on all the fold's training rows for E*
  epochs, standardisation recomputed on them. This network makes the test forecasts: it learns from the same rows as XGBoost, and
  the validation slice only sets the number of epochs.
- **Seeds:** 0, 1, 2, 3, 4 — XGBoost's five. A seed sets the starting weights and the batch order. The forecast is the mean of the
  five seeds' changes.

These settings were chosen before any run, from library defaults and the earlier project LSTM (32 units, a 16-unit dense layer,
Adam, weight decay 0.0001, early stopping), not from any test result. There is no tuning.

## Check before the new methods are run

With the addition's own harness, reproduce from `phase3/out/`: (a) the *no change* MAE for every indicator and horizon in both
tests; (b) the XGBoost and hybrid forecasts for ΔP 7 days ahead (rolling origin, all four blocks; unseen train, every block and
held-out train) and for flux 14 days ahead (rolling origin, all four blocks), against `pred_rolling.csv` and
`pred_unseen_train.csv`, and their skills against `scores_rolling.csv` and `scores_unseen_train.csv`. It passes if the test pairs
are the same, the *no change* MAE agrees to 1e-9, the forecasts to 1e-6 and the skills to 0.01 points (same library versions as the
saved run: numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1). If it fails, the new methods are not run until the
difference is found and reported.

## Not part of this addition

- The 90 % ranges and the early warning are not run for the new methods: the addition answers question 1 (forecast skill) only.
- The six declared methods are not re-scored or re-ranked; their numbers in the report are the saved ones.

## Reported whatever the result

Every new method (7, 7b, 8, 9), indicator, horizon, block and test: MAE, skill, R² of the change, skill per block and pass or fail
at the bar, next to the six declared methods' saved numbers; run time; anything that failed (Prophet fits that fell back to
*no change*, networks that stopped early). A caution, stated now: four new rows × four horizons × two tests × two indicators make 64
more verdicts, so a single pass among them is weak evidence on its own.
