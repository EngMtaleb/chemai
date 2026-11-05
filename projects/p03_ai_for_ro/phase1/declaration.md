# Phase 3 — RO fouling forecast, re-tested: what we will test, declared before any model is run

Written 28 October 2026, before any phase-3 model was trained or scored. The tables of pairs and events below were built
first (`prep.py`); they contain no model output. Nothing here will be changed after seeing results; any later change will
be listed as a change, with its reason, next to the original.

## The questions

1. **Forecast:** can any method forecast a train's fouling indicators 1, 3, 7 and 14 days ahead better than assuming *no change*?
2. **Uncertainty:** does a 90 % range around the forecast hold — does the truth fall inside it about 90 % of the time on days the
   model never saw?
3. **Early warning:** can a forecast see the normalised feed-channel ΔP crossing +15 % before it happens, and how often does it
   warn for nothing?
4. **State versus forecast:** how much of any gain comes from reading *today's* state better (smoothing a noisy daily value)
   rather than from predicting the change?

## Data

Su, Meng & Chang (2026), *Membranes* 16(8) 257; deposit doi:10.5281/zenodo.21055948 (CC BY 4.0): one seawater RO plant,
3 trains (A, B, C), day 0–771, 18 cleanings (CIP). Daily indicators, each a fractional rise from the commissioning baseline
(never reset at a cleaning): `dp_pct` (normalised feed-channel ΔP), `flux_pct` (loss of 25 °C specific flux), `sp_pct`
(25 °C salt passage), `ndp_pct` (net driving pressure). The two files are checked against the deposit's SHA-256 list.
The deposit holds no raw 10-min data, no feed quality (SDI, turbidity), no temperature and no operating flux: every method
learns only from the indicators' own history and the cleaning days.

## Forecasting pairs

- A pair is (train, day *t*) → the same train on day *t + h*, for h = 1, 3, 7, 14 days. Both days must have data, and no
  cleaning may fall in (t, t + h]: we forecast fouling, not the operators' decision to clean.
- Inputs use only what is known on day *t*: today's four indicators; their values 1, 3, 7 and 14 days back (nearest earlier day
  within 2 days); the 14-day slope of ΔP and of flux inside the current cleaning cycle (≥ 4 points); days since the last
  cleaning; the number of 10-min samples behind today's value; the smoothed state; and the fouling-law forecast. No train label.
- Targets: `dp_pct` and `flux_pct`. Every forecast is clipped to the deposit's 0–1 range.

| | h = 1 | h = 3 | h = 7 | h = 14 |
|---|---|---|---|---|
| pairs | 1,169 | 1,083 | 994 | 881 |
| before day 550 (training only) | 527 | 466 | 422 | 379 |
| test block 1, days 550–599 | 142 | 137 | 126 | 105 |
| test block 2, days 600–649 | 149 | 147 | 143 | 136 |
| test block 3, days 650–699 | 150 | 150 | 150 | 150 |
| test block 4, days 700–771 | 201 | 183 | 153 | 111 |

## Methods

1. **No change:** the value on day *t*.
2. **Smoothed no change (the state):** the median of the last three observations of the current cycle within the last 7 days.
3. **Trend:** the 14-day slope inside the current cycle, extrapolated *h* days (no change if fewer than 4 points).
4. **Fouling law (physics):** linear growth since the last cleaning — a straight line through the current cycle's observations
   up to day *t* (at least 5), extrapolated to *t + h*; no change if fewer than 5. Before the first cleaning, the cycle starts
   at the first record.
5. **XGBoost:** predicts the change from day *t* to *t + h* from the inputs above. Fixed settings: 400 trees, learning rate 0.03,
   depth 4, subsample 0.8, column sample 0.8, min child weight 5; the average of 5 seeds. No tuning on test data.
6. **Hybrid:** the fouling law plus XGBoost (same settings) trained on the law's error.

The LSTM of the earlier test is dropped: it was the weakest on future data and the data are small.

## Tests

1. **Future, rolling origin:** four test blocks by day *t* (table above). Each block's models train on all pairs whose target
   day *t + h* falls before the block starts — all trains, all earlier days.
2. **An unseen train, in the future:** for each held-out train and each block, train on the other two trains (target day before
   the block) and test on the held-out train in that block.

## Scores and what counts as success

- **MAE** (percentage points of the indicator) and **skill** = 1 − MAE / MAE(no change); positive skill beats no change.
- **R² of the change** (t → t + h), not of the level: a level R² looks high for any method that copies today's value.
- A method **beats no change** at a horizon only if its skill over all test blocks is at least **+5 %** and it is positive in at
  least **3 of the 4 blocks**. Both tests are reported; the unseen-train test is the stricter one.

## Uncertainty

For each method, horizon and block, a 90 % range = forecast ± the 90th percentile of the absolute errors on the last quarter
(by day) of that block's training pairs, made by the method fitted on the first three quarters only — errors on days it did not
see. The test forecasts themselves come from the method fitted on all the block's training pairs. On the test block we report
the share of truths inside the range and the range's width. The range **holds** if 85–95 % of test truths fall inside it.

## Early warning

- **Event (onset):** the first day a train's 7-day median ΔP rise reaches **+15 %** after at least 14 days below it — the
  membrane maker's cleaning threshold (pressure drop +15 %). The same with **+20 %** as a check.
- **Warning:** on a day when the 7-day median is below the threshold, a method warns if its forecast for *t + h* (h = 7, 14)
  reaches the threshold. Compared with: warning on the day the threshold is crossed (lead 0), and with no change (which never
  warns ahead).
- **Scores:** events caught (a warning in the 14 days before the onset), lead time in days, and false warnings (no onset within
  *h* days of the warning).
- The test period holds only **5 onsets** at +15 % (A: days 610, 642; B: 689, 753; C: 642) and 5 at +20 %. Five events cannot
  support statistics: this part is **descriptive**, and it is reported as such whatever it shows.

## Reported whatever the result

Every method, horizon, block and test; the earlier 7-day result (no method beat no change on future data) is the reference
this re-test either confirms or overturns. The notebook re-runs everything from the public deposit.

## Corrections after the run (28 October 2026)

- The early-warning section said *no change* "never warns ahead". That expectation was wrong: *no change* forecasts the raw daily
  value, which can reach the threshold while the 7-day median is still below it. The warning rule itself is unchanged; *no change*
  is scored by it like every other method.
- **The 90 % range had a small leak, now closed.** The declaration asks for errors "on days it did not see". The first run fitted
  the method on the first three quarters of the training pairs with no gap, so some of those pairs' target days fell inside the
  last quarter. Now the last quarter starts at a whole day, and the fit uses only pairs whose target day is before it.
  Found by the independent review; the corrected ranges are the ones reported. Over all four blocks the ΔP ranges now
  hold in 22 of 24 cases (23 before the fix) and the flux ranges in 4 of 24 (3 before).
- **Pairs that start on a cleaning day.** The rule "no cleaning in (t, t + h]" keeps a pair whose day *t* is itself a cleaning
  day, and that day's value can still show the state before the cleaning, so the pair can span the cleaning drop. These pairs are kept as
  declared; the notebook shows the results without them, in both tests, as a check.
- **Added after the run, descriptive only:** the mean error (bias) of each method; a split of false warnings into "the 7-day
  median crossed again, but not after 14 days below" and "no crossing at all"; the ranges' coverage block by block (declared,
  but not displayed in the first run).
- **Second cleaning legs (found 28 October 2026, while preparing phase 3b).** The ledger lists 18 campaigns, but the daily
  table's own cleaning days include five second legs, one or two days after a campaign day (train B: days 526 and 545; train C:
  days 345, 525 and 547), all before day 550. Phase 3 cut its pairs at the ledger days only, so a few training pairs span a second
  leg; no test pair can. A re-run of all of phase 3 with the legs added changes no verdict: the three simple methods that depend on
  the cleaning days move by at most 0.7 points of skill, XGBoost and the hybrid by up to 11.8. With only the order of the training
  rows changed, the same re-run without the legs already moves the learned methods by up to 2.7 points (rolling origin) and 7.1
  points (unseen train), because XGBoost samples rows at random: their skills carry that much run-to-run spread.
- **`ndp_pct` is reset at each cleaning (found 29 October 2026, while preparing phase 4).** The data section above follows the
  deposit's dictionary ("never reset at a cleaning"). The data show that `ndp_pct` falls back to zero after each cleaning, as
  Project 3 found (its VALIDATION, section 3.1), while `dp_pct` does not. Phase 3 used `ndp_pct` only as one of the inputs of
  XGBoost and the hybrid; no number changes, only the description.
