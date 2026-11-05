# Phase 1B again (note 18): the simulated ΔP reading matched to the real record, one merged dataset — design, written before the run

Asked by Mohammed on 4 Nov 2026 (10:57 UTC+3), after this finding (10:36): scored with the same function as the real record
(phase3b/fingerprint.py, from day 300), the proof of concept's simulated ΔP reading moved far more than the real one —
day-to-day noise 2.64–3.40 points per plant against 0.95; mean change over 1 day 3.47–4.31 against 1.49; over 7 days
5.29–7.20 against 2.82. A week ahead, even a forecast that knew the true fouling exactly was 4.54 points from the reading
(XGBoost: 4.89). He approved: make the data again with the real plant's movement, one dataset from the three plants, run
pages 38 and 39 again, and add a page after 35: the schedule from the real record's columns only.

## Checks made before writing this (the simulator's reading only: no model, no forecast)

- `noise_floor.py`: with a perfect ΔP transmitter (no noise, no outliers) the reading still moved 2.13 points a day and 4.70 a
  week. The transmitter is not the main source.
- `noise_floor2.py`: the screen normalises flow with exponent 1.5; the simulator's physics is ΔP ∝ q^(2−m), m = 0.19–0.30,
  so 1.70–1.81. When trains stop and start, each running train carries 0.75–1.10 of its design flow, and the mismatch moves
  the reading. With exponent 1.75 and a perfect transmitter: noise 0.79, 1 day 0.93, 3 days 1.14, 7 days 1.71 — below the
  real record.

## The simulator: phase 3b's version 2, with its measurement changed and nothing else

The plants, their fouling, the sea, the instruments (σ_Z = 0.75, s = r = 0.5), train stops and the operator's rule are as in
declaration_poc.md. Only how the ΔP reading is measured and shown changes:

1. **The screen's flow exponent: 1.75** (the middle of the simulator's own 1.70–1.81) instead of 1.5; the viscosity exponent
   stays 0.25. It applies everywhere the reading is normalised: the operator's screen, the table's `dp_pct`, the schedule runs.
2. **The ΔP transmitter's noise:** per train, uniform between 0.5·x and 1.5·x (was 0.8–3 %).
3. **Outliers:** probability p (was 5 %), still × 4.
4. **The reading's slow movement** (phase 3b's attempt 2, correlation time 7 days, its own random stream): standard deviation w
   (was off).

## Calibration: x, p and w chosen against the real record before any model runs

- **Grid:** x ∈ {0.002, 0.003, 0.004, 0.005, 0.006, 0.008}; p ∈ {0, 0.02, 0.05}; w ∈ {0, 0.005, 0.01, 0.015, 0.02} — 90 settings.
- **Calibration plants, never used again:** seeds 4301–4303 (linear), 4311–4313 (biofilm), 4321–4323 (asymptotic), simulated
  as the proof of concept's plants are.
- **Statistics** (fingerprint.py, unchanged; from day 300): the day-to-day noise; the mean absolute change over 1, 3 and 7
  days; the lag-1 autocorrelation of daily changes. Each plant scored on its own, then averaged over the nine.
- **Choice:** the smallest sum over the five statistics of ((simulated − real) / half-width of the real 90 % interval)².
- **Reported:** the chosen setting; which of the five fall inside the real intervals; the fingerprint's other statistics as
  they come (cycle length, levels, drops — not tuned).

## The table (page 37)

`make_table.py`, unchanged except the measurement above: plants P1–P3 (seeds 4101–4103), the same laws, columns, no-link
columns and gaps. New table.csv, cleanings.csv and hidden.csv; their SHA-256 written before any model reads them.

## The forecast (page 38, code 05): one dataset from the three plants, phase 1's steps

- **Pairs:** phase 1's build_days with every logged column (record 2), per train; target `dp_pct`, 7 days ahead; both days
  recorded, no cleaning in between.
- **One dataset:** the three plants' pairs together (a train is named by plant and letter). **Test:** phase 1's rolling origin
  — blocks 550–599, 600–649, 650–699, 700–771 — each trained on every earlier pair of all three plants (t + 7 < block start).
- **Methods, settings unchanged:** no change (the rival); the fouling law; XGBoost (phase 1's settings, five seeds); the hybrid;
  the LSTM and the RNN (addition 1's networks and fitting, five seeds, every logged column in their 17-day sequence).
- **Scores:** MAE in points, pooled; skill against no change; phase 1's bar (pooled skill at least +5 % and positive in at
  least 3 of the 4 blocks) — the ✓ on code 05.
- **Negative control:** XGBoost with SDI and turbidity shifted in time (circular, a random 60–300 days per plant, after each
  column's yearly cycle is taken out and put back — the fix the last control's investigation found) against XGBoost without
  the two columns, 7 days ahead. Skill within 2 points, or nothing is reported until the cause is found.
- **The chart (plant 1, trains A, B, C) and its 90 % range:** make_poc_chart.py's method, with the model trained on all three
  plants' earlier pairs. Coverage counted as before.
- **Not run again (not shown):** record 1, the unseen-train test, `flux_pct`, horizons 1, 3 and 14 days, permutation importance.

## The schedules (page 39 and the new page after 35): declaration_schedule.md, with these changes

- The 15 test plants (seeds 4201–4205, 4211–4215, 4221–4225) are simulated with the calibrated measurement.
- The models learn from the new table (all three plants).
- Two models, each against its own calendar (the calendar's gap chosen by the same rule: the smallest that cleans at least
  as often as that model):
  - **every column** (page 39): the inputs declared there;
  - **the real record's columns only** (the new page): ΔP's own history, days since the last cleaning, feed temperature and
    flux, today and their 7-day means — no SDI, no turbidity, which the real record does not have.
- Counted as before, from the simulator's truth: cleanings and days above the limit per train-year; one example train by the
  median rule.

## Reported as it comes out

Whatever the numbers are, the pages say them. If a model's schedule does not beat its calendar, that is what the page says.
