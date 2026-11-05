# Phase 1, addition 2 — the hybrid LSTM and the hybrid RNN: what happened

Declared in `declaration_addendum2.md` (SHA-256 `8b319bc2…ebb3`, recorded 2026-11-03T05:42:08Z, before any run). That file
is left exactly as hashed, and its SHA-256 still matches; this report records the run.

## The run

- `run_hnets.py`, two parts side by side: 64 declared folds (both tests, four horizons, every block; `dp_pct` and
  `flux_pct`), five seeds per method, plus the four chart re-runs of addition 1's LSTM and RNN.
- `score2.py`: joined 64 folds (9,332 forecast rows). The eleven saved methods (phase 1's six, addition 1's four, and
  "no change") re-scored with `p3lib.scores` / `p3lib.verdict` unchanged: largest difference from addition 1's score
  files 2.8e-14, in both tests.
- Early stopping (median over 640 seed fits each): hybrid LSTM best epoch 12, stopped at 27; hybrid RNN best epoch 11,
  stopped at 26.

## Result — skill against “no change”, % (pass: +5 % pooled and better in 3 of 4 blocks)

Future blocks (rolling):

| | ΔP 1 d | 3 d | 7 d | 14 d | flux loss 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|---|---|---|
| hybrid LSTM | −57.5 | −22.6 | −30.0 | −30.8 | −46.1 | −22.7 | −17.1 | −21.0 |
| hybrid RNN | −42.0 | −27.9 | −32.1 | −30.6 | −33.9 | −21.3 | −21.3 | −16.2 |

An unseen train:

| | ΔP 1 d | 3 d | 7 d | 14 d | flux loss 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|---|---|---|
| hybrid LSTM | −65.6 | −41.6 | −42.3 | −38.0 | −49.0 | −27.1 | −27.1 | −28.8 |
| hybrid RNN | −54.9 | −55.4 | −43.5 | −42.4 | −40.8 | −24.8 | −26.0 | −30.5 |

**Passes: none**, in either test. As hybrids, both networks did worse than alone (LSTM and RNN: −1.6 to −17.5 on the
rolling test) — close to phase 1's own hybrid (the law plus XGBoost). The expectation written before the run held.
Across all the methods the one pass is still phase 1's: the smoothed state on flux loss, two weeks ahead, in both tests.

## Also saved for the deck (descriptive, computed afterwards)

- Stage-1 curves on the fit rows and the validation slice, and a 90 % range from the validation slice, for the two
  hybrids and for addition 1's LSTM and RNN (`parts/*_curves.csv`).
- The LSTM and RNN re-runs (`dp_pct`, 7 days, rolling) reproduced the saved test forecasts exactly (largest
  difference 0.0, checked again in `make_p1_data.py`).
- Prophet's chart (`prophet_chart.py`, addition 1's `forecast()` unchanged, phase 1's corrected range split): 803 fits;
  3 fell back to “no change” by addition 1's declared rule — the first one or two days of a train (A day 0, C day 0,
  B day 1), too little history to fit.

## Changes after the run

None to the declared method, data, folds, networks, test or bar.
