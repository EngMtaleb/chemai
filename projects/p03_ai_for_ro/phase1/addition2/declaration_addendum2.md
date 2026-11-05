# Phase 1 — addition 2: the hybrid with the two networks, declared before it runs

(The folder is still called `phase3`; in the deck this is phase 1. The names follow at the repo redo.)

Written 3 November 2026, 08:42 (UTC+3) — 3 November 2026, 05:42 UTC — before any hybrid network was trained or scored,
and before any code of this addition touched the study data. Nothing here will be changed after seeing results; if
something must change, it goes into a dated section "Changes after the run", with its reason, below the original text.

**Phase 1's declared results and addition 1's stay exactly as saved.** This addition edits no file outside
`phase3/addition2/`. Their forecasts and scores are read from `phase3/out/` and `phase3/addition/` as saved; whatever the
new methods show, no earlier verdict changes.

## Why

The project owner asked (3 November 2026, 08:30 UTC+3): "بس نبي نسوي HYBIRD WITH DL MODELS ممكن يعطينا نتايج اقوى ,
لازم نجرب", and confirmed (08:38) that Prophet gets no hybrid. Known before writing, as background only: in addition 1
the LSTM and the RNN beat *no change* in none of their verdicts, and phase 1's hybrid (XGBoost on the fouling law's
error) did not pass either.

## What stays the same (from `declaration.md` and `declaration_addendum.md`, and the code that ran them)

- **Pairs, inputs, targets, horizons, blocks, folds, scores, bar:** exactly addition 1's, through the same code, imported
  unchanged: `p3add.load`, `tables`, `folds`, `sequences`, `net_inputs`, `validation_split`; `p3lib.scores`, `p3lib.verdict`.
  Every fold of both tests: rolling origin (4 horizons × 4 blocks) and an unseen train (each held-out train, each block),
  for both indicators, `dp_pct` and `flux_pct`. A method beats *no change* at a horizon only if its skill over all test
  blocks is at least **+5 %** and it is positive in at least **3 of the 4 blocks**; both tests are reported, the unseen
  train being the stricter.
- **The networks:** exactly addition 1's LSTM and plain RNN — `p3add.make_net`, `Scaler`, `_epoch` and the settings in
  `p3add.py` (hidden 32, dense 16, learning rate 1e-3, weight decay 1e-4, batch 64, at most 150 epochs, patience 15,
  gradient clip 1.0, validation slice from 80 % of the fold's training rows), the same inputs (the 17-day sequences and
  XGBoost's other inputs), the same five seeds (0–4), the same two stages: stage 1 fits on the rows before the validation
  slice and stops early on the slice, giving E*; stage 2 refits on all the fold's training rows for E* epochs and
  forecasts the test rows.

## The only change: what the network learns

Addition 1's networks learn the change from today, *y − today*, and forecast clip(today + the mean of the five seeds'
changes, 0, 1). The hybrids learn the fouling law's error, *y − law*, and forecast clip(law + the mean of the five
seeds' errors, 0, 1). *law* is phase 1's fouling-law forecast exactly as phase 1's hybrid uses it:
`p3lib.simple(frame, c, h)["fouling law"]` — today's value where the law has no forecast. This is phase 1's hybrid with
the network in XGBoost's place. In code, the fitting is addition 1's `fit_predict_seed` with the law in the place of
today; its early stopping is then on clip(law + error).

- **Method 10: hybrid LSTM.** **Method 11: hybrid RNN.**

## What is reported

For both methods, both tests, both indicators and every horizon: MAE, skill, R² of the change, the four block skills
and the verdict, scored with `p3lib.scores` and `p3lib.verdict` on the same test pairs against the same *no change*
forecasts, next to the eleven saved methods (phase 1's six and addition 1's four). Reported whatever they show.

## Also saved for the deck — descriptive, computed after the declared test, no verdict rests on them

- **Training and validation curves:** for the rolling test, stage 1's forecasts (the five seeds' mean, with each seed's
  weights at its epoch E*) on its own fit rows and on its validation slice.
- **A 90 % range, computed afterwards** (no declared test had one for these methods): the 90th percentile of the
  absolute errors of those stage-1 forecasts on the validation slice — errors on rows stage 1 did not train on; reported
  as how often the truth fell inside on the rolling test.
- **The same curves and range for addition 1's LSTM and RNN**, by re-running their rolling folds for `dp_pct`, 7 days
  ahead, with the same code and a base of today: their stage-2 forecasts must equal the saved ones
  (`phase3/addition/pred_nets_rolling.csv`); if not, the saved ones stand and the difference is reported here.
- **Prophet's chart** (`dp_pct`, 7 days ahead): forecasts with addition 1's `forecast()` unchanged, at the origins of
  each rolling block's training pairs; a 90 % range computed afterwards the way phase 1 sizes its ranges — the 90th
  percentile of the absolute errors on the last quarter, by day, of each block's training pairs.

## Expectation, written before the run

We do not expect either hybrid to pass. The record holds four daily scores and the cleaning days, not the causes of
fouling; the law's error comes mostly from cleanings and jumps the inputs do not explain, and a different learner adds
no information. The test is run because the owner asked, and a declared result counts whichever way it falls.

## Changes after the run

(none yet)
