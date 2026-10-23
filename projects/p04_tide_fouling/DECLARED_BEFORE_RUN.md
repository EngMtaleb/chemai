# Project 4 — declared before the tide test was run

Written and committed before any tide term was computed against the real clock
of the data. Everything below is fixed: responses, model, null, thresholds,
and what each outcome will be taken to mean. A change after this commit is
listed in VALIDATION.md with its reason, and both versions are reported.

## 1. The question

At an open seawater intake, is the fouling of the pre-treatment membrane locked
to the astronomical tide? If it is, part of the plant's future fouling load is
known years ahead, and backwash, coagulant and production can be planned
against it. If it is not, planners can ignore the tide — also worth knowing.

## 2. What was done before this declaration

- **Audit** (step1, `results/step1_audit.json`). 180 files; 150 carry a logged
  clock. Six clocked files start on a different day than their name says and
  are excluded. Runs shorter than 6 h are excluded. Kept: 107 runs, 5,105.8 h,
  2012-06-23 to 2015-04-01. The logger runs on local civil time: the daily
  temperature maximum falls at 12–19 h in all 34 runs of 48 h or more.
- **Segments** (step2, `results/segments.parquet`). 14,998 filtration segments
  in 106 runs, 86 episodes. Median fouling rate 13.2 %/h.
- **Power** (step3, `results/step3_power.json`) — synthetic M2 waves added to
  noise shaped like the data; the data's real alignment is never used. For the
  fouling rate, a wave whose 5–95 % range is 10 % of the median (≈ 1.3 %/h) is
  found at p ≤ 0.01 in 99.5 % of 200 trials; 5 % in 43 %; 2 % in 3 % (no
  power). Anything at the 20 % relevance line below would not be missed.

No tide term has been fitted to any real response.

## 3. Responses

| role | response | data |
|---|---|---|
| **primary (H1)** | fouling rate per segment, `fouling_pct_h` = 100·d ln R/dt | all segments |
| positive control (C1) | feed-pump drive speed, segment mean | all segments |
| positive control (C2) | MF inlet pressure (pump discharge), segment mean | all segments |
| mechanism (S1) | feed turbidity, segment mean | runs whose median turbidity is ≥ 0.20 NTU |

R = TMP / (μ(T)·Q), μ from the Vogel equation. Each response is winsorised
within its run at the 1st and 99th percentiles.

## 4. The test (same for every response)

- **Clock**: logged local time, daylight saving removed (−1 h from the second
  Sunday of March 02:00 to the first Sunday of November 02:00). Hours are
  counted from 2000-01-01 00:00 on that clock; phase is fitted, so the offset
  from UTC does not matter.
- **Base model** (pooled OLS): run levels; per-episode slow trend (linear below
  96 h, cubic spline with a knot every 48 h above); time of day, 4 harmonics of
  24 h; temperature and ln(flow) as deviations from the run mean;
  ln(segment minutes); element-pair dummies.
- **Tide block**: cos/sin at M2 (12.4206 h), N2 (12.6583 h), O1 (25.8193 h),
  M4 (6.2103 h). S2, K1 and P1 are left out: they sit on 12.00, 23.93 and
  24.07 h and cannot be told apart from the day.
- **Statistic**: the residual sum of squares the tide block removes beyond the
  base model.
- **Null**: 999 surrogates. Episodes closer than 72 h form one shift block (38
  blocks for the primary response). In each surrogate every block's clock is
  moved by k whole weeks, k drawn independently from ±1…52, and the tide block
  is recomputed. Whole weeks keep time of day and day of week; only the moon is
  scrambled. The data are never moved.
- **p** = (1 + number of surrogates at or above the observed statistic) / 1000.

## 5. Decision rules

| check | passes when |
|---|---|
| H1, primary | p ≤ 0.01 |
| Confirmation (out of time) | tide component fitted on runs before 2014-01-01, frozen, removes a positive share of the residual variance of runs from 2014-01-01 on, with p ≤ 0.05 against the same week-shift null |
| Calibration | the placebo block (11.0, 13.5, 27.0 and 5.3 h, same model and null) on the primary response gives p > 0.01 |
| Operational relevance | the 5–95 % range of the fitted tide component over 2012–2015 is at least 20 % of the median fouling rate, and above the 95th percentile of the same range under the null |

**What the outcomes will mean**

1. H1, confirmation and calibration all pass → fouling at this intake is locked
   to the tide; its size decides whether it matters (relevance row).
2. H1 passes, confirmation fails → a tide-locked signal in part of the record
   only; not stable enough to plan with.
3. H1 fails → no tide effect at this intake larger than the minimum detectable
   size from step 3 (stated with that size, not as "no effect").
4. Calibration fails → the null is not trusted; H1 is reported as
   uninterpretable whatever its p.
5. Positive controls (C1, C2) at p ≤ 0.01 → the intake's hydraulics feel the
   tide and the clock is right. If both fail, a negative H1 is weaker: the feed
   may pass through storage that hides the tide.

## 6. Reported but not decisive

- The depositors' own split: fit on `training/`, frozen, scored on `test/`.
- Sensitivity: no daylight-saving correction; no covariates; no winsorising.
- Per-constituent amplitudes and phases; M2 phase fitted separately before and
  after 2014-01-01.
- S1 (turbidity) uses 2013 runs only — the only year the sensor read the feed.

## 7. What this is not

One pilot at one intake. A tide-locked signal is an association with the
astronomical clock, not proof of a mechanism. Turbidity covers 2013 only.
Membrane area is not in the record, so the fouling rate is relative (%/h), not
an absolute resistance.
