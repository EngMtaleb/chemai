# Phase 3b — Fouling with its causes, on a simulator calibrated to the real plant: declared before any simulator run

Written 28 October 2026, before the new simulator was run even once. The real plant's fingerprint below was measured first
(`fingerprint.py`): it describes the deposit and contains no model output. Nothing here will be changed after seeing simulator
output; any later change will be listed as a change, with its reason, next to the original.

## Why

- Phase 3: with the deposit's four daily indicators, no method beat *no change* for ΔP, 1–14 days ahead. The deposit holds
  none of fouling's causes (feed quality, operating data) and no calendar dates (its column dictionary: "Exact calendar dates
  are NOT deposited"), so real feed records cannot be matched to its days.
- Adding simulated cause columns to the real ΔP is ruled out. A column made independently of ΔP carries nothing; a column
  made to explain ΔP has to be built from how ΔP changed afterwards — from the answer (leakage).
- So the causes are simulated inside a simulator whose ΔP is first made to behave like the real plant's.

## The questions

1. **Calibration (a gate).** Can the simulator reproduce the real plant's ΔP fingerprint, and the real answer with ΔP history
   only? If not, the work stops there.
2. If a plant logs the causes of fouling, from what point does forecasting ΔP beat *no change*?
3. Which counts more — operating data every plant logs (feed temperature, flux, recovery) or feed-quality instruments
   (SDI, turbidity) — and what does instrument noise cost?
4. **Controls.** Does the gain appear when the planted cause is logged, and vanish when its record is shifted out of step?

## What the simulation cannot say

The results describe this simulator under the assumptions below. They are not expected results for any real plant, and no
number from them will be offered as one. The deliverable is a map: the gain from forecasting against how much of the fouling
driver a plant logs, and how precisely. Every figure carries the word SIMULATION.

## Target and scope

ΔP only (`dp_pct`, the cleaning trigger). Flux loss is out of scope: the deposit's flux indicator is exactly 0 on 59 % of the
test train-days, which leaves no shape to calibrate against.

## The real plant's fingerprint (measured first)

Days 300–771 (before day 300 the record is sparse). Cleanings: the daily table's own cleaning legs, legs within 7 days forming
one campaign — 18 campaigns and 15 complete cycles, as in the ledger. Percentage points (pp) of `dp_pct`. 90 % intervals:
bootstrap over whole cycles and whole campaigns, 2,000 draws.

| statistic | value | 90 % interval |
|---|---|---|
| cycle length, median (days) | 42 | 39–45 |
| share of cycles longer than 60 days | 0.27 | 0.08–0.46 |
| rise over days 4–45 of a cycle, median (pp per week) | 0.30 | 0.00–0.64 |
| day-to-day noise, median per cycle (pp) | 0.95 | 0.78–1.16 |
| lag-1 autocorrelation of daily changes | −0.14 | −0.25 to −0.05 |
| mean absolute change over 1 day (pp) | 1.49 | 1.29–1.68 |
| mean absolute change over 3 days (pp) | 2.41 | 2.06–2.70 |
| mean absolute change over 7 days (pp) | 2.82 | 2.48–3.08 |
| mean absolute change over 14 days (pp) | 3.77 | 3.13–4.33 |
| drop at a cleaning, median (pp) | 4.8 | 3.9–7.3 |
| level the operator saw before cleaning, median (pp) | 22.8 | 20.9–32.5 |
| level, 10th percentile (pp) | 1.7 | 0.0–6.7 |
| level, median (pp) | 16.0 | 14.2–18.8 |
| level, 90th percentile (pp) | 33.4 | 24.6–41.5 |
| missing days | A 30 %, B 8 %, C 9 % | — |

## The simulator, version 2

The base is Project 3's `chemai/data/ro_sim.py`: a train-level physics model (permeate held at its set point; ΔP from a
blocked-gap law; three fouling laws — steady deposition, biofilm, deposition balanced by removal — faster in warm water and with
unlogged upsets; cleanings that remove part of the layer; an operator who cleans on the normalised ΔP on the screen; sensor
noise). The base file is not changed: Projects 3 and 6 rest on it. Version 2 lives beside it and changes only these:

1. **Feed temperature:** each simulated plant takes a random 772-day window of the daily sea temperature used in phase 2
   (NASA POWER, MERRA-2 earth skin temperature at 29.00 N, 48.75 E, off Kuwait, 2015–2024; 14.9–34.9 °C), with 0.2 °C sensor
   noise.
2. **A fouling driver Z(t)** — unitless, unit variance: 30 % a yearly cycle with a random phase, 70 % slow random variation
   (first-order, correlation time 7 days). *Assumption.*
3. **The daily fouling rate** = the base law's rate × q10^((T − 25)/10) × (J / J_design) × exp(σ_Z·Z − σ_Z²/2) × the base's
   daily noise; upsets as in the base, never logged. J / J_design: particles are carried to the membrane by the permeate flow.
   *Assumption.* σ_Z is set in calibration.
4. **What the plant logs.**
   - Level 1, operating data: feed temperature, permeate flux and recovery, as measured.
   - Level 2, feed quality: two instruments, SDI and turbidity, each reading L(t) + r·e(t), with e daily and independent for
     each instrument, and L = √s·Z + √(1 − s)·V, where V is an independent copy of the same kind of signal as Z.
     **s** is the share of the driver the instruments see (coverage); **r** is their noise (precision). Daily noise averages out
     over days; the unseen part does not. SDI-like and NTU-like scales are used for readability only.
   - The plant itself — its ΔP — is the same for every s and r: only what is logged changes.
5. **The indicator:** the measured ΔP normalised to the reference flow and to 25 °C viscosity, as on the operator's screen,
   over its first-14-days baseline, minus 1, clipped 0–1, never reset at a cleaning — the deposit's kind of construction.

## Step 3 — calibration, the gate

- **Tuned, within these ranges:** σ_Z 0–1.5; relative ΔP sensor noise 0.004–0.04; outlier probability 0–0.05; the operator's
  cleaning level 1.12–1.40 (normalised ΔP) and delay 0–14 days; the longest time without a cleaning 40–160 days; the shortest
  time between cleanings 10–30 days; the base fouling rates × 0.3–3; cleaning efficiency 0.4–0.95; the on and off blocks that
  make missing days (means 5–40 and 1–10 days). Everything else as in the base.
- **Search:** 300 settings drawn by Latin hypercube (seed 3101), each run on 6 development plants (seeds 3201–3206; the three
  fouling laws in turn), pooled into one fingerprint.
- **Fingerprint pass:** the six statistics that set how predictable ΔP is — day-to-day noise, the lag-1 autocorrelation and the
  mean absolute change over 1, 3, 7 and 14 days — all inside their 90 % intervals; of the other nine (cycle length, share over
  60 days, rise, drop, level before cleaning, the three level percentiles, and the missing-day share within 8–30 %), at most two
  outside.
- **Forecast pass:** the real target is the real plant run with **ΔP history only** — phase 3's rolling origin, inputs limited to
  `dp_pct` today, its lags, its 14-day slope, the smoothed state, the fouling-law forecast and days since cleaning (no flux,
  salt-passage or NDP inputs, which the simulator does not calibrate), the table's own cleaning legs, and one XGBoost seed as
  in step 4. This real run is made once, right after this declaration.
  A setting passes if, on its 6 development plants pooled, every method × horizon gets the same verdict (beats *no change* or
  not, by phase 3's rule) as the real plant.
- **Choice:** among the settings that pass both, the one with the smallest σ_Z and the one with the largest (the least and the
  most room for causes to matter). If only one passes, that one. **If none passes, the work stops and is reported.**

## Step 4 — the test: the map

- **Plants:** 20 fresh test plants for each chosen setting (seeds 3301–3320; the three laws in turn), 3 trains, 772 days.
- **Protocol:** phase 3's rolling origin within each plant — blocks 550–599, 600–649, 650–699, 700–771; train on every earlier
  pair of that plant — horizons 1, 3, 7 and 14 days; the six methods; XGBoost and the hybrid with phase 3's settings but one seed
  (five in phase 3), for time.
- **Inputs:** level 0, ΔP history (as above). Level 1, plus feed temperature, flux and recovery (today, 7-day mean, values 1, 3,
  7 and 14 days back). Level 2, plus SDI and turbidity (the same), on the grid s = 0, 0.25, 0.5, 0.75, 1 × r = 0, 0.5, 1.
  Only XGBoost and the hybrid use the added inputs; the four simple methods are the same at every level.
- **Scores and verdict:** phase 3's skill against *no change* and its rule (+5 % or more over all blocks and positive in at
  least 3 of the 4), with errors pooled over the 20 plants; also the share of plants in which a method passes on its own.
- **Answer to question 2:** for each r, the smallest s at which XGBoost or the hybrid beats *no change* 7 or 14 days ahead — or
  "none".
- **Answer to question 3:** level 1 against level 0, and level 2 against level 1, at every grid point.

## Controls

- **Positive:** at s = 1, r = 0, XGBoost's skill 7 and 14 days ahead must be higher than at level 1. If it is not, the map is
  reported as "the planted driver could not be used" — itself an answer.
- **Negative:** at s = 1, r = 0, the SDI and turbidity records are shifted in time within each plant by a random 60–300 days
  (circularly: the same series, the wrong days). XGBoost's skill must fall back to within 2 points of level 1, 7 and 14 days
  ahead. If it does not, something leaks, and no map is reported until the leak is found.
- **Sensitivity** (reported, not gated): the driver's correlation time 2 and 21 days instead of 7, at s = 1, r = 0.5, with plants
  re-simulated and everything else as calibrated; their fingerprint is shown next to the result.

## Also reported, descriptive only

The 90 % ranges (phase 3's construction, with its gap) and the early warning of ΔP +15 % (phase 3's rule), at level 0, level 1
and level 2 with s = 1, r = 0.5.

## Reported whatever the result

The fingerprint comparison of every setting tried, the gate, every grid point, both controls and the sensitivity. The notebook
re-runs everything from the public deposit and the public temperature file.

## A correction to phase 3, found while measuring the fingerprint

The ledger lists 18 campaigns, but the daily table's own cleaning days include five second legs, one or two days after a campaign
day (train B: days 526 and 545; train C: days 345, 525 and 547), all before day 550. Phase 3 cut its pairs at the ledger days
only, so a few training pairs span a second leg; no test pair can. A re-run of all of phase 3 with the legs added (both
tests, five seeds) changes no verdict. The three simple methods that depend on the cleaning days move by at most 0.7 points of
skill; XGBoost and the hybrid move by up to 11.8 points. For comparison, the same re-run without the legs, with only the order
of the training rows changed, already moves them by up to 2.7 points (rolling origin) and 7.1 points (unseen train), because
XGBoost samples rows at random. Phase 3b uses the table's own legs, and keeps phase 3's row order (by day, then train) so that
every run repeats exactly.

## The gate's real target — run once, after this declaration was written

The real plant, ΔP history only, the table's own legs, one XGBoost seed: skill against *no change*, % (`real_level0.py`).

| method | 1 day | 3 days | 7 days | 14 days |
|---|---|---|---|---|
| smoothed no change | −20.8 | 2.2 | 1.8 | 3.6 |
| trend | −4.6 | −12.6 | −26.7 | −53.8 |
| fouling law | −48.6 | −6.4 | −16.6 | −21.1 |
| XGBoost | −12.7 | −4.4 | −19.4 | −28.2 |
| hybrid | −43.2 | −13.9 | −35.3 | −24.4 |

*No change*'s error: 1.76, 2.83, 2.91 and 3.49 points. No method beats *no change* at any horizon — so a setting passes the
forecast gate only if none does on its development plants either. (The first run used five XGBoost seeds by mistake — change 3
below: XGBoost −10.5, −5.1, −16.4, −25.9 and hybrid −42.0, −15.8, −34.7, −24.6, with the same verdicts. The table shows the
run with one seed, as declared.)

## Changes after the declaration

Both found by the step-2 checks (`check_sim_v2.py`, 29 October 2026), before any calibration run.

1. **The twin V's yearly cycle runs a quarter-year from Z's.** Two yearly sines with unrelated phases are correlated within a
   plant, by the cosine of their phase gap, so the instruments would have seen more or less than √s of Z. "Independent" is kept
   in the sense the questions need: V is uncorrelated with Z. Check: the instruments' correlation with Z equals √s / √(1 + r²)
   within 0.02 at all 15 grid points.
2. **The clock for the longest time without a cleaning starts at day 0.** The base simulator starts it at minus 10,000 days,
   so it cleaned every train as soon as its 14-day baseline was in: in 30 plants of the base simulator, all 90 first cleanings
   were for "time since last cleaning", on day 23 (median). With the clock at day 0, the median first cleaning comes after 182.5
   days (30 plants of version 2); the real plant's first cleanings came on days 344 to 579. This is a fourth switch: with every
   switch off, version 2 is still the base simulator, draw for draw.
3. **The real target was first run with five XGBoost seeds, not the declared one.** The seed count was set after phase 3's
   library had been imported, which does not change a default fixed at import. Found while writing the calibration code
   (29 October 2026), before any calibration run. Re-run with one seed, checked by counting the models of a fit; the verdicts
   are the same (no method beats *no change*), and the table above shows the one-seed run.

## Step 3 result (29 October 2026): the gate is not passed

- **0 of 300 settings passed the fingerprint**, so none reached the forecast gate (`calibrate.py`; every setting's fingerprint
  is in `calibration_fingerprints.csv`). Three settings had all six core statistics inside; each had five or six of the other
  nine outside — always the cycle length, the share of long cycles, the level before cleaning, and the median and 90th-percentile
  levels.
- Where the simulator does not reach the real plant, within the declared ranges:
  1. **Day-to-day changes cancel out too much.** The lag-1 autocorrelation of daily changes ran from −0.51 to −0.18 across the
     300 settings (real −0.14; interval −0.25 to −0.05), inside in 19. In the simulator most of the daily movement is independent
     sensor noise, so one day's change is undone the next; in the real plant more of it persists for days.
  2. **The real plant stays high and cleans often.** Its median rise is 16.0 %; its operator cleaned at 22.8 %, every 42 days
     (median), and a cleaning took off 4.8 points. In the simulator ΔP stays low most of the time and reaches the operator's level
     only through long cycles: the level before cleaning was below the real interval in 242 settings, the cycle length above it
     in 278.
- As declared, the work stops here and is reported. Anything further is a new attempt, declared before it is run.

## Attempt 2 — declared 29 October 2026, before it was run

The owner chose a second and last attempt, on one condition: each addition must be realistic for real plants.

### Why the two additions are realistic

1. **Slow movement in the ΔP reading that is not fouling.** A normalised ΔP also moves with the instruments and with the
   normalisation itself. Pure Tec's guide to RO data normalisation puts a falling normalised ΔP down, usually, to faulty
   instruments or data-collection mistakes (*RO Data Collection and Normalization*, 2024); Hydranautics' *RO DataXL user manual*
   stresses that normalised data mean something only with the correct feed temperature and calibrated instruments. A calibration
   offset, a drifting transmitter or an imperfect flow correction lasts until it is fixed or the operation changes — days, not
   hours. The real plant shows that kind of persistence (lag-1 autocorrelation of daily changes −0.14, against −0.51 to −0.18 in
   attempt 1). Which cause acts in this plant is not known; the addition stands for all of them, and no instrument logs it.
2. **A larger share of fouling that a cleaning leaves for good.** Chesters & Armstrong (*RO Membrane Cleaning – explaining the
   science behind the art*, IDA World Congress, 2013) describe foulants that grow harder to remove the longer a cleaning waits,
   and incomplete cleans that leave a higher underlying fouling rate behind. On this plant, the deposit's own analysis finds
   normalised ΔP at or above +10 % on 65 % of days and at or above +15 % on 47 % (`analysis_outputs/vendor_rules_opchar.json`),
   and the fingerprint shows cleanings taking about 5 points off a level of about 23 %.

### What is added

- **A. A slow movement u(t) in the ΔP transmitter's reading:** the reading is multiplied by (1 + u), u a first-order random
  process for each train with standard deviation σ_u and correlation time τ_u. The operator sees it and so does `dp_pct`; nothing
  logs it. It is drawn from a third random stream, so attempt 1's plants stay exactly as they were. Tuned: σ_u 0–0.05,
  τ_u 2–30 days (log scale).
- **B. The irreversible share** of what a cleaning leaves behind: tuned 0.02–0.60 (the base draws it from 0.02–0.10).
- Everything else as in attempt 1: the other eleven ranges, the simulator, the fingerprint and its rules.

### The search

- 300 settings by Latin hypercube over the 14 tuned quantities (seed 3102); then 200 settings around the 10 settings with the
  smallest fingerprint distance — every quantity moved uniformly by up to ±10 % of its range (on the log scale for the
  log-scaled ones), clipped to the range (seed 3103). The fingerprint distance: the sum over the 15 statistics of the squared
  distance from the real value, in half-widths of its interval.
- Development plants 3201–3206 as before. The test seeds 3301–3320 stay untouched.
- The same fingerprint pass, the same forecast gate, the same choice (the smallest and the largest σ_Z among the settings that
  pass both). In step 4 the test plants' own fingerprint is reported too, as a check that the calibration holds on plants it
  never saw.
- **If no setting passes, phase 3b stops for good** and is reported as a negative result.

### What it changes in the map

u is movement that no instrument sees, so it makes ΔP harder to forecast for every method. The map then answers: on a plant
whose ΔP moves the way this one's does, how much does logging the causes help?

## Attempt 2 result (29 October 2026): the gate is not passed — phase 3b stops

- **0 of 500 settings passed the fingerprint** (`calibration_fingerprints_attempt2.csv`), so none reached the forecast gate.
  Before the search, attempt 1's setting 0 was re-run with attempt 2's code and reproduced exactly.
- The two additions moved single statistics towards the real plant — the lag-1 autocorrelation was inside in 115 of the 500
  settings (attempt 1: 19 of 300), the cycle length in 72 (18), the median level in 63 (23) — but never all together.
- Seven settings had all six core statistics inside. In all seven the plant sat low — level before cleaning 4.6–7.3 % (real
  22.8 %), median level 2.3–4.7 % (real 16.0 %) — and a cleaning took off 0.0–3.2 points (real 4.8). The setting closest
  overall (distance 31.0; attempt 1's closest: 40.7) had its median level nearer (13.5 %) but four of the six core statistics
  outside.
- **Conclusion, as declared: phase 3b stops for good.** In two declared attempts and 800 settings, this physics simulator could
  not move like the real plant from day to day and, at the same time, hold its high level with short cycles. A map built on it
  would describe the simulator, not plants like this one, so none is built. How much logging the causes would help stays an
  open question that only plant records carrying the causes can answer.
