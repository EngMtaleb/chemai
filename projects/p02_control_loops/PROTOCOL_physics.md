# Physics-informed diagnosis — protocol written BEFORE any result

Written: 2026-10-23, after the model comparison (`PROTOCOL_plant.md`, `compare_models.py`,
`plant_check.py`) and before any method below has been run on the simulation test set or on
plant data. Nothing in this file is changed after the run; anything learned later goes into
VALIDATION.md section 23.

## Why

On the 22 plant loops of the episode the rule (regularity > 1 and shape > 0.6) caught 12 of 14
sticky valves with 2 false alarms, and no model trained on simulation did better. On all 58 loops
with a stated label the same rule caught only 12 of 26 sticky valves — most of the others do not
draw a regular cycle, so the rule cannot see them. Two causes were measured: the models learned
from numbers that mean something else on plant data (Harris read three times lower, because the
plant dead time is unknown), and the four numbers describe the waveform, not the valve.

The question here: does putting the physics of the valve inside the method help where the rule is
blind, without losing what it already does well?

## Four methods

1. **Valve fit — physics alone.** For each loop, CO is passed through the two-parameter stiction
   model (Choudhury, Thornhill & Shah, 2005) for a grid of S and J. For every candidate the
   reconstructed valve position MV̂ feeds a linear process model (ARX: two output lags, ten input
   lags, a constant) fitted to CV by least squares — the Hammerstein approach, with a grid instead
   of an optimiser so every candidate is visible. The candidate that explains CV best gives Ŝ and
   Ĵ. The fit gain, 1 − RSS(best) / RSS(perfect valve), says how much better a sticky valve
   explains CV than a perfect one. CO is scaled by its standard deviation, so Ŝ is comparable
   across plants. Diagnosis: stiction when the fit gain and Ŝ both exceed thresholds chosen on the
   simulation development set only.

2. **Physics-informed models.** The five models of the comparison (logistic regression, tree,
   forest, boosting, forest without Harris), same grids, same nested GroupKFold by simulated loop,
   with three physics features added: Ŝ, Ĵ / Ŝ and the fit gain. In simulation, Harris is computed
   with the same assumed delay as on plant data (2 samples), so the number means the same thing in
   training and in use.

3. **Hybrid.** The rule decides where it is clear: regularity > 1 and shape > 0.6 → stiction;
   regularity > 1 and shape < 0.4 → not stiction. Everything else — no regular cycle, too few
   cycles, or a shape between 0.4 and 0.6 — is decided by the physics-informed model with the best
   out-of-fold stiction F1 on the development set.

4. **Rule as teacher.** The 65 real loops without an expert label are labelled by the rule where it
   is clear (the same two zones as the hybrid); the others stay unlabelled and are not used. A
   forest with the physics-informed features and the settings chosen for method 2 is trained on
   the simulation development set plus these pseudo-labelled loops, stiction against the rest.

## Data, and what is touched when

- **Development:** the simulated set, `random_state = 42` (240 runs, 120 loops). The estimator's
  grid, the ARX orders and the physics-alone thresholds are chosen here, and only here.
- **Simulation test:** new plants, `random_state = 2026`, touched once.
- **Plant:** (a) the same 22 loops as the episode; (b) all 58 loops with a stated label, stiction
  against the rest. Both sets have been seen before, so the plant numbers describe — they do not
  prove.
- **Valve truth:** in simulation the true S is known. On the test set Ŝ, converted back to % of
  travel, is compared with it: rank correlation and median absolute error, by visibility
  (buried / marginal / visible).

## Reported for every method

Caught / of, false alarms, precision, recall and F1 for stiction against the rest; on the
simulation test a 95 % interval by resampling whole loops. All methods are reported, not only the
best. Nothing is re-tuned after seeing the test or plant numbers.

## Decision rule (fixed now)

A method replaces the rule as the shipped diagnosis only if all three hold:

1. simulation test: the lower end of its stiction F1 interval is above the rule's F1;
2. plant (b), 58 loops: caught − false alarms beats the rule by MORE than two loops;
3. plant (a), 22 loops: caught − false alarms is not worse than the rule by more than one loop.

Otherwise the rule stays the diagnosis, and the method is reported for what it adds — for example
as a second opinion on the loops the rule cannot judge.
