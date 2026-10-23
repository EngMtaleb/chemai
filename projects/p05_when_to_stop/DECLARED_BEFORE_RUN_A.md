# Declared before the run — Part A: when to end a closed-circuit RO batch

Written on 2026-10-23, after the development work below and **before any test batch was scored
with the rule**. Committed to the repository before `step3_ccro_rule_test.py` is run. Nothing in
this file is edited after that run; corrections, if any, go below the line at the end, dated.

## The data and the split

The closed-circuit RO testbed record (doi:10.7481/1963162, CC BY 4.0): 881 batches in six
datasets, June–October 2021. A batch fills a closed loop with brackish NaCl feed, concentrates it
while permeate leaves, and ends when the reject conductivity reaches a chosen value; its brine is
then drained with fresh feed.

| Role | Data | Batches |
|---|---|---|
| Development | dataset 3 (extremum-seeking control, ESC) and dataset 4 (open-loop experiments) | 623 |
| **Test, scored once** | **dataset 6 (Bayesian optimisation, BO)** | 159 |
| Outside the rule's scope | dataset 5 (model-predictive control) — it changes the flow setpoints 18–66 times inside every batch; the rule below assumes setpoints held for the batch | 98 |
| Not used | datasets 1–2 (12 mS/cm feed, an older logger without phase labels) | 36 files |

**What was seen of the test data before this file was written.** During the audit, the median
cost curves J(c) of the four BO experiments were printed (one table per experiment, cost against
the conductivity a batch could have stopped at), and the setpoints of its batches were counted.
The rule was never run on BO batches, and none of its settings was chosen with them in view —
the settings below come from the development batches, where they were fixed first. The MPC
batches were looked at only for their setpoint counts.

## The cost of a batch

For a batch ended at time t, per cubic metre of permeate:

    J(t) = [ E(t) + E_drain + b * V_brine ] / V(t)        kWh/m3

E(t) is the energy of both pumps since start-up, V(t) the permeate since start-up, E_drain and
V_brine the energy and brine volume of draining this batch (measured at the start of the next
file), and b the price of brine in kWh per m³ — the testbed's own optimisers used b = 1 and b = 10.
Start-up and stabilisation are counted (the testbed's objective left them out).

## The rule (frozen)

For a batch whose flux and recirculation setpoints are held:

1. live cost m(t) = (feed-pump power + recirculation-pump power) / permeate flow, kWh/m³, trailing
   median over **120 s**;
2. fixed cost F = E_drain + b·V_brine of the **previous** batch — the drain that opens the batch's
   own file (for the first batch of a run, the development medians: 0.034 kWh and 16.0 L);
3. average cost J_rule(t) = (F + E(t)) / V(t);
4. **stop at the first filtration sample, at least 5 minutes after start-up, where m(t) ≥ J_rule(t).**

The square-root law, as a forecast of the stop: fit m = m0 + s·V by least squares on the
filtration samples between 5 and 20 minutes after start-up (only batches that run at least 25
minutes); then V* = sqrt(2·F_eff / s), with F_eff = F + E(start of filtration) −
m0·V(start of filtration).

## How a batch is scored

- The batch's **own best**: the sample where its J(t) — with the drain of its own brine — is lowest.
- **Inside**: the own best lies at least 5 minutes before the batch's end, and filtration lasted at
  least 10 minutes. Only then is the regret exact; otherwise the cost was still falling when the
  batch ended.
- Regret = J at the stop / J at the own best − 1. If the rule never fires, its stop is the batch's end.

## What the development batches gave (step2_ccro_rule_dev.json)

| b | Batches inside | Rule regret, median | p90 | max | Controller's end, median |
|---:|---:|---:|---:|---:|---:|
| 0 | 69 | 0.067 % | 0.105 % | 0.126 % | 0.155 % |
| 1 | 8 | 0.080 % | 0.100 % | 0.103 % | 2.40 % |
| 3 | 8 | 0.071 % | 0.138 % | 0.179 % | 1.22 % |
| 10 | 3 | 0.082 % | 0.096 % | 0.099 % | 0.10 % |

Batches whose own best was not inside (519–585 per b): the rule fired in 0–3.5 % of them, and in at
most 2 did its stop cost more than 0.5 % above the batch's end. The square-root law (fit on
minutes 5–20): median |V*/V_best − 1| 2.1–6.3 % (V*/V_best 0.82–1.08 from the 10th to the 90th
percentile), and stopping at V* cost a median 0.006–0.04 % above the best. A fixed end at
10 mS/cm cost 4–27 % above the best; at 15 mS/cm 0.03–4.5 %; at 20 mS/cm 0.1–0.9 %, depending on b.

On the development data the ESC controller ended **every one** of its 580 batches while the live
cost was still below the average — the cost was still falling (b = 1 in Exp1, b = 10 in Exp2–3).
That is a development finding; it is not part of the test.

## The tests on dataset 6, run once

Batches with held setpoints and their own brine measured. All four b values are scored on every
batch (a batch's path does not depend on b; only where it is stopped does).

**T1 — the rule finds the best end (primary).** For b = 1 and for b = 10 separately, over batches
whose own best is inside: median regret ≤ **0.5 %** and 90th percentile ≤ **1.0 %**. b = 0 and 3 are
reported the same way. A b with fewer than 10 inside batches is reported as *too few*, not as pass or fail.

**T2 — it does not stop early when the cost is still falling.** Over batches whose own best is not
inside, for each b: the rule's stop costs more than 0.5 % above the batch's end in at most **10 %**
of them.

**T3 — the rule against the optimiser that ran the batches** (reported, no pass mark). For each BO
experiment at its own b (Exp1 b = 1; Exp2, Exp3A, Exp3B b = 10): the share of batches BO ended
while the cost was still falling; on inside batches, BO's regret against the rule's; over all its
batches, the mean J at BO's ends against the mean J at the rule's ends (where the rule would have
run past a batch's end, J at the end is used — a bound that favours BO). Fixed ends at 10, 15, 20
and 25 mS/cm are scored the same way, on the batches that reached them.

**T4 — the square-root law forecasts the stop.** On inside batches at b = 1 and b = 10 (and 0, 3
reported): median |V*/V_best − 1| ≤ **10 %**, and stopping at V* costs a median ≤ **0.5 %** above the best.

## What would count against the idea

T1 or T2 failing at b = 1 or b = 10: the rule as frozen does not carry to a different month and a
different optimiser, and the README says so in its first table. T4 failing alone: the closed form
is a poor forecast at this scale even if the online rule works. A pass would mean that one batch's
own live signal is enough to end it at the best point, with no search across batches — not that
the testbed's absolute energy (about ten times a full-scale plant) carries to a real plant.

---
*Corrections after the run (dated, never edits above this line):*
