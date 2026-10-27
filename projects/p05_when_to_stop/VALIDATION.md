# Project 5 — validation notes

Every claim in README.md traces to a section here, and every number here to a file in
`results/`, a declaration, or a test in `tests/`. Four parts, four declarations, each committed
and pushed before its run:

| Part | Decision | Data | Declaration | Run |
|---|---|---|---|---|
| A | when to end a closed-circuit RO batch | CCRO testbed, doi:10.7481/1963162 | `DECLARED_BEFORE_RUN_A.md` (bb33102) | `step3`, 6fb54c6 |
| D | when to clean an RO train | RO plant deposit of Project 3 | `DECLARED_BEFORE_RUN_D.md` (b3da9f1, corrected 349394b) | `step4`, 584fb88 |
| B, C | when to backwash, when to clean a UF membrane | UF pilot record of Project 4 | `DECLARED_BEFORE_RUN_BC.md` (cdf64b7) | `step5`, 1053c14 |

## 1. The closed-circuit RO record, as found

Source: `results/step1_ccro_audit.json`; code `chemai/data/nawi_ccro.py`.

"Foundational Control Methods for Water Treatment Systems — Online Actuator and Sensor Data",
doi:10.7481/1963162, CC BY 4.0. Seven zip files, 351 MB (sizes and sha256 in
`chemai/config.py:CCROConfig`), about 3.4 GB of CSV. Read as text only: one experiment folder ships
a 216 MB Python pickle, which is never opened.

- **A testbed, not a plant.** One spiral-wound element (~7.2 m², from permeate flow ÷ flux), a loop
  of about 20 L, a feed pump that tops the loop up as fast as permeate leaves and a recirculation
  pump at 45–50 L/min. Feed: NaCl in water, 1.7–2.8 mS/cm in the experiments used here.
- **One file per batch.** Each opens with the drain of the previous batch's brine ("Discharge"), then
  start-up, stabilisation and filtration ("Experiment") until the reject conductivity reaches the
  batch's critical value. A batch's own brine is therefore measured at the start of the next file;
  `batch_table` pairs them when the next file is numbered next and starts within 120 s.
- **Pump power comes from the drives and is approximate.** At this scale the pumps run far below
  their best point: specific energy is 10–12 kWh/m³ (median 12.1 in the ESC runs, 10.3 in the BO
  runs), about ten times a full-scale brackish plant. Its *shape* against conductivity is physics;
  its *level* is the testbed's pumps (§5).
- **The feed-conductivity column** reads a sensor whose range starts at 15 mS/cm; at a 2 mS/cm feed
  it reads 10–20 and is not used.
- One sample every 0.25–0.7 s (median by dataset in the audit file).

| Dataset | What it is | Dates (2021) | Files | Used as |
|---|---|---|---:|---|
| 1 | conventional run, 12 mS/cm feed, other phase names | 3 Jun | 1 | not used |
| 2 | open loop, 12 mS/cm feed, no phase labels | to 2 Jun | 35 | not used |
| 3 | extremum-seeking control (ESC), b = 1 then 10, feed +50 % | 13 Jun – 14 Jul | 584 | development |
| 4 | open loop and set-point steps (system identification) | 23–26 Aug | 39 | development |
| 5 | model-predictive control (MPC) | 16–24 Sep | 98 | outside the rule's scope (setpoints change inside every batch) |
| 6 | Bayesian optimisation (BO), b = 1 then 10, feed +50 % | 6–21 Oct | 159 | **test, scored once** |

881 files with phase labels (880 batches and the conventional run), 861 batches with their own
brine measured. 753 batches hold their flux and recirculation setpoints for the whole batch: all
584 ESC and 159 BO batches and 10 open-loop ones — no MPC batch and no set-point-step batch.

## 2. The cost of a batch and the rule

Code `chemai/features/ccro_batches.py`, `chemai/models/stopping.py`; tests
`tests/test_ccro_batches.py`, `tests/test_stopping.py`.

    J(t) = [ E(t) + E_drain + b * V_brine ] / V(t)          kWh per m3 of permeate

E and V run from the start of start-up; b is the price of brine in kWh per m³ of brine (b = 1 is
brine at $0.10/m³ with power at $0.10/kWh). This is the objective the testbed's own optimisers
minimised, `SEC + b·SBC`, with one change: their code (shipped in the ESC folder) leaves start-up
and stabilisation out, and pairs each batch with the drain of the *previous* batch's brine. Here
both are counted and each batch gets its own brine.

**The rule**: live cost m(t) = pump power ÷ permeate flow, trailing 120-s median; fixed cost F from
the drain that opens the batch's own file; stop at the first filtration sample, 5 minutes or more
after start-up, where m(t) ≥ (F + E(t))/V(t). Setting dJ/dV = 0 gives exactly m = J: the stop is
where the next cubic metre costs as much as the average so far. Under a straight-line live cost
m = m0 + sV this is the square-root law V* = √(2F/s).

## 3. Development (ESC and open-loop batches)

Source: `results/step2_ccro_rule_dev.json`, `results/step2_dev_scores.parquet`.

The three settings (120-s window, 5-minute floor, the previous drain as F) were chosen here; the
1-, 2- and 5-minute windows and a 2-minute persistence were tried, and set-point-step batches were
found to fool any version of the rule (the live cost jumps when the flux is stepped down), which is
why the rule is declared only for batches with held setpoints.

| b | Batches whose own best lies inside | Rule above the best: median | p90 | max | Controller's end: median |
|---:|---:|---:|---:|---:|---:|
| 0 | 69 | 0.067 % | 0.105 % | 0.126 % | 0.155 % |
| 1 | 8 | 0.080 % | 0.100 % | 0.103 % | 2.40 % |
| 3 | 8 | 0.071 % | 0.138 % | 0.179 % | 1.22 % |
| 10 | 3 | 0.082 % | 0.096 % | 0.099 % | 0.10 % |

**The ESC controller never let a batch reach its best end.** In all 580 ESC batches with their brine
measured, the live cost at the end was still below the average — the cost was still falling. Median
end conductivity: 7.2 mS/cm in Exp1 (b = 1), 10.5 and 13.2 in Exp2 (b = 10), 15.4 in Exp3 (b = 10).
At b = 1, the BO batches of October found their best ends at 16–19 mS/cm (§4); ESC spent the 2.3
days of Exp1 drifting from 7.5 down to 6.8 and back up to 10 mS/cm. Development data, not part of
the test.

## 4. The test, as run (BO batches, once)

Source: `results/step3_ccro_rule_test.json`, `results/step3_test_scores.parquet`; the script refuses
to run unless the declaration is committed and unchanged, and records its commit (bb33102).

153 batches scored (159 files, 6 without their own brine).

| b | Inside | **T1** median | p90 | max | **T2** stopped early, costlier than the end by > 0.5 % | **T4** median abs. error of V* | stop at V*: median regret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 103 | 0.080 % | 0.135 % | 0.69 % | 0 of 50 | 4.3 % | 0.011 % |
| 1 | 97 | **0.069 %** | **0.112 %** | 1.24 % | 0 of 56 | 5.1 % | 0.014 % |
| 3 | 75 | 0.063 % | 0.090 % | 1.74 % | 0 of 78 | 7.5 % | 0.032 % |
| 10 | 10 | **0.019 %** | **0.061 %** | 0.15 % | 0 of 143 | **11.3 %** | 0.149 % |

- **T1 passes** at b = 1 and b = 10 (declared: median ≤ 0.5 %, p90 ≤ 1.0 %), and at 0 and 3.
- **T2 passes**: on batches whose cost was still falling at their end, the rule fired in 1.3–2.1 %
  and never cost more than 0.5 % above the end.
- **T4 fails at b = 10** (declared: ≤ 10 %). The square-root law, fitted on minutes 5–20, put the
  stop a median 11 % short (V*/V_best 0.76–1.04 from the 10th to the 90th percentile). The live
  cost is not quite a straight line: it rises a little faster early in a batch, and at b = 10 the
  best end lies far out, where the early slope overstates the later one. Stopping at the law's V*
  would still have cost only 0.15 % above the best — but as a *forecast* of the stop, the closed form
  is reliable only when the stop is near (b ≤ 3). The online rule needs no straight line and is
  unaffected.

**T3 — against the optimiser that ran the batches** (reported, no pass mark):

| BO experiment | b | Batches | Ended while the cost was still falling | Inside: BO's end above best, median (mean) | Rule, median (mean) | Mean J over all batches, rule vs BO |
|---|---:|---:|---:|---:|---:|---:|
| Exp1 | 1 | 39 | 69 % | 0.52 % (0.80 %) | 0.06 % (0.06 %) | −0.28 % |
| Exp2 | 10 | 88 | 95 % | 0.06 % (0.20 %) | 0.03 % (0.04 %) | −0.01 % |
| Exp3A | 10 | 17 | 100 % | — | — | 0.00 % |
| Exp3B | 10 | 9 | 100 % | — | — | 0.00 % |

The last column is a bound in BO's favour: where BO stopped while the cost was still falling, the
rule would have run on to a lower cost the record does not show, and is charged BO's cost instead.
So the measured advantage is small (0.28 % at most) *because* it can only be measured where BO ran
past its best. What the record does show is how BO searched: 69–100 % of its batches ended early,
and its nine exploratory batches that stopped below 8 mS/cm cost 5–9 % more than the experiment's
other batches at b = 1, and 8–127 % more at b = 10. The rule needs no search: it reads each batch's
own live signal.

**Fixed end points**, on Exp1 inside batches (b = 1): 10 mS/cm cost a median 3.8 % above the best,
15 mS/cm 0.07 %, 20 mS/cm 0.41 %. On Exp2 (b = 10): 10 → 21.8 %, 15 → 4.5 %, 20 → 0.71 %, 25 → 0.02 %.
A fixed end can be as good as the rule — if it was picked for that brine price and that feed. The
rule does not need to know either.

*After the test (not declared):* the best ends of the inside BO batches lay at 14.9–17.3 mS/cm for
b = 0, 16.2–18.9 for b = 1, 18.7–21.2 for b = 3 and about 24 for b = 10 (medians by experiment). The
rule stopped a median 1.2–1.6 mS/cm past them — the 120-s trailing median lags, and the cost is so
flat there that the lag costs 0.02–0.08 %.

## 5. After the test: what the rule means at full scale

Source: `results/step6_ccro_full_scale.json`, `results/step6_dev_constants.csv` (424 development
batches). Model-based, no pass mark.

In a closed loop the salt that enters stays, so the loop concentrates linearly in permeate volume,
and the feed pressure follows its osmotic pressure. On the development batches: dp/dc = **0.49 bar
per mS/cm** (NaCl osmotic pressure plus polarisation), loop volume **20.3 L**, brine drained per batch
**0.80** of it. Then the live cost rises at s = k_π·c_f/(36·η·V_L) and the square-root law becomes

    V* / V_loop = sqrt( 72 * beta * eta * b / (k_pi * c_feed) )

The testbed's live cost rises at 12.1 kWh/m³ per m³ — at η = 1 it would be 1.48: its pumps' marginal
efficiency is **12 %**. With 80 % pumps and a complete drain, the energy-and-brine optimum is:

| Feed, mS/cm | b = 1 | b = 3 | b = 10 | b = 30 |
|---:|---:|---:|---:|---:|
| 1.0 | 92.3 % (10.7 mS/cm) | 95.4 % (17.8) | 97.4 % (31.7) | 98.5 % (54.1) |
| 2.0 | 89.5 % (15.7) | 93.7 % (25.8) | 96.4 % (45.4) | 97.9 % (77.1) |
| 3.5 | 86.6 % (21.6) | 91.8 % (34.9) | 95.3 % (60.9) | 97.2 % (102.9) |

(batch recovery, end conductivity in brackets). At full-scale efficiency, energy is cheap enough that
almost any brine price pushes the best end past where real brackish water scales (gypsum, silica) —
so in a real plant the scaling limit, not the energy curve, will usually end the batch. On this
testbed the inefficient pumps made energy the larger term; the published optimisers balanced energy
against brine at a price the pumps, not the membrane, had set.

## 6. Part D — RO cleaning by energy alone

Source: `results/step4_ro_cip_economics.json`, `step4_cycles_mid.csv`, `step4_cycles_corner.csv`,
`step4_grid.csv`; code `chemai/features/ro_physics.py:fouling_energy_penalty`,
`chemai/models/stopping.py:interval_economics`.

The deposit of Project 3 (Su, Meng & Chang 2026; doi:10.5281/zenodo.21055948). **15** complete cycles
(the declaration said 14 — a hand count wrong by one, corrected below its line before the results
were committed; the thresholds hold either way).

The live cost is the extra energy a fouled train spends to hold its production:
price × kWh-per-bar × (NDP0 · ndp_pct + ½ · DP0 · rise of dp_pct), with 0.0368 kWh/m³ per bar with
an energy-recovery device (0.0772 without). Mid case: $0.10/kWh, NDP0 20 bar, DP0 2 bar, a cleaning
costing $1 per m³/day of train capacity.

| | Result | Declared | |
|---|---|---|---|
| **D1** mid case: cycles where the rule would have cleaned before the operators | **0 of 15** | ≤ 4 | pass |
| **D2** median implied cost of a cleaning | **$0.00** per m³/day | < $0.25 | pass |
| **D3** corner most in favour of cleaning | **6 of 15** | ≤ 7 | pass |
| Any of the 216 grid cases | 0 of 15 in 156 cases; 2–4 in 57; 6 in 3 (price $0.15, no energy recovery, cheap cleaning) | | |
| **D4** cycle length vs pressure drop left after the closing cleaning | Spearman **0.63**, p = 0.016, 14 cycles | reported | |

- **In 9 of the 15 cycles the energy penalty had no rising trend at all** (Theil–Sen slope ≤ 0): `ndp_pct`
  sits at its floor 15–65 % of the days. In the 6 that rose, the energy-optimal interval was 3.2–110
  times the actual one (median 4.6); 1.0–36 (median 1.6) in the corner case.
- The penalty at the moment of cleaning was **0.01–0.59 US cents per m³** in the mid case. A cleaning
  at $1 per m³/day costs as much as 100 days of a 1-cent penalty.
- **The plant's cleanings were not energy decisions.** The published DP_norm 1.20 trigger was reached
  in 10 of the 15 cycles — in 6 of them on the first day, because `dp_pct` keeps a fixed
  commissioning baseline and later cycles start above 1.20. What set the cleanings was a limit, salt
  passage, or routine (Project 3 found three cleanings with no hydraulic reason at all).
- **D4 is the one cost of waiting energy does not see**: longer cycles left more pressure drop after
  their cleaning. It rests on four long cycles (97–148 days) against ten shorter ones (18–45 days),
  on one plant; it is a reason to look, not a measured price.

## 7. Parts B and C — the UF membrane

Source: `results/step5_uf_economics.json`, `step5_backwash_mid.csv`, `step5_chemical_cleaning_mid.csv`,
`step5_uf_runs.parquet`, `step5_uf_daily.parquet`; code `chemai/features/uf_cycles.py`; tests
`tests/test_uf_cycles.py`.

**B — backwash.** 104 runs (107 with a trusted clock; 3 without 3 backwashes or a positive fouling
rate). The pilot spent a median **28.7 %** of its feed on backwash. A backwash cost
$0.017–0.027 in the mid case (water at $0.05/m³, power at $0.10/kWh, η 0.7); between backwashes the
TMP rose at the run's fouling rate.

| | Median V*/V_cycle | Declared | |
|---|---:|---:|---|
| **B1** mid case | **15.7** (10th–90th pct 11.6–21.0) | ≥ 5 | pass |
| **B2** corner most in favour of backwashing | **6.9** | ≥ 2 | pass |
| all 18 grid cases | 6.9–30.0 | | |
| by year (mid) | 2012: 10.8 · 2013: 17.2 · 2014: 16.3 · 2015: 15.1 | | |

Energy and water alone would have filtered about 16 times more per backwash. The pilot's interval
was set by something else — holding the TMP down, and the fouling a longer filtration might leave
behind.

**The question this record cannot answer.** Does filtering longer between backwashes leave more
fouling that only a chemical clean removes? On the development years (2012–2013), the change in the
post-backwash resistance over one rotation of the element pairs (3,760 rotations, 83 days) was
regressed on the filtration period, with the fouling rate, temperature, resistance level and pair as
controls and errors clustered by day (`explore_uf_period_dev.py`, `results/explore_uf_period_dev.json`).
Per m³ filtered, a longer period came out *worse* (+6.5e-5 per minute, p = 0.007); written as volume
times period, it came out *better* (−1.1e-4, p = 0.04). A sign that flips with the specification is not a finding, so no claim is
made and the 2014–2015 years were not used for it. The pilot's periods were partly set by a
resistance threshold, so they followed the fouling rather than causing it. **What would settle it**:
a week of alternating days at two fixed periods (say 20 and 40 minutes), same flux and dose — the
cheapest experiment a plant can run, and the one that prices most of its backwash water.

**C — chemical cleaning.** Falls of more than 0.4 in the daily ln R per element, the logged days at
most 30 apart: 2013-10-09, 2014-03-18, 2014-07-28 (the fall of 2013-05-30 lies across a 58-day gap
and is not dated, as declared). Two complete intervals:

| Interval | Days | Resistance rise (ln R) | T*/T, mid case | Implied cost of a clean |
|---|---:|---:|---:|---:|
| 2013-10-09 → 2014-03-18 | 160 | 0.86 (TMP × 2.4) | 1.07 | $0.20 per m³/day |
| 2014-03-18 → 2014-07-28 | 132 | 0.67 (TMP × 2.0) | 1.28 | $0.14 per m³/day |

Over the whole grid the median T*/T runs from 0.33 to 3.0. Unlike the RO train, the UF resistance
roughly doubles between cleans, and energy alone lands near the intervals the pilot ran — at cleaning
costs ($0.14–0.20 per m³/day) a small UF train can plausibly have. Two intervals: an illustration,
not a test.

## 8. One rule, four stops

`figures/fig3_four_stops.png`.

| Decision | Time scale | Is energy the reason to stop? |
|---|---|---|
| End a CCRO batch | ~1 h | **Yes** — and the rule ends batches a median 0.02–0.07 % above their own best, where the record shows it (§4), tested |
| Backwash a UF element | ~20 min | **No** — energy and water alone: ~16× less often |
| Clean a UF membrane chemically | ~4–5 months | **About** — energy alone: 1.1–1.3× the actual interval |
| Clean an RO train | weeks to months | **No** — energy savings alone never covered a cleaning in 15 cycles (mid case) |

The same equation tells which costs matter at which scale. Where the live cost rises steeply and
reversibly (a concentrating batch, a resistance that doubles), energy times the square-root law is the
decision. Where it barely moves (a backwash cycle, an RO train between cleanings), the stop is set by
a limit or by fouling that hardens — costs this project names but, except for the RO hint in D4,
cannot price from these records.

## 9. Limits

- **Testbed, not plant.** The CCRO numbers are a one-element NaCl testbed with inefficient pumps;
  §5 is the arithmetic of carrying the tested rule to 80 % pumps, not a test of it at scale.
- **Prices are assumptions**, run on grids stated before the runs. The deposit and the UF record
  carry no prices.
- **The RO deposit is one plant, two years, 15 cycles**; ndp_pct is floored at its post-cleaning
  baseline, which can hide a small rise.
- **The UF cleaning dates are inferred** from the resistance, not logged.
- The rule assumes setpoints held for the batch. A controller that moves them (MPC) needs the live
  cost predicted at the setpoints it will use — a model this project did not build: on the
  development data, pump-power curves fitted on the June–July batches (1–3 % error there) missed the
  August batches by 17 % (feed pump) and 33 % (recirculation pump), all in one direction
  (`explore_ccro_pump_transfer.py`, `results/explore_ccro_pump_transfer.json`). The testbed's
  equipment drifts between months; a model of it would need re-fitting every few days, which is
  what the published MPC did.

## 10. Reproduce

```bash
pip install -e ".[dev,viz,data]"
# the seven zips from doi:10.7481/1963162 into data/nawi_ccro/ (urls: chemai.data.nawi_ccro.check_files)
python projects/p05_when_to_stop/step1_ccro_audit.py        # check, convert, cut into batches
python projects/p05_when_to_stop/step2_ccro_rule_dev.py     # development batches only
python projects/p05_when_to_stop/step3_ccro_rule_test.py    # the BO batches, once
python projects/p05_when_to_stop/step4_ro_cip_economics.py  # Part D (needs data/ro_deposit, Project 3)
python projects/p05_when_to_stop/step5_uf_economics.py      # Parts B, C (needs data/uf_pilot, Project 4)
python projects/p05_when_to_stop/step6_ccro_full_scale.py   # after the test, model-based
python projects/p05_when_to_stop/make_figures.py
```
