# Project 6 — validation notes

Every number in README.md traces to a file in `results/` (collected in `results/summary.json` by
`make_summary.py`, and in `results/posthoc_checks.json` by `posthoc_checks.py` for the checks made
after the tests) or to a test in `tests/test_ml_ladder.py`. Three parts, three declarations, each
committed and pushed before its test ran; each test script refuses to run unless its declaration is
committed and unchanged, and records the commit. Section 7 is an independent audit made after the
three tests: what it found, and how much each finding moves.

| Part | Question | Development | Test (once) | Declaration |
|---|---|---|---|---|
| A | an RO train reaches its review level within 28 days | 30 simulated plants (seed 1001) | 60 new simulated plants (seed 6006); real deposit exploratory | `DECLARED_BEFORE_RUN_A.md`, e7ad90b |
| B | a UF cycle's resistance climb, minute 4 → 11 | 4,168 cycles, 2012–13 | 10,830 cycles, 2014–15 | `DECLARED_BEFORE_RUN_B.md`, c0c9264 |
| C | a CCRO batch's best end, forecast from minutes 5–20 | 336 batches (ESC + open loop) | 142 batches (Bayesian optimisation) | `DECLARED_BEFORE_RUN_C.md`, aead79c |

## 1. Rules every part follows

- **Grouped development.** Tuning and out-of-fold scores split by plant (A), episode (B) or
  experiment (C) — rows of one group share one membrane or one controller. The development score of
  the chosen settings is mildly optimistic (chosen on the same folds); the test decides. Part B's test
  is split by date, so one episode that runs across 2014-01-01 sits on both sides (section 7).
- **The ladder, the same each time** (`chemai/models/ladder.py`): logistic regression or ridge,
  k-nearest neighbours, an RBF SVM / SVR, random forest, extra trees, XGBoost, LightGBM — each on a
  small grid; the ensemble is the mean of the three best on development. Then the same seven with
  physics features; an MLP and a sequence network; a physics-informed network
  (`chemai/models/deep.py`).
- **The pick of each rung is chosen on development** (best average precision or MAE) and named in
  the declaration; all models are reported.
- **Networks** train with Adam, weight decay and early stopping on a held-out fifth of the development
  groups (one experiment in C — with seed 0, the open-loop experiment: section 7); one seed.

## 2. Part A — the cleaning question

Source: `results/stepA1_ro_dev.json`, `results/stepA2_ro_test.json`; features
`chemai/features/ro_decisions.py`.

**The decision set reproduces Project 3's.** On the real deposit the builder gives the same 213
decisions, 61 crossings and 11 cycles as Project 3's external test, the forecast scores 0.213 / 0.722
and the margin rule 0.426 / 0.929 — once the rule reads the 7-day median exactly as Project 3 did
(a trailing median of the daily grid needing three recorded days). An earlier version read the
median of whatever days fell in the last week and scored the rule 0.410 / 0.893; it was corrected
before the declaration.

**Development** (10,192 decisions, 627 cycles, 27 % positive): margin rule 0.370 / 0.561, forecast
0.299 / 0.445. Average precision of the seven: 0.585–0.627 (plain), 0.570–0.630 (physics); MLP 0.609,
CNN 0.595, GapLawNet 0.577 (0.547 before an upset probability was added to it, on development).

**Final exam** (60 plants, 21,955 decisions, 5,221 crossings): margin rule 0.373 / 0.574, forecast
0.307 / 0.452. All 19 models: recall 0.552–0.632 at precision 0.576–0.617; recall minus the rule's
+0.18 to +0.26, every interval above zero. Frozen picks: random forest 0.632 / 0.591 (+0.259,
0.225–0.290); ensemble 0.619 / 0.604; XGBoost + physics 0.603 / 0.615; its ensemble 0.632 / 0.602;
MLP 0.608 / 0.604; GapLawNet 0.617 / 0.576. **All six pass.**

**Real deposit, exploratory** (213 decisions; intervals by cycle): picks at recall 0.607–0.738,
precision 0.627–0.740; best F1 0.739 (random forest + physics) against the rule's 0.584. Post hoc,
from the picks' precision-recall curves: at the rule's recall (0.426) their precision is
0.659–0.824; at the rule's precision (0.929) their recall is 0.000–0.098. Trained on the matched
simulator instead (30 plants, seed 1005, re-tuned): random forest 0.754 / 0.742, XGBoost 0.820 /
0.704.

## 3. Part B — the UF cycle's climb

Source: `results/stepB1_uf_dev.json`, `results/stepB2_uf_test.json`; curves `results/uf_curves.parquet`
(`chemai.features.uf_segments.segment_curves`, the same cutting as Project 4: all 14,998 segments
recovered); features `chemai/features/uf_forecast.py`.

**What the plant may use at minute 5:** the cycle's binned ln R up to minute 5; earlier cycles of the
episode within 3 h, each only after it ended; flow per element and temperature (segment means — the
declaration assumed the pilot holds its flow and the temperature moves slowly; the audit found the
flow assumption too strong, so these two inputs read past minute 5: section 7); elements online; hours
into the episode; days since the last chemical clean seen in the record. Not used: pump speed and MF
inlet pressure, whose segment means include the rest of the cycle.

**Development** (MAE, % of R): no model 1.537; straight line from 5 minutes 3.605; Hermia's best law
2.974; history 1.730. Seven ML 1.396–1.473 (plain) and 1.393–1.438 (physics); ensembles 1.389 and
1.377; MLP 1.427, GRU 1.437, HermiaNet 1.434. The best law of the first five minutes was complete
blocking (2,558 cycles) or cake (1,548) in 98 % of cycles — the two that bend most.

**Test** (10,830 cycles): no model 0.988; line 2.476; Hermia 2.070; history 1.088. Frozen picks
against no model (interval by episode): extra trees 0.943 (−0.045; −0.080 to −0.003) **passes**;
ensemble 0.952 (−0.036; −0.077 to +0.009) ties; extra trees + physics 0.948 (−0.040; −0.077 to
+0.002) ties; its ensemble 0.959 ties; MLP 1.453 (+0.465) **loses**; HermiaNet 0.942 (−0.046; −0.092
to +0.017) ties. Others: SVR + physics 0.914 and k-NN + physics 0.919 (the best); random forest,
XGBoost and LightGBM 1.07–1.21 and ridge 1.56–1.57 lose to no model.

**The clock:** extra trees with hour of day and M2 phase: 0.956, against 0.943 without.

**Separability:** for each test cycle, the four laws fitted to the whole curve (0–12 min) differ by a
median 0.05 % at minute 12; divided by the cycle's own point-to-point noise, a median 0.05; above 1
in 0 of 10,830 cycles. HermiaNet's mean weights on the test cycles: complete 0.08, standard 0.64,
intermediate 0.15, cake 0.13 — a curve fit, not a mechanism.

## 4. Part C — the batch's best end

Source: `results/stepC1_ccro_dev.json`, `results/stepC2_ccro_test.json`, `results/stepC2_test_stops.parquet`;
features `chemai/features/ccro_forecast.py`.

**Range.** After minute 20 the development batches reach r = V/V20 ≤ 2.34 (extremum seeking) and
≤ 6.58 (eight open-loop batches); the test batches reach 6.92. Development has four experiments, not
five: no Exp1 batch runs 25 minutes.

**Development** (MAE of m/m20 after minute 20; all folds · open-loop fold): the line 0.0143 · 0.0169;
flat 0.0285. Plain ML 0.007–0.050 · 0.023–0.073 (ridge best overall); hybrid 0.0045–0.0165 ·
0.008–0.120 (extra trees best on both); MLP 0.397 · 0.568; PowerLawNet 0.0083 · 0.0353.

**Test**, median |V*/V_best − 1| on inside batches, with the difference from T4 and its interval
(resampling batches):

| | b = 0 | b = 1 | b = 3 | b = 10 |
|---|---|---|---|---|
| T4 (Project 5's numbers, reproduced) | 4.31 % | 5.14 % | 7.46 % | 11.35 % |
| ridge | 6.87 (+2.52; +0.76 to +3.86) | 6.99 (+1.87; +0.45 to +3.22) | 9.83 (+2.12; −0.91 to +4.12) | 16.73 |
| ensemble | 6.31 (+1.89) | 8.07 (+2.89) | 11.49 (+3.79) | 13.82 |
| hybrid extra trees | 4.24 (−0.09; −2.04 to +1.24) | 3.98 (−1.18; −2.42 to +0.45) | **3.66 (−3.88; −4.59 to −2.09)** | 10.17 |
| hybrid ensemble | 4.86 (+0.57) | 3.83 (−1.31; −2.51 to +0.23) | **4.10 (−3.37; −4.27 to −2.18)** | 9.30 |
| MLP | 8.32 (+3.96) | 6.56 (+1.37) | 8.06 (+0.44) | 8.23 |
| PowerLawNet | 13.30 (+8.99) | 13.79 (+8.58) | 12.53 (+5.01) | 11.18 |

**No rung meets the declared bar** (better at b = 0, 1 and 3): the hybrid wins at b = 3 and ties at
0 and 1. Stopping at the hybrid's V* cost a median 0.010–0.011 % above the best at b ≤ 3 (T4:
0.011–0.032 %). Collapses: SVR 43.7 % (b = 0) and 268 % (b = 1); k-NN 78 % (b = 0).

## 5. Found during the work

1. **A network blew up on a held-out experiment.** The batch MLP scored 53 MAE on development: a
   setpoint held constant through the training experiments was standardised by a near-zero spread and
   became a huge input on the held-out one. Part C's networks now leave such features unscaled and
   clip standardised inputs at ±10 (`Standardiser(robust=True)`); Parts A and B, already run, keep the
   plain version. The MLP still fails (0.397) — the honest result for an unconstrained network asked
   to extrapolate.
2. **The margin rule must read the median the way Project 3 did** (section 2) — found by reproducing
   Project 3's real decisions before declaring.
3. **GapLawNet needed an upset term:** without it, its development average precision was 0.547;
   with a learned probability of a jump the law cannot see, 0.577.

## 6. What it is — and is not

**Is:** the same ladder run on three questions, each declared before its test, with the losses
reported beside the wins. **Is not:** a validation on plant data (one deposit, one pilot, one
testbed), nor an exhaustive search: the grids are small and each network has one seed.

## 7. An independent audit, after the tests

After the three tests, a separate reviewer — an AI agent that had not built the ladder, given only the
repository — read the code, the declarations and the results for leakage and protocol gaps. It found
no test data touching development in any part, and Part A clean. Its other findings follow, each with
what it moves. The numbers come from `posthoc_checks.py` → `results/posthoc_checks.json`: each check
first rebuilds the declared scores from the same code path (it does, exactly), the networks are not
refitted, and **nothing here replaces a declared result.**

1. **Part B: flow and temperature read past minute 5.** Both enter as means over the whole cycle (a
   median 13 minutes long). Temperature barely moves within a cycle (whole-cycle minus first five
   minutes: s.d. 0.02 °C), but flow does: 100 · ln(whole-cycle flow / first-5-minute flow) has a spread
   of 0.51 % in development and 0.28 % in the test, and correlates with the climb at −0.33 and −0.26.
   Refitted with first-5-minute values, every frozen pick scores better: extra trees 0.937 (−0.050;
   −0.085 to −0.013); its ensemble 0.936 (−0.052; −0.087 to −0.019); extra trees + physics 0.946
   (−0.042; −0.080 to −0.001); its ensemble 0.951 (−0.037; −0.074 to −0.003) — the last three move
   from a tie to a pass. Of the 14 models only XGBoost gets worse (1.162 → 1.169); ridge improves most
   (1.57 → 1.22). The look-ahead did not inflate the declared result; it slightly hid the models' gain.
2. **Part B: a narrow pass, and one episode on both sides of the split.** The pass holds under all 20
   bootstrap seeds tried (0–19; the declared seed is 7), but leave out any one of 7 of the 47 test
   episodes (41, 44, 45, 47, 80, 81, 82) and the interval's upper bound lands between +0.0001 and
   +0.0037: a tie. The split is by date, so episode 41 (30 December 2013 to 4 January 2014) has 78
   cycles in development and 240 in the test; "grouped by episode" holds inside development, not across
   the cut.
3. **Part C: three minute-20 inputs read a little past minute 20.** Conductivity is smoothed with a
   centred 15-sample median (7 s ahead at minute 20); flux, recirculation and temperature are medians
   of the whole filtration phase. With a trailing median and medians up to minute 20, the hybrid gives
   4.03 / 3.76 / 3.66 / 9.95 % at b = 0 / 1 / 3 / 10 (declared 4.24 / 3.98 / 3.66 / 10.17), its
   ensemble 4.73 / 4.37 / 4.10 / 9.30 and ridge 7.19 / 7.33 / 10.23 / 16.92. No frozen pick's verdict
   changes; one model that was not a pick, the hybrid XGBoost, would move from a tie to a win at b = 1.
4. **Part C: the networks never trained on a long batch.** Their early stopping holds out one
   development experiment; with seed 0 that is the open-loop experiment, the only one whose batches
   run past r = 2.34. The final MLP and PowerLawNet learned only from batches that stop by r = 2.34,
   while the other rungs trained on all four experiments. The deep-learning comparison in Part C is
   unequal, and PowerLawNet's poor stops may owe as much to that as to its form.
5. **The guards lock the declaration, not the code.** The test scripts refuse to run unless the
   declaration file is committed and unchanged; they do not check the code, the development JSON or
   the input tables (some are gitignored), and nothing stops a second run. Checked after the fact: the
   test scripts, feature modules, `ladder.py` and the development JSONs are unchanged since their
   declaration commits, and `deep.py` changed only in ways Parts A and B do not use (additions, and an
   option that is off unless asked for); the picks, ensemble members, thresholds and pass marks match
   the declarations; GitHub's activity log for the branch records the three pushes at 22:58:30,
   23:15:07 and 23:31:15 UTC on 26 October 2026, and the result files put the tests' starts 5, 3 and
   3 s later (file time minus the run time each records). Each test ran once; the guard does not
   enforce that.
6. **Smaller points.** Part B's one-minute bins are aligned to the clock, so "the first 5 minutes"
   reaches about 5.5 minutes (at most 6) — still before the target window. The look-ahead unit test
   covers Part A only (`tests/test_ml_ladder.py`), not Parts B and C, where these look-aheads were.
   README's "best model of all" in Part B (SVR + physics, 0.914) was picked on the test, and is now
   labelled so. The headline verdict for Part C now says that the declared bar was not met.
