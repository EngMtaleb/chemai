# Engineering Validation — Project 3

**RO membrane fouling and cleaning (CIP)**
What was decided, in what order, what the data then said, and what had to change

---

## Purpose

This file is the record behind every number in the project's README and on the live page. Each
section says what was decided before a result was seen, what the result was, and what changed
because of it. Negative results are kept with their numbers.

**Real data:** one published plant — Su, Meng & Chang (2026), *Membranes* **16**(8), 257,
doi:[10.3390/membranes16080257](https://doi.org/10.3390/membranes16080257); data and code
deposited at Zenodo, doi:[10.5281/zenodo.21055948](https://doi.org/10.5281/zenodo.21055948)
(deposit v1.3, data CC-BY-4.0, code MIT). The plant and its owner are not named by the authors and
are not named here.

**Order of work.** Simulated plants were split by plant and by seed before anything was scored:
development plants (seed 1001) chose every setting; test plants (seed 2002) checked them once; the
real deposit was scored once, after a written declaration. A further seed (3003) is used only for
illustrations, and one more (5005) for a check made after the external test (section 13). One plan
rule was added after the external test; section 12 says which, and how far its scores can be
trusted.

| Step | Script | Writes |
|---|---|---|
| 1 · read the deposit | `step1_audit.py` | `results/step1_audit.json`, two figures |
| 2 · physics | `step2_physics.py` | `results/step2_physics.json`, two figures |
| 3 · simulator against the deposit | `step3_simulator.py` | `results/step3_simulator.json`, two figures |
| 4 · choices, development plants | `step4_forecast_dev.py` | `results/step4_forecast_dev.json` |
| 4b · why the interval is drawn (recorded re-run) | `step4b_interval_check.py` | `results/step4b_interval_check.json` |
| 5 · test plants, once | `step5_forecast_test.py` | `results/step5_forecast_test.json` |
| 6 · real deposit, once | `step6_external_test.py` | `results/step6_external_test.json`, two figures |
| 7 · post hoc: a quieter simulator | `step7_matched_simulator.py` | `results/step7_matched_simulator.json` |

Run from the repository root, in this order. Step 1 downloads the deposit (about 100 kB) and checks
its md5; the later steps reuse it.

---

## 1. The question, and the answer it must have

A seawater RO plant cleans a train when its feed-channel pressure drop has risen enough. Clean too
late and the fouling hardens, the lead elements telescope, and the next cleaning recovers less.
Clean too early and the plant spends chemicals, downtime and membrane life for nothing. The
question a plant asks on Monday is not "what will DP_norm be on day 37?" but:

> **Which train do we clean, in which week, and why — and who acts?**

So the product is a **weekly plan**: one row per train with an action, a week, the evidence, a
confidence and an owner. The forecast date is one input to it, never the answer on its own.

| Category | Action | Owner |
|---|---|---|
| `limit` | manufacturer pressure-drop limit reached — clean now | operations |
| `review` | at or above the review level — decide this week | process engineering |
| `plan` | will reach it inside four weeks — plan a CIP | operations planning |
| `watch` | rising, further out — re-forecast next week | process engineering |
| `none` | flat or falling — no hydraulic reason to clean | – |
| `data` | too few recorded days to judge | instrumentation |
| `wait` | too soon after the last cleaning to judge | – |

**The review level is the plant's, not the model's.** The default, DP_norm 1.15, is the common
vendor rule of "+15 % since start-up". It is a parameter of `/forecast` and `/plan`; the results below are
scored at both 1.15 and 1.20.

---

## 2. The data: one published plant

**What is deposited**

| File | Content |
|---|---|
| `derived_daily_indicator_table.csv` | 1,340 train-days, three trains (A, B, C), day index 0–771 |
| `cip_campaign_ledger.csv` | 18 cleaning campaigns: train, day, chemistry stage |

Four daily indicators per train, each a fractional deviation from a baseline: net driving pressure
(`ndp_pct`), normalised feed-channel pressure drop (`dp_pct`), temperature-corrected specific flux
(`flux_pct`) and salt passage (`sp_pct`). The raw 10-minute SCADA series, the calendar dates and
the membrane element model are withheld by the authors; hashes are published for the raw series.

**What the article says about the pressure drop.** Normalised with a vendor flow ratio raised to
**1.5**, computed separately for each operating mode (the number of trains running), corrected
for viscosity (formula not given), against a **fixed commissioning baseline** that is never reset
at a cleaning. Intervals with fewer than two trains running are excluded. Each train has three
pressure vessels of eight elements.

**Coverage.** Days with a record, over each train's span:

| Train | Train-days | Coverage | Gaps > 7 days | Longest gap | Censored (zero) |
|---|---:|---:|---:|---:|---:|
| A | 381 | 49 % | 19 | 45 days | 18 % |
| B | 485 | 63 % | 12 | 44 days | 16 % |
| C | 474 | 61 % | 12 | 45 days | 12 % |

Half the calendar is missing for train A. Any method that needs a continuous record fails here
before it starts; that is why "too few recorded days" is a plan category of its own.

The deposit's own README names the journal the manuscript was first submitted to; the article
appeared in *Membranes* 16(8) 257 (doi:10.3390/membranes16080257, checked at the publisher).

The download is checked against the md5 printed on the record page
(`68b0223978f0e674d51cfc4c24e95435`); a different file is refused, not used.

---

## 3. What the files are — read before modelling

### 3.1 `dp_pct` is DP_norm − 1 on a fixed baseline — and a zero is not "clean"

Three sources disagree about the baselines:

| Source | Says |
|---|---|
| The deposit's column dictionary | all four indicators: fixed commissioning baseline, never reset |
| The article, and one deposited analysis script | `dp_pct` and `ndp_pct`: per-cleaning baselines |
| **The data** | `dp_pct` is **not** reset (median 0.167 in the days after a cleaning); `ndp_pct` **is** (median 0.006) |

The data side with the dictionary for `dp_pct` and with the article for `ndp_pct`. This project
therefore reads **DP_norm = 1 + dp_pct**, on the fixed baseline (`figures/step1_reset_check.png`).

Every indicator is clipped to [0, 1]. **A zero means "at or below the commissioning baseline"**,
not "clean": the value is censored there. 15 % of all train-days are zeros for `dp_pct` (36 % for
`ndp_pct`, 67 % for `flux_pct`). A recorded 1.00 is therefore an **upper bound** on the true value.
The forecast fits censored days at 1.00 like any other day; when more than a fifth of its window is
censored it warns that the level is an upper bound, and the date is marked as outside its tested
envelope (flagged).

### 3.2 Legs are not cleanings

The table marks 23 cleaning **legs** (an acid stage and an alkaline stage count twice); the ledger
lists 18 **campaigns**. Legs closer than 7 days collapse into one campaign: 23 → 18, matching the
ledger exactly (A: 3, B: 8, C: 7). The article's own crosswalk uses four different units for the
same 18 episodes (23 legs, 17 evaluable pairs, 14 complete-window campaigns, about 11
independent ones). Here the unit is the campaign.

### 3.3 Six impossible days

A 10-minute log holds at most 144 samples a day. Six plant days (18 train-days) hold more, up to
168. Duplicated or re-gridded records — **flagged, not repaired**; they are a small share and
nothing below depends on them.

### 3.4 The published numbers reproduce

| Published | Reproduced here |
|---|---|
| DP_norm recovers after 16 of 17 evaluable cleanings | **16 of 17** |
| Share of running time above 1.10 / 1.15 / 1.20: 0.65 / 0.47 / 0.30 | 0.672 / 0.487 / 0.313 |

Close, not identical: the article's exact daily basis is not deposited.

### 3.5 Cycles

The 18 campaigns cut each train's record into **cleaning cycles**: 3 initial cycles (from
commissioning), **15 complete cycles** (cleaning to cleaning) and 3 open ones at the end of the
record. Complete cycles last a median of 42 days (10th–90th percentile 23–138).

---

## 4. The unit of every score: the cleaning cycle

Days inside a cycle are not independent: the same fouling, the same cleaning, the same noise.
Scoring them as independent samples would multiply the evidence by about thirty. So:

- **Simulation is split by plant**, never by day: development, test and illustration plants are
  different simulated plants with different seeds.
- **Decisions are taken weekly** inside each cycle (day 14, 21, 28, …), as a planner would.
- **Real-data intervals are bootstrapped by cycle**, never by day.
- **The deposit is spent once.** No forecast setting was tuned on it; one plan rule was added after
  seeing it (section 12).

---

## 5. Physics: what the pressure drop measures

The feed-channel pressure drop is friction along a spacer-filled channel. With a friction factor
f ~ Re^−m,

> **ΔP ∝ Q_avg^(2−m) · μ^m · gap^(−n)**

| Term | Value | Source |
|---|---|---|
| m | 0.3 (empirical), ~0.2 (CFD on spacer channels) | Schock & Miquel (1987); CFD literature |
| **flow exponent 2 − m** | **1.7–1.8** | physics |
| vendor flow exponent | 1.0–1.5 | manufacturers' normalisation bulletins |
| viscosity μ(T) | Vogel equation, 0.89 mPa·s at 25 °C | standard |
| gap exponent n | 2.2 (CFD) to 3 (slit) | section 7 |

The same package carries the rest of the ASTM-style normalisation (temperature correction factor,
osmotic pressure, log-mean concentration factor, net driving pressure, normalised permeate flow and
salt passage), tested against textbook values: seawater osmotic pressure about 27 bar, water
viscosity 0.89 mPa·s at 25 °C.

**The plant can measure its own exponent.** Taking logs and day-to-day differences,

> d ln ΔP − 2 d ln μ = n · (d ln Q − d ln μ)

so every change in production is a free experiment. The fit refuses when production barely moved
(less than 3 % spread): steady operation cannot reveal the exponent.

---

## 6. Normalisation decides the answer

**Decided before:** compare five ways of normalising the same simulated pressure drop
(15 illustration plants), on how much fouling each one *invents* when production and temperature
change and the membrane does not.

| Normalisation | Invented fouling, SD | 5–95 % band |
|---|---:|---:|
| linear flow ratio (exponent 1.0) | **12.6 %** | −16.6 to +22.4 % |
| vendor (1.5) | 4.8 % | −6.4 to +8.0 % |
| vendor, no viscosity correction | 4.8 % | −7.3 to +7.0 % |
| physics (1.75) | 2.7 % | −3.9 to +3.9 % |
| **the plant's own fitted exponent** | **2.7 %** | −3.9 to +4.0 % |

The review level is +15 %. A vendor normalisation moves the signal by up to 8 % with nothing
fouling — half the distance to the level (`figures/step2_normalisation.png`).

**Can a plant fit its own exponent?** On 45 simulated trains (true exponent 1.71–1.81), from the
first 120 recorded days: mean absolute error **0.032** with a differential pressure transmitter,
**0.077** when ΔP is the difference of two pressure transmitters.

**What it costs the forecast** (development plants, section 9; median absolute error of the
crossing date):

| Normalisation | Median error | Crossings dated |
|---|---:|---:|
| linear ratio | 32.7 days | 1,284 of 4,742 |
| vendor 1.5 | 28.0 days | 3,778 of 7,211 |
| physics 1.75 | 18.9 days | 5,493 of 8,004 |
| **own fit** | **20.0 days** | 5,460 of 7,964 |

**No model choice in this project moved the answer as much as the normalisation did.**

**Temperature on its own:** ±5 K changes viscosity by about 11 % and the pressure drop by about
3 % — a seasonal swing a fifth the size of the review margin.

**Instrumentation.** On the test plants, ΔP taken as the difference of two pressure transmitters
instead of a differential transmitter: median date error **30.4 days against 19.4**; for "will it
cross within 28 days?", recall 0.20 at precision 0.43 against 0.51 at 0.78. A differential
transmitter is worth more than any model in this project.

---

## 7. The gap law: physically right, worth a day or two

As fouling closes the feed channel, the pressure drop accelerates: ΔP ∝ gap^(−n). With n = 2.2,
losing 10 % of the gap raises ΔP by 26 %. If fouling starts on the lead element, the curve is more
convex still.

**But at the levels plants clean at, it barely bends.** Between DP_norm 1.00 and 1.30 (11 % of the
gap blocked), the curve sags at most **0.015** DP_norm below its straight chord
(`figures/step2_gap_law.png`).

The forecast can fit its line through the transform DP_norm^(−1/n) instead of DP_norm. Declared
before the external test as a **secondary** analysis:

| | Linear (adopted) | Gap-law transform |
|---|---:|---:|
| development plants, median error | 20.0 days | 19.1 days |
| test plants | 19.4 days | 17.8 days |
| real deposit (13 cycles) | 28.4 days | 26.5 days |
| real deposit, level 28 days ahead (MAE) | 0.053 | 0.057 |

Always one to two days better on dates — and, on the simulated plants, possibly by construction,
because the simulator itself uses the gap law. The real deposit is the first evidence that is not
circular; 13 cycles cannot separate two days from chance. **Not adopted**; available as
`law="blockage"`.

---

## 8. The simulator: where the ground truth comes from

A real plant never shows what would have happened without a cleaning. A simulator does: the
**counterfactual path** — the same train, the same fouling, no cleaning — gives the true crossing
day for every forecast.

**What is in it.** Three trains of three 8-element vessels, 770 days. Lumped solution–diffusion
transport with concentration polarisation; seasonal temperature; production that steps as trains
start and stop (a day is recorded only when two trains run, as in the deposit). Three fouling
laws, ten plants each: linear, biofilm (accelerating, faster in warm water) and asymptotic
(levelling off). Upsets about three times a year. Cleanings that remove 55–95 % of the fouling,
leave an irreversible share, and relax for days afterwards. An operator who cleans on what the
screen shows, near 1.15–1.35 with a delay, plus cleanings for other reasons. A differential
transmitter with noise and outliers, or two pressure transmitters that drift apart. Readings are
normalised and censored at the baseline the way the deposit is (`figures/step3_simulated_plants.png`).

**How it compares with the deposit** (15 complete real cycles, 254 simulated, first six weeks of
each; `figures/step3_sim_vs_real.png`):

| Median | Deposit | Simulator | Simulator ÷ deposit |
|---|---:|---:|---:|
| cycle length | 42 days | 118 days | **2.8×** |
| slope | 0.0030 /week | 0.0050 /week | 1.7× |
| scatter about the line | 0.015 | 0.035 | 2.4× |
| jump between consecutive days | 0.011 | 0.024 | **2.2×** |
| cycles with a falling slope | 27 % | 19 % | |

**Said plainly:** the fouling-rate ranges were set after looking at the deposit's slopes in
step 1. Cycle lengths and noise were **not** matched before the settings were frozen, and this
comparison was written after the external test. The simulated plants are noisier and clean later
than the real one. Sections 9 and 10 therefore describe a harder-noise, longer-cycle world;
section 13 checks what changes in a quieter one.

---

## 9. Development choices (30 plants, seed 1001)

**The forecast.** A robust straight line (Theil–Sen) through the current cycle, skipping the three
flushing days after a cleaning; the crossing date of the review level is read off the line.

**Grid, decided before it was run:** normalisation (4) × law (linear, gap law) × trailing window
(28, 42, 63 days), weekly decisions from day 14, levels 1.15 and 1.20.

| Own fit, linear law | 28 days | **42 days** | 63 days |
|---|---:|---:|---:|
| median error | 19.7 | 20.0 | 20.9 |
| crossings dated | 4,942 | **5,460** | 5,552 |
| median bias (late +) | +4.7 | +7.6 | +10.0 |

The window barely matters: 42 days dated 10 % more crossings than 28 at the same error. The law is
section 7.

**The interval.** The first version bootstrapped the residuals. An exploratory Monte Carlo check
rejected it; that run was not saved, so `step4b_interval_check.py` repeats it on record (written
after the external test; no plant data). A known straight line plus AR(1) noise, fitted on days
3–42 like the forecast; the share of the true slope spread each method recovers:

| Lag-1 autocorrelation | 0 | 0.3 | 0.6 |
|---|---:|---:|---:|
| residual bootstrap, independent draws | 84 % | 63 % | 41 % |
| residual bootstrap, blocks of seven days | 66 % | 68 % | 58 % |
| drawn lines, AR(1) sample size, no inflation | 93 % | 88 % | 78 % |

A cycle is too short and too autocorrelated to bootstrap. Replaced by drawing plausible lines
around the fit, with the sample size shrunk for lag-1 autocorrelation, and one inflation factor
calibrated on development plants — on **crossing dates**, not slopes. On a pure straight line the
factor of 2.0 widens the slope spread to 1.6–1.9 times the truth; the extra covers what a straight
line cannot see (curvature, the wrong fouling law):

| Inflation | 1.0 | 1.5 | **2.0** | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|
| P10–P90 coverage (target 0.80) | 0.658 | 0.738 | **0.800** | 0.851 | 0.886 |

**The flag.** A date is flagged — marked outside the tested envelope — when it lies further ahead
than twice the days the line has seen, or when more than a fifth of the window is censored.
Section 10 shows why.

**The decision rule.** "Plan a CIP when the P50 date is within 28 days", compared with the simplest
rival: "plan when the 7-day median is within a margin of the level". For "will it cross within 28
days?":

| | Forecast, P50 ≤ 28 | Margin 0.03 | Margin 0.05 |
|---|---|---|---|
| own fit | recall 0.49, precision **0.75** | 0.32, 0.78 | 0.51, 0.70 |
| vendor 1.5 | 0.30, 0.45 | 0.37, 0.56 | 0.50, 0.52 |

With the plant's own exponent the forecast edged ahead of the margin rule; with the vendor's it
lost. **Frozen:** linear law, 42-day window, inflation 2.0, flag at 2×, plan at P50 ≤ 28 days.

---

## 10. Test plants (30 plants, seed 2002) — scored once for the forecast

(Their saved scores were reused after the external test to check the plan rule of section 12.)

| Normalisation | Median error | Dated / crossed | Coverage | Precision at recall 0.5: forecast · margin |
|---|---:|---:|---:|---|
| **own fit** | **19.4 days** | 5,477 / 7,968 | 0.78 | **0.78** · 0.74 |
| physics 1.75 | 18.5 days | 5,478 / 7,930 | 0.78 | 0.80 · 0.75 |
| vendor 1.5 | 23.4 days | 3,533 / 6,945 | 0.85 | never reaches 0.5 · 0.61 |
| linear 1.0 | 30.9 days | 1,280 / 5,498 | 0.92 | never reaches 0.5 · 0.33 |
| two pressure transmitters | 30.4 days | 2,155 / 5,774 | 0.88 | – |

The settings held: coverage 0.78 against a target of 0.80. Wide coverage on the bad normalisations
is not a virtue — their intervals are wide because their signal is noisy.

**By fouling law** (own fit):

| Law | Median error | Median bias | Coverage |
|---|---:|---:|---:|
| linear | 13.6 days | +10.5 days late | 0.78 |
| **biofilm (accelerating)** | 25.6 days | **+23.2 days late** | **0.65** |
| asymptotic | 22.9 days | −7.2 days early | 0.91 |

**A straight line is late on accelerating fouling.** On biofilm-like fouling the P50 date came a
median 23 days after the true crossing, and the interval held only 65 %. For a train that has
fouled faster than linearly before, plan by the P10, not the P50.

**The extrapolation flag, measured** (median error by how far ahead the date lies, in multiples of
the window the line has seen):

| Days ahead ÷ days seen | ≤ 0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–3 | 3–5 | > 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| forecasts | 1,177 | 1,108 | 875 | 608 | 822 | 680 | 200 |
| median error, days | 6.3 | 11.9 | 20.0 | 28.8 | **41.4** | **71.5** | **80.2** |

Flagged dates (either reason) missed by a median of 51.8 days, the rest by 12.2. By the
twice-the-window rule alone: 55.2 against 12.5 days.

---

## 11. The external test: the real deposit, once

### 11.1 Declared before the run

`results/step6_declared_before_run.md` was written after steps 4 and 5 and before step 6 touched
the deposit's forecasts: the frozen settings, linear law primary, gap law secondary, and this
expectation — *"the deposit's DP_norm was normalised by its authors with a vendor-style flow
exponent (1.5, per operating mode) … the same may happen here. A forecast that does not beat the
margin rule on the deposit will be reported as such."* (The repository history cannot prove the
order: everything was committed together. The file is the record.)

**Truth on real data.** The crossing day is the first day the 7-day median reaches the level. If
a cleaning came first, the truth is **censored** — "not before day x" — and scored as such.

### 11.2 Crossing dates

| | Level 1.15 | Level 1.20 | Both |
|---|---:|---:|---:|
| decisions with enough data | 110 | 135 | 245 |
| crossings observed | 75 | 76 | 151 |
| **dated by the forecast** | 25 | 29 | **54** |
| median error, days | 36.4 | 24.7 | **28.4** |
| P10–P90 coverage | 0.85 | 0.92 | **0.89** |
| censored truths consistent with the forecast | 1.00 | 0.74 | 0.82 |
| median error: unflagged · flagged | 11.6 · 67.5 | 13.2 · 57.0 | **12.5 · 59.2** |

**Most real crossings were not dated at all.** Of 151, 85 came after a "not in sight": DP_norm was
sitting on the censored floor (median 1.007), the line had no slope to extend, and the train crossed
a median of 75 days later. A forecast cannot see a rise that has not started.

**The dates it gave were late.** 69 % of dated crossings came before the P50 (median +18 days). The
lateness sits in the flagged dates: unflagged ones were nearly unbiased (+3 days, n = 30), flagged
ones +59 days (n = 24; 18 of them for looking beyond twice the window, 6 only for a mostly censored
window). By the twice-the-window rule alone: 75 days against 17 (n = 18 / 36). The flag did its job
here too.

**And 41 % of all weekly decisions had too few recorded days to forecast at all** — the gaps of
section 2.

### 11.3 Level forecasts against "it stays where it is"

| Ahead | Decisions (cycles) | Forecast MAE | Persistence MAE | Difference, 90 % CI by cycle |
|---|---:|---:|---:|---|
| 14 days | 113 (15) | 0.0338 | 0.0342 | −0.009 to +0.009 |
| 28 days | 85 (7) | 0.0525 | 0.0558 | −0.024 to +0.028 |

**No better than assuming tomorrow equals today.** The forecast was better in 41 % of decisions at
14 days.

### 11.4 The decision: will it cross within 28 days?

213 decisions with a known answer (11 cycles), 61 of them crossed:

| Rule | Caught | False alarms | Recall | Precision |
|---|---:|---:|---:|---:|
| forecast, P50 ≤ 14 days | 10 | 3 | 0.16 | 0.77 |
| **forecast, P50 ≤ 28 days (frozen)** | 13 | 5 | **0.21** | **0.72** |
| forecast, P50 ≤ 42 days | 16 | 10 | 0.26 | 0.62 |
| margin 0.02 | 21 | 1 | 0.34 | 0.95 |
| **margin 0.03** | 26 | 2 | **0.43** | **0.93** |
| margin 0.05 | 41 | 7 | 0.67 | 0.85 |
| margin 0.08 | 47 | 19 | 0.77 | 0.71 |

**The forecast lost to the margin rule at every operating point.** The declaration had named one
reason in advance: the deposit's pressure drop carries a vendor normalisation. A second was found
afterwards (sections 8 and 13): the real cycles are short and quiet, so a train near the level
crosses on its day-to-day swings before any trend matters.

### 11.5 The weekly plan, replayed against the operators

The plan was replayed every 7 days over the whole record (`figures/step6_plan_replay.png`), with the
rules as finally adopted (section 12). What it said in the week before each of the 18 real
cleanings:

| Plan the week before | Cleanings |
|---|---:|
| review (at or above the level) | 12 |
| plan | 1 |
| wait (a separate campaign 18 and 21 days after the previous one) | 2 |
| watch | 1 (C#1; last week’s median 1.08) |
| none — no crossing in sight (slope slightly negative) | 2 (B#5 at 1.11, B#6 at 1.04) |

**Three cleanings had no hydraulic reason.** The deposit does not record why a train was cleaned.
A look at the other indicators, made after the test, over the month before each: before C#1, salt
passage stood a median 59 % above its baseline (48–100 % on all 15 recorded days), with specific
flux a median 7 % and at most 18 % off — a reason the plan, which reads only the pressure drop,
cannot see. Before B#5 and B#6 the medians had not moved (salt passage +6 %, flux 0), though single
days reached 12 % on flux and 19 % on net driving pressure before B#6.

Over the whole replay, a third of weekly rows (32 %) said `data` — check logging and
transmitters — because of the gaps.

---

## 12. After the external test: the margin trigger (post hoc)

**What changed.** Rule 4 of the plan now has two triggers: a forecast crossing inside four weeks,
**or** a 7-day median within **0.03** of the review level ("day-to-day swings will reach it",
planned within two weeks).

**Why it is honest to keep it.** The change is a rule a planner can read, not a fit, and it was
checked on simulated plants — with a caveat. The development and test plants were **reused** for
this check, and the simulator's fouling rates were set from the deposit's slopes (section 8); the
quieter variant was built to resemble the deposit (section 13). Only the quieter plants are new to
the combined rule, and there it gained 2 points of recall:

| "Crosses within 28 days" | Forecast alone | Margin alone | **Both (adopted)** |
|---|---|---|---|
| development plants | R 0.49 · P 0.75 | R 0.32 · P 0.78 | **R 0.58 · P 0.74** |
| test plants | R 0.51 · P 0.78 | R 0.30 · P 0.78 | **R 0.60 · P 0.76** |
| quieter simulator (section 13) | R 0.50 · P 0.78 | R 0.18 · P 0.95 | **R 0.51 · P 0.79** |
| real deposit (post hoc) | R 0.21 · P 0.72 | R 0.43 · P 0.93 | R 0.51 · P 0.84 |

On every simulated set, adding the margin raised recall — by 9 points on the reused development
and test plants, 2 on the new quieter plants — at nearly the same precision. **The real-data row is
optimistic:** the rule exists because of the deposit's result, so the deposit is no longer an
untouched test of the plan. It needs a second plant.

`test_a_train_hovering_just_below_the_level_is_planned` keeps it from disappearing.

---

## 13. Post hoc: a simulator closer to the deposit

Section 8 found the simulator 2.2× noisier than the deposit, with cycles 2.8× longer. One variant
was written down before it was scored, and not tuned afterwards: day-to-day noise and outliers cut,
operators cleaning earlier (near 1.10–1.25, at most 45–120 days apart). New plants, seed 5005.

| Median | Deposit | Original simulator | Quieter variant |
|---|---:|---:|---:|
| cycle length | 42 days | 118 days | 54 days |
| jump between consecutive days | 0.011 | 0.024 | 0.011 |

| Own fit | Test plants (section 10) | Quieter variant |
|---|---:|---:|
| median error | 19.4 days | 22.5 days |
| P10–P90 coverage | 0.78 | **0.69** |
| median error, unflagged · flagged | 12.2 · 51.8 | 12.1 · 40.9 |
| precision at recall 0.5: forecast · margin | **0.78 · 0.74** | **0.78 · 0.81** |
| vendor 1.5, same | never · 0.61 | 0.40 · 0.55 |

**What survives:** normalisation matters as much as before; the flag still separates
good dates from bad. **What does not:** the forecast's edge over the margin rule — in a world as
quiet as the deposit, the margin rule edges ahead, as it did on the real data. **And the interval
calibration does not transfer:** 0.78 on the test plants, 0.69 here, 0.89 on the deposit. The
inflation factor absorbs the gap between a straight line and a curved truth, and that gap depends on
the noise. **Treat P10–P90 as a rough band, not a calibrated one.**

---

## 14. Errors found, and the tests they became

| Error, as found | Test |
|---|---|
| a temperature in kelvin accepted as Celsius | `test_rejects_kelvin_given_as_celsius`, `test_normalise_refuses_kelvin` |
| DP_norm or the review level sent in percent (15 for 1.15) | `test_dp_norm_sent_in_percent_is_refused`, `test_a_review_level_in_percent_is_refused` |
| a zero in the deposit read as "clean" | `test_a_zero_is_censored_not_clean`, `test_a_mostly_censored_window_warns`, `test_the_view_censors_at_the_baseline_like_the_deposit` |
| cleaning legs counted as cleanings | `test_legs_are_read_back_from_the_table_and_collapse_into_campaigns` |
| days with more than 144 ten-minute samples | `test_more_than_144_ten_minute_samples_in_a_day_is_flagged` |
| a silently different download | `test_download_refuses_a_file_with_the_wrong_md5` |
| a linear flow ratio reading production as fouling | `test_a_linear_flow_ratio_reads_a_production_increase_as_fouling` |
| an exponent "measured" from steady production | `test_steady_production_cannot_measure_the_exponent` |
| unbalanced flow meters normalised anyway | `test_normalise_returns_one_at_start_up_and_rejects_unbalanced_flows` |
| a cycle that spans a cleaning | `test_cycles_are_cut_at_cleanings_and_never_span_one`, `test_cycles_never_span_a_cleaning` |
| the flushing transient fitted as fouling | `test_the_flushing_days_after_a_cleaning_are_skipped` |
| a bootstrap interval too narrow | `test_the_interval_contains_the_date_and_widens_with_noise` |
| an expected gap-law effect that was wrong (3.6 days, not under 3) | `test_the_gap_law_bends_earlier_but_only_by_a_tenth_at_these_levels` |
| trains already above the level counted as perfect forecasts | `test_decisions_already_above_the_level_are_not_scored` |
| a cleaning before the horizon counted as "did not cross" | `test_a_cleaning_before_the_horizon_leaves_the_answer_unknown` |
| two trains planned into one week below the supply minimum | `test_two_trains_are_not_cleaned_in_one_week_when_supply_needs_two` |
| a flat train just below the level left unplanned | `test_a_train_hovering_just_below_the_level_is_planned` |
| the headline failure listed last | `test_health_leads_with_the_real_data_result` |

---

## 15. Limits

1. **On the one real plant, the date forecast did not beat a simple margin rule** (section 11.4),
   and its level forecasts were no better than persistence (11.3). The plan uses both triggers; the
   union's real-data score is post hoc (12).
2. **One plant.** Three trains, 18 cleanings, 15 complete cycles. Not a validation.
3. **Blind on the floor.** While DP_norm sits at the commissioning baseline there is no trend to
   extend; most real crossings came after a "not in sight" (11.2).
4. **Late on accelerating fouling.** Biofilm-like fouling crossed a median 23 days before the P50
   on simulated plants (10). Plan by the P10 for a train with that history.
5. **The interval is rough:** 0.78, 0.69 and 0.89 coverage in three worlds (13).
6. **Hydraulic only.** Scaling of the tail elements and membrane damage show first in permeate flow
   and salt passage; one of the three unexplained real cleanings followed a month of salt passage
   about 60 % up (11.5).
7. **Normalisation decides the answer** (6). Measure the plant's own exponent before trusting a
   date; prefer a differential transmitter to two pressure transmitters.
8. **Upsets** — a bloom, a pretreatment failure — change the rate overnight; no line sees them
   coming.
9. **Gaps.** A third of the real weekly plans could not forecast for lack of recorded days.
10. **The review level and the limits are inputs.** 1.15 is a vendor convention; the 1.0 bar per
    element and 3.4 bar per vessel limits are typical 8-inch values — replace both with the plant's
    own.

---

## 16. What this is, and what it is not

**What this is:** a deployed prototype, complete from a public deposit to a service anyone can
call — physics-based normalisation, a forecast that says how far it can see, and a weekly plan with
an action and an owner per train — tested once on a real plant, with the result reported as it
came out.

**What it is not:** a plant system. No historian connection, no authentication, no monitoring, no
field validation, and no engineer has yet acted on one of its dates. One published plant is the
only real evidence behind it.
