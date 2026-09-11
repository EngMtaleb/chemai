# Engineering Validation — Project 2

**Control Loop Performance Assessment**
Data-stage findings, recorded before modelling begins

---

## Purpose

This file records what was found in the data, what it constrains, and what the resulting
system will and will not be able to claim. It is written **before** the model exists, because
three of the findings below would silently corrupt any result built on top of them.

**Data source:** SACAC PID Data Repository — `sacac.org.za/resources/`
Described in Bauer et al. (2019), *Ind. Eng. Chem. Res.* **58**, 11430–11439.

---

## 1. Inventory

| Category | Files | Independent loops |
|---|:---:|:---:|
| Tuning | 14 | **3** |
| Stiction | 13 | **7** |
| Unknown | 6 | — |
| **Other** | 4 | **3 labelled** — 2 healthy · 1 external disturbance (see below) |
| Plant-wide | 3 CSV + 3 `.mat`/`.xls` | **6 datasets** (the 46 counts CSVs only) |
| Quantization | 3 | — |
| Saturation | 2 | — |
| Sensor faults | 1 | — |

The single-loop files use `time · SP · PV · OP`, semicolon-separated, but the loader must
handle every exception below — **read by name, case-insensitive, never by position**:

| Exception | Files |
|---|---|
| Header case varies (`time`/`Time`, `error`/`Error`) | many |
| Extra error column | 7 |
| `PV` before `SP` | `sensor-F-oilgas-thornhill-2007`, `unknown-P-oilgas-thornhill-2007-1/2` |
| `PV` only — no `SP`, no `OP` | `quantisation-T-chemicals-thornhill-2003` |
| Bare `\r` line endings | many — normalise before parsing |
| Comma-separated, plant tag names | the three plant-wide CSVs |

**The `time` column is not a clock.** The `baccidicapaci` files carry restyled timestamps with
minute resolution (true sampling 1–30 s, or non-constant with a 12 s mean);
`tuning-L-paper-horch` steps irregularly by 1–4 s. Treat `time` as a sample index; resample
where sampling is irregular.

**`Other` is not noise — it holds the only non-faulty references:**

| File | Published description | Use |
|---|---|---|
| `other-F-paper-horch-2003-2` | normal operation with a setpoint change · dead time 7–8 s | healthy |
| `other-L-paper-horch-2003` | normal operation · dead time 4 s | healthy |
| `other-F-paper-horch-2003` | **oscillation from an external disturbance** · dead time 3–8 s | the class most often mistaken for stiction |
| `other-F-chemicals-thornhill-2003` | non-periodic transient disturbance | excluded from shape tests |

These three `horch` files are also **the only real loops with a published dead time** — the one
input the Harris index cannot do without.

---

## 2. ⭐ The measurement that is missing — `MV`

A control loop carries four signals:

| Signal | Meaning | Recorded here |
|---|---|:---:|
| `SP` | setpoint — what we want | ✅ |
| `PV` | measurement — what happens | ✅ |
| `OP` | controller **command** to the valve | ✅ |
| **`MV`** | **actual valve stem position** | ❌ |

**Stiction is, by definition, the gap between `OP` and `MV`** — the valve does not move even
though the command changes.

The classical diagnostic signature — the parallelogram — appears in a plot of **`OP` against
`MV`**. That plot cannot be produced from this dataset.

### What we plot instead, and what it costs

```
OP  →  valve (MV)  →  process  →  PV
            ↑                ↑
      stiction lives   further distortion
         here             here
```

Using `OP` against `PV` inserts the **process dynamics** between the two signals:

| Process type | Effect on the shape |
|---|---|
| Fast, self-regulating (flow, pressure) | passes the shape through — **approximates the parallelogram** |
| **Integrating (level)** | ⚠️ **the vessel integrates**: a square wave becomes triangular, and the `OP–PV` plot becomes an ellipse |

**This explains a result that first looked like a labelling error.** `stiction-L-thornhill`
shows an ellipse in `OP–PV` while its `PV` is clearly triangular — the label is correct; the
tank integrated the signal.

> **Consequence for feature design:** look for the triangular signature in **`OP`** for
> flow and pressure loops, and in **`PV`** for level loops. The loop type is encoded in the
> filename (`F`, `P`, `L`).

> **Engineering recommendation, independent of any model:** valve position feedback is
> available on smart positioners and is not recorded here. **Logging `MV` would make stiction
> detection direct rather than inferred.**

---

## 3. ⚠️ Data defect — 26% of one file belongs to another

`tuning-L-paper-horch-2003.csv` — rows **846 to 1146** are value-identical to the same rows of
`stiction-L-paper-horch-2003.csv`, across `time`, `SP`, `PV` and `OP`. Maximum absolute
difference: **0.000**.

The discontinuity is visible in the raw file:

| row | time | SP | PV | OP |
|---:|---:|---:|---:|---:|
| 845 | 1198 | 50.084 | 51.038 | 60.297 |
| **846** | **1694** | **37.014** | **36.688** | **26.031** |

Time jumps from 1198 to 1694 and the setpoint jumps to the stiction file's value.

**Action:** the loader excludes rows ≥ 846 as a documented data-quality rule. **The raw file is
not edited.** 301 of 1147 rows — 26% of a labelled example — carry the wrong label.

**What remains is weak evidence:** 846 rows, irregular sampling, about four oscillation cycles
(period ≈ 298 s) — too few for the regularity index to be computed.

All 46 files were checked for repeated segments; this is the only occurrence.

---

## 4. ⚠️ The `Tuning` label covers two opposite faults

| Sub-group | Files | Behaviour |
|---|:---:|---|
| `tuning-F-chemical-DB-*` | 12 | **sluggish** tuning — the description says so explicitly |
| `tuning-L/Q-paper-horch` | 2 | **tight** tuning — produces the classic sinusoid |

**Sluggish tuning does not produce a limit cycle at all.** A regularity index (Thornhill et al., 2003)
returns 0.08–0.38 on 8 of the 12 sluggish files, against a threshold of 1; the other 4 have too
few autocorrelation zero-crossings for the index to be computed. Neither is an oscillation.

What appears as a diagonal random walk in those files is `PV` following an `SP` that changes
in 87% of samples — most likely the **inner loop of a cascade**.

> **These must not be one class.** A classifier trained on `Tuning` as a single label is being
> asked to learn two opposite conditions from the same target.

---

## 5. ⚠️ 46 files are not 46 independent samples

| Category | Files | Actual loops |
|---|:---:|:---:|
| Stiction | 13 | **7** — `stiction-P-oilgas-DB-1…7` are seven runs of **one** loop |
| Tuning | 14 | **3** — twelve files are the same sluggish loop |

**Splitting by file would place runs of the same loop on both sides of the split.** The
classifier would learn *"loop DB"* rather than *"stiction"*, and the reported score would not
reflect performance on a new loop.

> **Required: group-aware splitting (`GroupKFold`) with the loop as the group** — applied to
> the simulated data (group = one simulated loop, all seeds together). On SACAC, results are
> reported **per loop**: the seven `DB` stiction runs count as one loop, and the spread of the
> prediction across them becomes a **consistency test**, not seven extra points.

This is the same failure mode as the `simulationRun` leakage in Project 0 and the
campaign split in Project 1 — **the third appearance of one question: what makes two rows
similar for reasons other than the one being studied?**

---

## 6. ⚠️ Source imbalance is a second leakage path

| Source | Stiction | Tuning |
|---|---|---|
| `baccidicapaci-2018` | 10 files · 4 loops — pressure ×3 (oil & gas, chemicals), level ×1 (power) | 12 files · 1 loop — **flow, sluggish** |
| `horch-2003` | 2 files — flow, level | 2 files — level, quality · **tight** |
| `thornhill-2003` | 1 file — level | — |

**Loop type is confounded with the label:** all 9 pressure files are stiction; 12 of 13 flow files
are sluggish tuning.

**A model could learn "pressure ⇒ stiction, flow ⇒ tuning" instead of learning the shape.**

> **Loop type selects the signal (§2) — it is never a model input.** Feeding it as a feature
> hands the model the shortcut.

### ⚠️ Source hold-out and the Tuning split cannot both be satisfied on SACAC

`tight` tuning exists only in `horch`; `sluggish` only in `baccidicapaci`. Any split by source puts
each tuning class entirely on one side — the model is tested on a class it never saw, and the
class it trained on never appears in the test.

> **Resolution: SACAC is never training data.** Train on simulation (weeks 1–2), where loop
> type, source and label can be decorrelated by design. SACAC is the **external test set** in
> week 3 — 13 labelled loops (7 stiction · 2 tight · 1 sluggish · 2 healthy · 1 external
> disturbance), reported loop by loop. Three of them share one oscillation (§11.6): **11 independent events**.

---

## 7. ⚠️ The quantitative index has its own limit

A curve-fitting stiction index (He et al., 2007) fits each half-cycle with a sinusoid and with
a triangle; above 0.6 indicates triangular, below 0.4 sinusoidal.

Behaviour on synthetic signals:

| Signal | Index |
|---|---:|
| sinusoid | 0.15 |
| triangle | 0.82 |
| sawtooth | 0.97 |
| **square wave** | **0.42** ⚠️ |

**A square wave reads as a sinusoid.** This is why the index is applied to `OP` on fast loops —
their `PV` tends toward square.

**Noise pulls the index toward 0.5.** At noise of 0.3 × amplitude: sinusoid 0.47, triangle 0.66 —
the sinusoid enters the ambiguous band.

**This is our implementation of the He et al. method, not a reference code.** The low-pass
cutoff (7th harmonic), the drift window and the minimum segment length (6 samples) all move the
result; they are reported with every index value.

> **The index is a feature, not ground truth.** Where it disagrees with the published label,
> we do not yet know which is wrong. **The label remains the reference.**

---

## 8. What the shape test actually showed

**Stiction:** a triangular signature is present in **12 of 13 files** — but not always in the
same signal:

| Loop type | Signal carrying the triangle | Index |
|---|---|---|
| `F`, `P` — fast, self-regulating | **`OP`** | 0.63–0.87 across all ten files |
| `L` — integrating | **`PV`** | Horch 0.85 · Thornhill 0.71 · Power 0.51 *(ambiguous)* |

**Tuning:** the smooth sinusoid appears in **2 of 14** — the two `tight` files. Expected, given §4.

### Why a first, naive plot hid the effect

1. **Shared axis.** `PV` near 110 and `OP` near 14 plotted on one axis flattens both.
2. **Raw `PV` against a moving `SP`.** In the `P-oilgas-DB` files the setpoint moves across
   about 60% of the `PV` range. Subtracting `SP` and detrending makes `DB-3` a clear parallelogram (after low-pass filtering at
   the 7th harmonic — the filter also rounds the corners).
3. **First 800 samples only** — 11% of the DB files, and about two and a half cycles of Thornhill.

> The moving `SP` and the irregular sampling are properties of the data; **not handling them**
> was the plotting choice.

**One stiction run fails the oscillation gate:** `stiction-P-oilgas-DB-6` has regularity 0.98 (< 1),
and `DB-1`, `DB-2`, `DB-5` sit at 1.2–1.3. A pipeline that runs shape analysis only on loops that
pass oscillation detection will skip at least one labelled stiction case.

---

## 9. Constraints carried into modelling

| Decision | Basis |
|---|---|
| Loader reads columns **by name**, case-insensitive, tolerates missing columns, treats `time` as index | §1 |
| Loop type **selects the signal** (`F`/`P` → `OP`, `L` → `PV`) — **never a model input** | §2, §6, §8 |
| `tuning-L-paper` rows ≥ 846 excluded **in the loader**; raw file untouched | §3 |
| `Tuning` split into `sluggish` and `tight`; `healthy` and `external disturbance` added | §1, §4 |
| **Train on simulation, `GroupKFold` by simulated loop; SACAC = external test, per loop** | §5, §6 |
| Harris reported as a **range over the plausible delay**, never one number | §11.4 |
| **No Harris ranking of averaging level loops** | §11.5 |
| The three coherent `horch` loops reported **as one event** | §11.6 |
| ISDB loops already in SACAC counted **once**; label conflicts reported, not resolved | §12.3 |
| Cascade slaves are **diagnosed, not scored** with Harris | §13.3 |
| Every loop passes the five data-quality checks **before** diagnosis; flags, never repairs | §14 |
| PI-law R² is a **note**, never an exclusion | §14.5 |
| Loader returns a sampling time **only when description and data agree** | §11.6 |
| Stiction index used as a **feature**, not as a label | §7 |
| Detrend against `SP` before shape analysis | §8 |

---

## 10. Limitations to state before any result

1. ⭐ **`MV` is not recorded.** The stiction signature is inferred through process dynamics,
   not measured directly. Accuracy therefore depends on loop type.
2. **Seven stiction loops and three tuning loops.** Small, and unevenly distributed across
   sources and loop types.
3. **Labels are expert attributions made after the fact**, not measurements. Disagreement
   between a model and a label is not automatically the model's error.
4. **These datasets are curated.** Published benchmark sets are cleaner than plant historian
   data; good results here do not guarantee good results on a live archive.
5. **Plant-wide:** six datasets — three CSV (distillation · flotation · a composition-sensor
   failure that is not a propagation case) and three `.mat`/`.xls` (Eastman chemicals · two
   refineries). Only Eastman has a published root cause — and **that root cause (Tag 22, LC2) is
   the same loop as `stiction-L-thornhill`** (column 22 of the `.mat` file, matched value for
   value). A stiction result on that file and a root-cause result on Eastman are not independent.
6. **Archive compression cannot be reversed.** Its effect can be studied by compressing clean
   data, not by recovering what a historian already discarded.
7. **No cascade-specific performance index.** Joint assessment of master and slave with a
   cascade Harris index is advanced research, out of scope (§13.3).

---

## 11. Week 1 — simulator and Harris index

Code: `chemai/data/loop_sim.py` · `chemai/features/loop_performance.py` · `chemai/data/sacac.py`.
Numbers below: `projects/p02_control_loops/week1.py` → `results.json`, seeded, 120 plants × 2 seeds.
**Reference environment:** Python 3.14, NumPy 2.5.3, aarch64 (WSL). Other environments reproduce the
conclusions, but borderline runs near the oscillation threshold can flip — a few fractions differ by
up to ~0.1. Published numbers come from this environment only.

### 11.1 Design decisions the simulator forced

| Decision | Why |
|---|---|
| **Gain margin computed on the discrete loop as simulated** | The continuous gain margin ignores sampling lag; a target of 1.15 could be unstable once discretised. Checked against the simulator itself: 3% below the computed margin the loop is stable, 3% above it diverges — for F, P and L (`test_gain_margin_matches_simulated_stability`). |
| **Healthy tuning capped by noise amplification** (`Kc·σ_meas ≤ 0.25%` OP) | Uncapped SIMC (τc = θ) gives Kc up to ~20 on lag-dominant and integrating loops; measurement noise then shakes the valve continuously and no plant would be tuned that way. |
| **Constant load bias of 1–4% at the process input** | Without it the loop starts exactly at its balance point and a sticky valve can sit there forever — an artefact of the start-up, not physics. |
| **`tuning_tight` = gain margin 1.03–1.12**, not 1.15–1.5 | Sweep: regular oscillation in 100% of loops at GM 1.03, 70–90% at 1.1, 0–40% at 1.2, none at 1.5. SACAC's tight files oscillate regularly, so they sit within ~10% of instability. A textbook "tight" loop at GM 1.3 does not look oscillatory at all. |

### 11.2 What each condition does

Fraction of runs with a regular oscillation (regularity > 1), and median Harris index [10th–90th percentile]:

| Condition | Oscillating F · P · L | Harris F | Harris P | Harris L |
|---|---|---|---|---|
| healthy | 0 · 0 · 0 | 0.60 [0.36–0.83] | 0.43 [0.36–0.67] | 0.45 [0.25–0.58] |
| stiction | 0.56 · 0.50 · 0.56 | 0.17 [0.09–0.42] | 0.25 [0.11–0.41] | 0.14 [0.08–0.25] |
| tuning_tight | 1.00 · 0.69 · 0.69 | 0.29 [0.18–0.45] | **0.62** [0.47–0.70] | **0.60** [0.50–0.75] |
| tuning_sluggish | 0 · 0.19 · 0.13 | 0.18 [0.09–0.54] | 0.11 [0.02–0.18] | 0.02 [0.00–0.19] |
| external_oscillation | 1.00 · 0.88 · 1.00 | 0.07 [0.01–0.30] | 0.04 [0.01–0.09] | 0.02 [0.01–0.04] |

> ⭐ **The Harris index rates tight-tuned pressure and level loops ABOVE healthy ones** (0.60–0.62
> against 0.43–0.45). Minimum-variance control is itself aggressive, so the benchmark rewards a loop
> for moving toward instability. **Stiction, sluggish tuning and external oscillation overlap in the
> 0.02–0.25 band.** Harris says *how much* could be gained; it cannot say *why*, and it cannot see
> fragility.

### 11.3 ⚠️ A sticky valve does not always oscillate

Only 50–56% of simulated stiction runs produce a regular oscillation. What decides it is the slip
jump `J` against the load-noise standard deviation:

| J / σ_load | F | P | L |
|---|---|---|---|
| < 1 | 0.00 (n=6) | 0.00 (n=7) | 0.00 (n=1) |
| 1–2 | 0.23 (n=13) | 0.36 (n=14) | 0.20 (n=5) |
| > 2 | 0.95 (n=21) | 0.68 (n=19) | 0.65 (n=34) |

Below the noise, the valve is dithered by the disturbance and the limit cycle never forms. **The
fault exists but is not visible in OP or PV — only in MV, which plants do not record.** This is the
same lesson as TEP, detectability ≠ discriminability: the fix is instrumentation, not an algorithm.

**Consequence for SACAC:** 12 of 13 labelled stiction files oscillate regularly. Published benchmark
cases were chosen *because* they show a clean limit cycle. Good results on SACAC say nothing about
stiction buried in noise.

### 11.4 The Harris index depends on the assumed delay

Healthy loops, Harris at a wrong delay divided by Harris at the true one (median):

| Delay error | F | P | L |
|---|---|---|---|
| d − 1 | 0.77 | 0.89 | 0.89 |
| d + 2 | 1.24 | 1.22 | 1.23 |
| d + 5 | 1.33 | 1.55 | 1.59 |

**Overestimating the dead time by five samples flatters a loop by up to 60%.** Every real-data
Harris value is reported as a range over the plausible delay, never as one number.

### 11.5 Harris on the only real loops with a published dead time

| File | Label | Samples used | d range | Harris | Reading |
|---|---|---|---|---|---|
| `other-F-paper-horch-2003-2` | healthy | 603 of 1198 (constant-SP segment) | 8–9 | 0.74–0.75 | consistent with a well-tuned flow loop |
| `other-F-paper-horch-2003` | external oscillation | 1196 | 4–9 | 0.14 | **flat across d** — an oscillation dominates any delay effect |
| `other-L-paper-horch-2003` | healthy | 904 | 3 | **0.013** | ⚠️ see below |

**A "normal" level loop scores 0.013.** Level loops are usually tuned for *averaging* — the vessel is
there to absorb flow changes, and letting the level move is the point. The minimum-variance
benchmark is the wrong objective for such a loop. **Harris must not be applied to averaging level
control**, and a report ranking loops by Harris would put a correctly-working surge drum at the top of
the worst list.

### 11.6 ⚠️ New data findings

**Three labelled loops share one oscillation.** `stiction-F-paper-horch-2003`,
`tuning-Q-paper-horch-2003` and `other-F-paper-horch-2003` all oscillate at 29.2–29.3 s, and are
coherent at that frequency (magnitude-squared coherence 0.72–0.97, against about 0.1 elsewhere in the
spectrum and 0.08 for an unrelated `horch` loop). Phase-locked over roughly 40 cycles, they are
almost certainly **one oscillation seen at three points** — carrying three different published labels:
stiction, tight tuning, external disturbance. At most one of them is the source.

> The external test set is therefore **11 independent events, not 13**. These three are reported
> together, and they become a small root-cause case for Week 4.

**Sampling time: the description and the data disagree.**
`stiction-L-paper-horch-2003` is described as 1 s but its time column steps by 2. The Thornhill files
number samples (step 1) while the description says 20 s. The loader now returns a sampling time only
when both sources agree and the steps are regular; otherwise `ts` is undefined, because a wrong
sampling time silently corrupts every delay-based index.

### 11.7 Limitations of the Week 1 simulator

1. **The controller runs five times faster than the historian.** The minimum-variance benchmark at the
   historian rate is therefore an approximation of what the real controller could achieve.
2. **Regulatory data only** — constant setpoint. The SACAC sluggish loop tracks a moving setpoint
   (probably a cascade slave); cascades are simulated in Week 4 (Section 13).
3. **One stiction model** (Choudhury et al., 2005). Other models (Kano, He) differ in how the valve
   behaves after it stops.
4. **The external oscillation is a sinusoid.** Oscillations propagated from an upstream sticky valve
   are filtered triangles, not sine waves.
5. **The regularity index takes the ACF to half the record.** A lightly damped loop has regular early
   zero crossings and noisy late ones, which lowers its score — part of why GM 1.2 rarely registers.

---

## 12. ISDB — International Stiction Data Base (added after Week 1)

**Files:** `isdb10.mat` + `ISDB_manual.pdf` (v1.0, "Limited Trial Version", M. Jelali), from
`https://sites.ualberta.ca/~bhuang/ISDB.zip`, linked as "Data used in the book" on the book page
`https://sites.ualberta.ca/~bhuang/Stiction-Book.htm` (retrieved September 2026). The older address
`ualberta.ca/~bhuang/stiction-book` no longer serves the file. The manual states the "Stiction Group" published the
database for researchers to test their methods, and asks users to cite the book, its homepage or the
original source. **Not committed** — same rule as SACAC.

### 12.1 Content

114 records: **108 industrial loops** + 6 synthetic test signals (excluded). Buildings 8 · chemicals 76 ·
pulp & paper 13 · power 5 · mining 1 · metals 5. Each loop carries comments, type (0 self-regulating,
1 integrating), sampling period, and `t · SP · PV · OP`; some add `Kc`, `Ti` or normalised error.
(The comparative chapter of the book used 93 data sets; the file holds 108.)

### 12.2 Labels — read from the published comments, nothing inferred

| Stated in the comment | Loops |
|---|:---:|
| stiction | 15 |
| stiction + tight tuning | 1 |
| stiction **(likely)** | 9 |
| no stiction | 8 |
| no oscillation | 3 |
| tuning issue · marginal stability · dead zone | 3 |
| external disturbance | 3 |
| disturbance **(likely)** | 8 |
| quantisation · faulty sensor · OP missing | 3 |
| intermittent oscillation, cause not stated | 2 |
| **no label** | **53** |

> ⚠️ **"No stiction" is not "healthy".** `chem4` reads *"a tuning issue, no stiction"*; `pap6` — labelled
> "no stiction" here — is SACAC's *tight tuning* file. The label answers one question only.

All 17 "likely" labels come from one contributor. Whether they enter the test set is a decision,
not a default (open question below). The 53 unlabelled loops include two synchronous plant-wide sets:
**30 loops from a South-East Asian refinery** (`chem40–69`, 60 s) and **5 from a refinery separation
unit** (`chem13–17`, 20 s) — Week 4 material, not classification tests.

### 12.3 ⚠️ Overlap with SACAC — 10 files are ISDB loops

Normalised cross-correlation of PV = **1.000 at zero lag**:

| SACAC file | ISDB loop | ISDB comment |
|---|---|---|
| `stiction-P-chemical-baccidicapaci-2018` | `chem10` | pressure; with stiction (B. Huang) |
| `stiction-P-oilgas-baccidicapaci-2018` | `chem25` | ⚠️ **possibility of marginal stability** (C. Scali) |
| `stiction-L-power-baccidicapaci-2018` | `pow4` | ⚠️ level control — **no label** |
| `stiction-L-paper-horch-2003` | `pap3` | level; with stiction (A. Horch) |
| `tuning-L-paper-horch-2003` | `pap6` | level; no stiction (A. Horch) |
| `sensor-F-oilgas-thornhill-2007` | `chem14` | flow; faulty steam sensor |
| `unknown-P-oilgas-thornhill-2007-1` / `-2` | `chem15` / `chem16` | pressure PC1 / PC2 |
| `unknown-L-oilgas-thornhill-2002` | `chem54` | SE Asian refinery, LC6 |
| `saturation-T-oilgas-thornhill-2002` | `chem41` | SE Asian refinery, TC2 |

**Two label conflicts.** The same data is *stiction* in SACAC and *possible marginal stability* in ISDB
(`chem25`) — both from the Scali group, years apart. And `pow4` carries no diagnosis in ISDB while SACAC
calls it stiction. **Neither label is treated as settled; both loops are reported separately.**

**Provenance differs too:** SACAC attributes two of these files to `baccidicapaci-2018`; ISDB credits
B. Huang and S. Choudhury & S. Shah. The original contributors are cited.

**Independent confirmation of §3:** `pap6` has exactly **846 samples**, value-identical to rows 0–845
of the SACAC tuning file. The ISDB copy ends precisely where the loader cuts.

**Checked and not overlapping:** the three coherent `horch` loops of §11.6 are *not* ISDB `pap2`/`pap4`,
despite identical length and sampling — the values differ.

### 12.4 What the external test set becomes

Stated labels only, duplicates removed:

| | SACAC | ISDB, new | Total |
|---|:---:|:---:|:---:|
| stiction | 7 | 13 (+1 with tight tuning) | **20–21** |
| not stiction — healthy / no stiction / no oscillation | 2 | 10 | 12 |
| tuning (tight, sluggish, marginal, dead zone) | 3 | 2 | 5 |
| external disturbance | 1 | 3 | 4 |
| **labelled loops** | **13** | **29** | **≈ 42** |

From 13 labelled loops to about 42, from **seven contributors** instead of three — which makes a
hold-out **by contributor** feasible for *stiction vs not stiction*, the test §6 could not run.
Two of the seven SACAC stiction loops carry the conflicts of §12.3.

---

## 13. Scope change — cascade control gets its own week

**Decision (September 2026):** cascade control is added as a dedicated week. Cascades are standard
in refineries — level or temperature masters writing the setpoint of a flow slave — and the most
common misdiagnosis in them is blaming the wrong loop. The project grows from six weeks to seven;
the extra week comes from the plan's buffer (weeks 47–52).

| Week | Work |
|---|---|
| 1 | PID simulation · fault injection · Harris index ✅ |
| 2 | Data-quality checks · oscillation detection · classification |
| 3 | External test on SACAC + ISDB |
| **4** | **Cascade: structure detection · simulation · source attribution** |
| 5 | Plant-wide propagation and root cause |
| 6 | Archive compression |
| 7 | Validation · API · deployment · episode |

Cascade comes **before** plant-wide propagation on purpose: in a cascade the connection between the
two loops is known, so it is the simplest propagation case to test before the unknown connections of
a whole plant.

### 13.1 The data supports it — eight real cascade pairs, found from the data itself

No file description mentions a cascade, except one ISDB building loop described as the inner loop of
a room-temperature cascade (`bas8`; its master is not recorded). The pairs were found instead from
the **structural signature of a cascade: the slave's setpoint is the master's controller output.**
Every loop whose SP moves was correlated against the OP of every other loop in the same
synchronously recorded set:

| Plant | Master (OP) → Slave (SP) | r |
|---|---|---:|
| Eastman chemical plant (SACAC plant-wide, 30 loops) | tag 1 → tag 2 | 1.000 |
| | tag 5 → tag 7 | 0.999 |
| | tag 17 → tag 14 | 1.000 |
| | tag 13 → tag 19 | 1.000 |
| | tag 25 → tag 23 | 1.000 |
| South-East Asian refinery (ISDB `chem40–69`) | TC3 (`chem42`) → FC8 (`chem62`) | 0.992 |
| | **steam-drum level LC2 (`chem50`) → FC10 (`chem63`)** | 1.000 |
| Refinery separation unit (ISDB `chem13–17`) | **TC1 (`chem17`) → FC1 (`chem14`)** | 0.999 |

**Eight pairs from three plants.** Two are worth singling out:

- **Steam-drum level → feedwater flow** is the textbook cascade.
- **FC1 (`chem14`) is labelled "faulty steam sensor" in ISDB and is the slave of a temperature
  cascade.** A fault in the slave also appears in the master — exactly the case where a report must
  not blame the master's tuning. It becomes the first real test of source attribution.

Four more loops have a moving setpoint with **no matching master** in their set (Eastman tag 8;
refinery tags 23, 26, 27): the master is not recorded, or the moves are operator changes. Undetermined.

> **Caveat:** r ≈ 1 between one loop's SP and another's OP is a structural signature, not a published
> fact. Ratio control produces the same signature. The pairs are treated as cascade *candidates* until
> the scaling (slave SP range against master OP range) is checked in Week 4.

### 13.2 What the cascade week will do

1. **Detect the structure** automatically, tested on the eight pairs — and checked for false pairs
   across every other combination.
2. **Simulate** a flow slave under a level or temperature master, with the fault injected in one
   loop at a time, so the true source is known.
3. **Attribute the source.** If both loops oscillate, look at the slave's error (its SP − PV):
   - slave error oscillates → the slave cannot follow its setpoint; the fault is in the slave,
     usually its valve → **maintenance**;
   - slave follows its setpoint, but the setpoint itself oscillates → the slave is innocent; the fault
     is in the master → **control engineer**.
4. **Validate on real data**, starting with `chem14` under `chem17`.

### 13.3 ⭐ Explicit limit — cascade performance assessment

> **Assessing the joint performance of both loops with a cascade-specific Harris index is advanced
> research that would take weeks. It is out of scope and is stated as an explicit limit in the README.**
> The minimum-variance benchmark for a cascade must account for how the two loops share the
> disturbance and the delay; the single-loop Harris index of §11 does not apply to a slave whose
> setpoint is driven by a master. In this project, slaves are **diagnosed** (Section 13.2), not
> **scored** with Harris.

---

## 14. Week 2, part 1 — data-quality checks before any diagnosis

Code: `chemai/features/data_quality.py` · thresholds in `DataQualityConfig` · ISDB reader
`chemai/data/isdb.py`. Evidence: `projects/p02_control_loops/week2_data_quality.py`, run on
**141 distinct real loops** (43 SACAC files + 98 ISDB loops not already in SACAC).
Teaching version: notebook `p02_week2_data_quality`.

### 14.1 Five traps, five rules

| Trap | How it deceives | Rule | Threshold |
|---|---|---|---|
| **Frozen sensor** | looks like the best loop — and the integral drives the process away | PV flat **while OP moves** | 300 samples |
| **Manual / inactive** | assessing a controller that is not acting | OP flat, or never moving | 200 samples |
| **Saturation** | looks like bad tuning | OP *staying* (≥ 3 samples) at its recorded limit | > 5 % of samples |
| **Quantisation** | looks like a sticky valve (staircase PV) | distinct PV values | < 20 |
| **Moving setpoint** | SP steps counted as oscillation | SP changes in most samples → flag; few changes → longest constant segment | > 50 % of samples |

Plus one **note, never an exclusion**: *PI law not confirmed* when R² of `ΔOP ~ Δe, e` is below 0.5.

Two guards that the real data forced:

- **Coarsely recorded OP** (< 50 distinct values) skips the OP-based checks. ISDB `chem9` is a
  stiction loop whose OP has 28 values and flat stretches of 339 samples — without the guard it would
  be flagged *manual* and *saturated*, both wrong.
- **A frozen sensor needs OP to move.** PV flat with OP flat is a stopped unit or manual, not a stuck
  transmitter.

### 14.2 How the thresholds were set — and one that changed

Thresholds sit **above what normal loops do**, measured on the real set, never on the simulation.
The simulation gives the extremes (a 1500 s freeze, R² = 0 in manual), not the boundary.

| Quantity | Median | 95 % | 99 % | Max in an expert-labelled normal loop |
|---|---:|---:|---:|---:|
| Longest PV flat run (samples) | 1 | 39 | 202 | **275** (`buildings.8`, "no oscillation") |
| Distinct PV values | 793 | — | — | quantised files: 8 · 9 · 18 |

**The PV flat-run threshold moved from 200 to 300.** 200 was chosen on 150 loops including
duplicates, where the 99th percentile was 185. On the 141 distinct loops it is 202, and a room
temperature loop labelled *no oscillation* reaches 275 — most likely an archive deadband, not a stuck
transmitter. 300 clears every expert-labelled normal loop; a real freeze lasts far longer.

**Limitation:** counts are in samples. 300 samples is 5 minutes at 1 s but 100 minutes at 20 s.

### 14.3 Results on the real set

| Flag | Loops | Comment |
|---|---:|---|
| `moving_setpoint` | **40** | 28 % of real loops — DB runs, cascade slaves. Harris cannot be applied as-is |
| `op_constant` | 9 | OP never moves: manual, or OP not truly recorded (five SE Asian refinery tags) |
| `saturated` | 5 | both SACAC saturation files, plus three unlabelled ISDB loops |
| `op_inactive` | 4 | overlaps saturation: OP pinned at its limit is also flat |
| `quantised_pv` | 3 | both SACAC quantisation files with PV, plus ISDB `chem3` — labelled *quantisation* in ISDB |
| `op_coarse` | 3 | ISDB `chem7`, `chem8`, `chem9` |
| `no_op` / `short_regulatory_segment` | 1 / 1 | |
| `frozen_sensor` | 0 | (1 at the old threshold of 200 — see 14.2) |

**Every illustrative file is caught:** both quantisation files with a PV (`quantised_pv`) and both
saturation files (`saturated`). ISDB `chem3`, labelled *quantisation* by its contributor, is caught
without having been used to set anything.

**One miss, stated:** `quantisation-F-minerals-bauer-2017` has 271 distinct PV values — the
normalisation removed the quantisation fingerprint. It is flagged instead as `op_constant`: its OP
never moves in the whole record.

### 14.4 ⚠️ A third description conflict

`unknown-F-paper-horch-2003` is described as *"The loop was in manual control"*. The data says
otherwise: OP follows a PI law on the error with **R² = 1.00** (Kc ≈ 0.12). Its SP also moves in
almost every sample — the signature of a cascade slave. Possibly the *master* was in manual, possibly
the description is wrong. The loop is not treated as manual.

### 14.5 PI-law fit is a note, not a detector

R² across the real set: 10th percentile 0.015, 25th 0.53, median 0.96. The low values have causes
that are **not** manual: OP never recorded (constant), OP archived coarsely, controllers that are not
a plain PI (thickness control in the metals set). Excluding every loop below 0.5 would discard a
quarter of the real data for the wrong reason.

### 14.6 ISDB reader

`load_isdb` reproduces the label counts of §12.2 exactly (15 stated stiction, 1 stiction + tight
tuning, 9 likely stiction, 8 likely disturbance, …). Building it caught one bug worth recording:
ISDB `buildings.7` reads *"after **detuning** the controller"*, and a substring match read it as a
tuning diagnosis. The label rule now matches whole words.

---

## Open questions

- Does the shape test survive **detrending and cycle isolation** on all thirteen stiction files,
  or only on the fast loops?
- Is `stiction-L-power` (index 0.51) a genuine ambiguity, or an artefact of the level dynamics?
- Do the 17 "likely" ISDB labels (one contributor) enter the test set, as a separate weaker tier,
  or stay out?
- `stiction-P-oilgas` is stiction in SACAC and possible marginal stability in ISDB: which stands?
- Which of the three coherent `horch` loops (§11.6) is the source — the sticky flow valve, the
  tight quality loop, or neither?
- Should the week-2 classifier's positive class be *stiction* or *stiction that produces a limit
  cycle* — given that below J/σ ≈ 1 the fault is invisible in OP and PV (§11.3)?
- Should shape analysis run on every loop, or only on those that pass oscillation detection —
  given that `DB-6` fails the gate?
- Do the twelve sluggish files belong in the study at all, or are they a separate problem —
  **loop not oscillating but not controlling either**?

---

*Sections 1–10 recorded before modelling. Sections 11–13 recorded after Week 1, section 14 in Week 2. Every constraint came from
reading the data or the physics, not from a metric.*
