# Exploratory Data Analysis

**Project 1 — Distillation Soft Sensor**
Predicting laboratory vapour pressure from process measurements

> Restored from the original project record, October 2026. Two corrections: the test campaign ran at
> **less than half** the training campaign's average reboiler duty (about a third of the level at the
> start of the record), and the separation labels in §10 are now explained.

---

## Purpose of this document

This records **what was found while exploring the data, and what was decided as a result** —
before any modelling was finalised. It exists because three of the findings below would have
silently corrupted the results had they gone unnoticed, and two hypotheses that looked
physically sound turned out to be wrong.

**Two of the entries are hypotheses that were rejected, and one is a methodological error made
and caught during this analysis.** They are recorded rather than removed, because they are the
part of the work that is not reproducible from the final numbers alone.

---

## 1. Dataset

**Source:** OpenMV.net — Distillation Tower.
An industrial distillation column; variable names are coded, so the physical location of each
sensor on the column is not disclosed.

| | |
|---|---|
| Rows × columns | **253 × 28** |
| Date range | 2000-08-21 → 2003-02-04 |
| Missing values | **0** |
| Sampling | median **2 days** · mean 3.6 · **maximum gap 82 days** |

**Target:** `VapourPressure` — a laboratory quality measurement. The other values in each row
are the process conditions at the moment the sample was taken.

| | |
|---|---:|
| mean | 36.65 |
| median | 34.85 |
| std | 6.84 |
| min / max | 25.16 / 64.31 |

The target is **right-skewed** — mean above median, maximum roughly four standard deviations
above the mean.

> **Note on scope:** the target is vapour pressure, **not composition**. Estimating composition
> directly would require different data or a simulation reference.

---

## 2. Finding — five columns are exact derivations, not measurements

Correlation between each `Inv*` column and its candidate source:

| Column | Equals | Correlation |
|---|---|:---:|
| `InvTemp1` | 1000 / `Temp12` | **1.0000** |
| `InvTemp2` | 1000 / `Temp2` | **1.0000** |
| `InvTemp3` | 1000 / `Temp9` | **1.0000** |
| `InvPressure1` | 1000 / `PressureC1` | **1.0000** |
| `Temp11` | `Temp3` − `Temp9` | max difference **0.000** |

**Why this matters:** with both forms present, a feature-importance analysis will attribute the
signal to whichever version the model happens to select. An attribution naming `InvTemp3` as the
top driver is naming `Temp9` — and any physical interpretation built on that is describing a
variable that does not independently exist.

**Decision:** keep **one form per underlying measurement**. Which form is not arbitrary — see §7.

**On `Temp11`:** per the source documentation this is *the temperature rise of the stream leaving
the column and returning after passing through the reboiler.* It is therefore **reboiler
temperature rise**, not a column top-to-bottom difference. It is retained as a physically
meaningful variable rather than dropped as redundant.

---

## 3. Finding — three distinct operating periods

Two gaps in the record split the data into three campaigns:

| Period | Rows | Dates | VP mean | VP std |
|:---:|:---:|---|---:|---:|
| **0** | 163 | 2000-08-21 → 2002-01-26 | 35.62 | 6.40 |
| **1** | 23 | 2002-03-12 → 2002-05-21 | 35.82 | **3.38** |
| **2** | 67 | 2002-08-11 → 2003-02-04 | **39.44** | **7.97** |

Process variable means:

| Variable | P0 | P1 | P2 | Change |
|---|---:|---:|---:|---:|
| `FlowC1` | 348.81 | 266.05 | 226.43 | **−35%** |
| `Temp11` (reboiler rise) | 23.96 | 19.78 | 17.25 | **−28%** |
| `PressureC1` | 242.65 | 241.68 | 227.66 | −6% |
| `Temp3` | 488.75 | 482.49 | 467.19 | −4% |
| `Temp9` | 464.79 | 462.71 | 449.94 | −3% |
| `Temp2` | 374.95 | 372.88 | 365.56 | −3% |

**Reboiler duty proxy** = `FlowC1` × `Temp11`:

| | P0 | P1 | P2 |
|---|---:|---:|---:|
| mean | 8608 | 5410 | **3929** |
| std | 3322 | 2076 | **833** |

**Duty fell by roughly half** — more than flow alone, since flow and temperature rise both
declined. Flow variability also fell (std 73.0 → 30.7).

**Physical reading:** in period 2 the unit runs at less than half of its period-0 reboiler duty
(about a third of the level at the start of the record), and runs it far more steadily. Less stripping duty is consistent with a higher vapour pressure in
the product.

**But note the inversion:** flow became **more stable** while product quality became **more
variable** (VP std 6.40 → 7.97). This is consistent with operating the column away from its
design load, where quality control degrades.

---

## 4. ❌ Rejected hypothesis — progressive fouling

**Hypothesis:** the declining reboiler duty indicates progressive fouling.

**Rejected on two grounds:**

1. **The duty recovers.** Plotted over time, duty falls through early 2001, then **returns to its
   original level** by mid-2001 before falling again. Fouling does not reverse spontaneously.
2. **The correlation is weak.** Overall `corr(duty, VP) = −0.165`; **within period 0 it is
   −0.062.** The relationship appears between campaigns, not within them.

**Also noted:** the flat segment between June and September 2002 in the time plot is **matplotlib
interpolating across the data gap**, not a period of steady operation. It should not be read as data.

**Revised reading:** the load reduction is an **operational decision**, not equipment degradation.

---

## 5. ❌ Rejected hypothesis — the model will not generalise across the regime change

**Hypothesis:** a model trained on period 0 will fail on period 2, since reboiler duty differs by a
factor of about two.

**Test:** Ridge on 7 features, trained on period 0, evaluated on period 2.

| | R² |
|---|---:|
| In-regime (period 0) | 0.936 |
| **Out-of-regime (period 2)** | **0.957** · MAE 1.28 |

**The model did not degrade — it improved slightly.**

**Why:** the relationship being learned is **thermodynamic, not operational**. Temperature
determines vapour pressure through an Antoine-type relation regardless of throughput. A model
that had learned operational correlations would not survive this test.

> This is a positive result, and a testable distinction: **models built on physical relationships
> generalise across operating regimes; models built on correlations do not.**

---

## 6. Finding — two variables match seven

| Features | In-regime R² | Out-of-regime R² |
|:---:|---:|---:|
| 1 (`Temp9`) | 0.784 | 0.899 |
| **2** (`Temp9`, `PressureC1`) | **0.932** | **0.952** |
| 7 | 0.936 | 0.957 |

The additional five features contribute **0.004** — below measurement noise.

Best-performing pairs, evaluated out-of-regime:

| Pair | R² |
|---|---:|
| **`Temp9` + `PressureC1`** | **0.952** |
| `Temp2` + `Temp9` | 0.869 |
| `Temp12` + `PressureC1` | 0.844 |
| `Temp3` + `PressureC1` | 0.826 |

**The winning pair is one temperature and the operating pressure** — which is the structure of a
vapour pressure relation, not an arbitrary statistical selection.

**Industrial implication:** two sensors instead of seven means less calibration, less maintenance
and fewer failure points. That is an engineering decision, not a model result.

---

## 7. ⚠️ Methodological error — made and caught here

**First attempt:** replace `Temp9` with the Antoine form `1000/Temp9`, keeping Ridge as before.

**Result: R² fell from 0.95 to 0.524.**

The apparent conclusion — *"the physical form performs worse"* — is **wrong**.

**Cause — feature scaling:**

| Variable | Range | Std |
|---|---|---:|
| `Temp9` | 400.3 – 507.3 | 17.61 |
| `1000/Temp9` | 1.9712 – 2.4980 | **0.0846** |
| `PressureC1` | 199.8 – 254.6 | 13.92 |

The pressure variable has **165× the standard deviation** of the inverse-temperature variable.
Ridge penalises coefficients uniformly, so a low-variance feature requires a large coefficient to
contribute — and the penalty suppresses it.

**Fix:** `StandardScaler` before the regressor.

> **The physics did not fail. The evaluation did.** One missing line would have produced a
> published conclusion that was the exact opposite of the truth.

---

## 8. Finding — the Antoine form outperforms raw temperature

With scaling applied consistently, out-of-regime R²:

| Feature form | R² |
|---|---:|
| Raw temperature + pressure | 0.950 |
| ⭐ **Antoine — `1000/T` + pressure** | **0.980** |
| Rankine — `1000/T` + `ln T` + pressure | 0.977 |

**Antoine's equation:**

$$\log_{10} P^{sat} = A - \frac{B}{C + T}$$

Vapour pressure depends on the **inverse** of temperature, not temperature itself. Expressing the
feature in that form cuts the error by about a third — MAE 0.953 → 0.645, RMSE 1.764 → 1.118 — with no
additional variable.

**Rankine's logarithmic term does not help** (0.977 vs 0.980). Antoine is the appropriate form for
this system; the extra complexity is not justified. That is itself a result.

**Retrospective note on §2:** the dataset's `InvTemp` columns are not arbitrary transformations —
they are the form the physics requires. Whoever prepared this data knew that. Had they been
dropped as duplicates, this finding would have been lost.

---

## 9. Decisions carried into modelling

| Decision | Basis |
|---|---|
| Keep the **inverse** temperature form; drop the raw duplicates | §7, §8 |
| Retain `Temp11` as reboiler temperature rise | §2 |
| **Split by operating period**, never randomly | §3 |
| Train on period 0, evaluate on period 2 as the generalisation test | §5 |
| **Scale features before any penalised regression** | §7 |
| Baseline is `1000/Temp9` + `PressureC1` — two features | §6, §8 |
| Report out-of-regime performance as the headline number | §5 |

---

## 10. Reference column study — and a confounder that had to be removed

A seven-component reference column (25 stages, Wilson K-values, Wang–Henke solution) was built to
answer one question: **which stage temperature carries the most information about the bottom
product's vapour pressure?** That would indicate what `Temp9` most likely is.

Four sweeps of 50–60 cases each were run.

| Sweep | Pressure | Separation | VP range (kPa) | Best stage | \|corr\| | Top five |
|---|---|---|:---:|:---:|:---:|---|
| **A** | **varied** 6.5–8.5 bar | mild | 52–93 | 22 | 0.895 | 22·23·21·20·19 |
| **B** | **varied** 14–18 bar | severe | 72–118 | 24 | **0.719** | 24·23·**1·0·2** ⚠️ |
| **C** | **fixed** 7 bar | severe | **43–67** | **24** | **0.996** | 24·23·22·21·20 |
| **D** | **fixed** 7 bar | mild | 52–97 | **24** | **0.997** | 24·23·22·21·20 |

*Separation* is how sharply the column separates light from heavy components. A severe (sharp)
separation leaves fewer light components in the bottoms, so their vapour pressure is lower; a mild
(gentle) one leaves more. The R² of the best stage alone is the square of |corr|: 0.99, 0.80 and 0.52.

### ⚠️ The confounder

Sweep B was run to test whether the sweep-A result held under harder separation. It did not — and
the reason turned out to matter more than the original result.

**Look at sweep B's top five: stages 1, 0 and 2 appear.** Those are at the *top* of the column, in
an analysis of a *bottom* product. That is not physically meaningful.

**Cause:** pressure was varying across the sweep. Pressure raises every stage temperature *and*
raises the product's vapour pressure. Every stage therefore correlates with vapour pressure
**through pressure**, not through composition. The stage-specific signal is swamped.

The evidence is in the correlation strength: it *fell* from 0.895 to 0.719 when the sweep was made
"harder". A cleaner experiment should not produce a dirtier correlation.

**Fix:** hold pressure fixed and vary only feed composition, reflux ratio and distillate rate.

### Result with pressure fixed

Sweeps C and D agree closely, across a threefold difference in separation severity:

- **Best stage is 24 (the reboiler) in both**, correlation ≈ 0.996
- **The ranking is monotonic from the bottom upward** — 24, 23, 22, 21, 20 — with no
  physically implausible entries
- The sweep-A finding of stage 22 was **an artefact of varying pressure**, not a physical result

### ⚠️ But do not over-read the reboiler result either

The correlation of 0.996 is **partly definitional, not informational.**

Reboiler temperature is the bubble-point temperature of the bottom composition at column pressure.
Vapour pressure is the bubble-point pressure of the same composition at 37.8 °C. **Both are direct
functions of the same composition** — so a near-perfect correlation between them is close to a
tautology, not a discovery.

**And it is a model artefact as much as a physical one.** With seven components the bottom stream
is not nearly as pure as a real column's, where a continuous heavy tail dominates the reboiler
temperature and makes it insensitive to the light ends that set vapour pressure. That
insensitivity is precisely why real plants control composition on a **selected sensitive tray**
rather than on reboiler temperature.

### What can be concluded

> **With pressure controlled — as it is in a real plant — information about bottom product quality
> increases monotonically toward the bottom of the column. `Temp9` is therefore most likely a
> lower-section temperature.**
>
> **The exact stage cannot be identified.** The model's near-perfect reboiler correlation is partly
> definitional and partly an artefact of using seven discrete components.

This is a weaker conclusion than the one first reached, and it is the correct one.

### Comparison with the industrial data

| | Sweep C | Industrial |
|---|:---:|:---:|
| Vapour pressure | 43–67 kPa | 25–64 kPa |

**Sweep C is the closest operating regime to the real unit** — fixed pressure, severe separation.
The overlapping range is a consistency check, not an identification: feed, configuration and units
all differ, and the source does not state units for `Temp9` or `PressureC1`.

### Still open

- **Would a model with a continuous heavy fraction** (pseudo-components rather than seven discrete
  ones) reproduce the sensitive-tray behaviour seen in real columns? This is the most direct test
  of whether the reboiler result above is an artefact.
- **Was the load reduction a commercial decision or a capacity constraint?** The data cannot
  distinguish these.
- **What happened during the 82-day gap?** A turnaround or cleaning is likely but not documented.
- **Period 1 is only 23 points** with unusually low variance. Too small to conclude from.

---

## 11. Limitations

1. **253 rows.** Small-data regime — regularised linear models and light gradient boosting only.
   No deep learning.
2. **Coded variables.** Sensor positions are not disclosed. §10 narrows `Temp9` to a probable
   sensitive-tray location by analogy with a reference column, but this remains an inference.
3. **Snapshot sampling**, not a continuous time series. Samples were taken when laboratory
   measurements were made.
4. **No labelled faults.** This dataset supports soft sensing, not fault detection.
5. **Period 2 has 67 rows.** The generalisation result is encouraging but rests on a modest
   test set.
