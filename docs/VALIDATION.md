# Engineering Validation

**Project 1 — Distillation Soft Sensor**
Estimating laboratory vapour pressure from process measurements

> Restored from the original project record, October 2026. Updated: the duty ratio between the two
> campaigns, the reference-column status, and how the deployed service's interval relates to the
> full bootstrap interval (§6).

---

## 1. Results

| Metric | Value |
|---|---:|
| Model | StandardScaler + Ridge · 2 features |
| Features | `1000/Temp9` (Antoine form) · `PressureC1` |
| **R² out-of-regime** (period 2) | **0.980** |
| **MAE out-of-regime** | **0.645** |
| MAE inside validated envelope | **0.49** |
| Interval coverage (95% target) | **95.5%** full bootstrap · 94% deployed service (§6) |
| Residual σ | 1.51 |

**Split:** trained on period 0 (163 rows, 2000-08 → 2002-01), tested on period 2
(67 rows, 2002-08 → 2003-02) — **separate operating campaigns**; the test campaign ran at less than
half the training campaign's average reboiler duty.

> High R² values on this kind of data usually come from random row splits. The number above is a
> genuine out-of-regime result: the test set is a different operating campaign, not shuffled rows
> from the same one.

---

## 2. Physical consistency ✅

| Feature | Coefficient | Standardised contribution |
|---|---:|---:|
| `1000/Temp9` | **+81.82** | **6.92** |
| `PressureC1` | +0.164 | 2.28 |

**The inverse-temperature coefficient is positive, which is correct.** Lower temperature →
larger `1/T` → less stripping → higher vapour pressure in the product.

**Temperature contributes roughly three times more than pressure** — the expected ordering for a
vapour-pressure relation.

Had this coefficient come out negative, the model would have learned a relationship inverted
against the physics, and nothing downstream would have meant anything. It is checked first for
that reason.

---

## 3. Generalisation across operating regimes ✅

| Period | n | VP std | R² | **MAE** |
|:---:|:---:|---:|---:|---:|
| 0 (train) | 163 | 6.40 | 0.943 | 0.95 |
| 1 (unseen) | 23 | 3.38 | 0.835 | **0.87** |
| 2 (test) | 67 | 7.97 | 0.980 | **0.64** |

Three separate operating campaigns, with reboiler duty varying by a factor of about two. **Error is stable or
better outside the training regime.**

**Why it holds:** the relationship being learned is thermodynamic, not operational. Temperature
determines vapour pressure regardless of throughput. A model built on operational correlations
would not survive this test.

### ⚠️ A methodological correction

Period 1 has the **lowest R² (0.835) but a lower absolute error than the training period.**

R² is measured relative to the variance of the target. Period 1's target variance is roughly half
the others, so the same absolute error produces a mechanically lower R².

> **Comparing R² across sets with different target variance is misleading. Report MAE alongside.**

This was caught during validation. It is recorded rather than quietly fixed, because the same
error would recur in any project comparing performance across operating regimes.

---

## 4. ⚠️ The principal limitation — the model under-predicts high values

| Subset | n | MAE |
|---|:---:|---:|
| Top 10% of measured values | 7 | **2.19** |
| Remainder | 60 | **0.46** |

**Error is 4.8× worse on the highest values.**

And the bias is systematic, not random:

| | |
|---|---:|
| Mean residual | **+0.35** |
| corr(prediction, residual) | **+0.34** |

**The model consistently under-estimates high vapour pressure.**

### Why this matters more than the headline number

High vapour pressure is the **off-specification condition**. It is the case the system exists to
catch. The model is weakest exactly where it is most needed, and it errs in the **worse direction**
for an alarm system — reporting a value lower than reality.

### Physical explanation

Linear regression regresses toward the mean. High vapour pressure generally arises from strongly
non-linear conditions — severely reduced stripping — which a linear relation cannot follow. This
is a structural limitation of the model form, not a tuning problem.

### What follows from it

- **Do not use this system as an off-specification alarm** without accounting for the bias.
- A quantile or non-linear model for the upper range is a defensible next step.
- The under-prediction bias must be stated wherever the number is reported.

---

## 5. Out-of-range detection — and evidence that it works

Detection uses **Mahalanobis distance** in the two-feature space, so it flags implausible
*combinations* of temperature and pressure, not only individual values outside their ranges.

Threshold: 99th percentile of χ² with 2 degrees of freedom → **3.03**.

| | |
|---|---:|
| Flagged in training | 0.6% *(expected ~1%)* |
| Flagged in test | 4.5% *(3 of 67)* |

**Validation — the flagged points are exactly where the model fails:**

| | n | MAE |
|---|:---:|---:|
| Inside envelope | 64 | **0.49** |
| **Outside envelope** | 3 | **3.85** |

**Error is 7.8× worse on flagged points.**

> A safety mechanism that flags the wrong points is worse than none. This one was tested, not
> assumed. Most projects add an out-of-range check and never verify that it identifies the right
> samples.

---

## 6. Uncertainty — calibrated

Bootstrap resampling (B = 500) separates two sources of error:

| Source | Value |
|---|---:|
| Model uncertainty (spread across resampled fits) | 0.29 |
| Residual noise (irreducible scatter) | **1.51** |
| Mean 95% interval width | 6.05 |
| **Empirical coverage** | **95.5%** |

**The deployed service** uses the residual term only (±1.96 × 1.51): 63 of 67 lab values inside,
**94%**. The model term is small (0.29 against 1.51), so the two intervals differ by one test point.

**Coverage matches the stated 95%** — the intervals are calibrated. An interval claiming 95% while
covering 70% is worse than no interval, because it invites false confidence.

**Model uncertainty is 19% of residual noise.** The fit is well determined; the error is dominated
by process and measurement variability. **More data would not help much** — a different model form
for the upper range would.

---

## 7. What was found during development

Four issues that would have corrupted the result had they gone unnoticed. Two were hypotheses that
looked physically sound and turned out to be wrong; two were methodological errors made during the
work.

### ❌ Rejected — progressive fouling

Declining reboiler duty across the three campaigns suggested fouling.

**Rejected:** duty **recovers** to its original level mid-2001 before falling again — fouling does
not reverse. And `corr(duty, VP)` within period 0 is only −0.062; the relationship appears between
campaigns, not within them.

**Revised reading:** the load reduction is an operational decision, not equipment degradation.

### ❌ Rejected — the model will not survive the regime change

Reboiler duty differs about twofold between train and test, so failure seemed likely.

**Rejected:** out-of-regime performance is equal or better (§3). Physical relationships generalise
across operating regimes; correlational ones do not.

### ⚠️ Error — Ridge without feature scaling

The Antoine form initially scored **0.524**, apparently far worse than raw temperature.

**Cause:** `1000/T` has σ = 0.085 against 13.9 for pressure — a factor of **165**. Ridge penalises
coefficients uniformly, suppressing the low-variance feature.

**After scaling: 0.980.**

> **The physics did not fail. The evaluation did.** One missing line would have produced a
> published conclusion that was the exact opposite of the truth.

### ⚠️ Error — comparing R² across different target variances

See §3.

### Also found — five columns are not measurements

`InvTemp1 = 1000/Temp12`, `InvTemp2 = 1000/Temp2`, `InvTemp3 = 1000/Temp9`,
`InvPressure1 = 1000/PressureC1` (all r = 1.0000), and `Temp11 = Temp3 − Temp9` (exact).

With both forms present, feature attribution names whichever the model selected — and any physical
interpretation built on `InvTemp3` is describing `Temp9`.

**Retrospectively, the inverse columns are not arbitrary: they are the form Antoine's equation
requires.** Dropping them as duplicates would have lost the central finding of this project.

---

## 8. Limitations

1. ⭐ **Under-predicts high values** — 4.8× worse error, systematic bias (§4). The principal limitation.
2. **253 rows total**, 163 for training. Small-data regime; regularised linear models only.
3. **Coded variables** — sensor positions on the column are not disclosed, which limits physical
   interpretation. A 25-stage reference column ([EDA §10](EDA.md)) narrows `Temp9` to the lower
   section; the exact stage cannot be identified.
4. **Snapshot sampling**, not a continuous series. Samples taken when laboratory measurements were made.
5. **Two operating campaigns only.** Generalisation is demonstrated across one regime change, not many.
6. **No labelled faults.** This dataset supports soft sensing, not fault detection.
7. **Period 1 is 23 points.** Reported for completeness; too small to conclude from.

---

## 9. What this system is, and is not

**It is** a soft sensor that estimates laboratory vapour pressure from two process measurements,
reports a calibrated confidence interval, and **refuses when asked outside its validated envelope**.

**It is not** a replacement for the laboratory measurement, and **not** an off-specification alarm —
it under-estimates exactly the high values such an alarm would need to catch.

**Correct use:** an interpolating estimate between laboratory samples, during normal operation,
with its interval and envelope flag always carried alongside the number.

**And it produces one engineering conclusion beyond the model:** two sensors match seven. Fewer
sensors means less calibration, less maintenance and fewer failure points.

---

## 10. Next steps

| Priority | Action | Rationale |
|:---:|---|---|
| 1 | **Quantile or non-linear model for the upper range** | §4 — the limitation that matters |
| 2 | **Reference column with a continuous heavy fraction** | EDA §10 — tests whether the reboiler result is an artefact |
| 3 | Re-check bias after any model change | The under-prediction must not silently persist |
| 4 | Evaluate on a third campaign if data becomes available | §8.5 |
