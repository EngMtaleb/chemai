# Project 1 — Distillation Soft Sensor

## The problem

The bottoms product of a distillation column carries a vapour pressure specification — an upper
limit. That property is measured in a laboratory, and the result takes 4 to 8 hours.

The operator cannot set vapour pressure. He sets a temperature on a selected tray, and the
controller holds it by moving steam to the reboiler:

```
setpoint → controller → steam valve → reboiler → stripping → vapour pressure
                                                                ↑ lab only
```

So for hours at a time he adjusts the one thing he can see, to control the one thing he cannot. He
responds by running well below the limit — stripping harder than necessary.

| | |
|---|---|
| **Below the limit** | over-stripping · more steam, and sellable light material sent overhead |
| **Above the limit** | under-stripping · off-spec — reprocess, downgrade or reject |

Both are losses. Only one gets audited — so he stays far below, and pays for it in energy and yield
every hour.

## Results

| Metric | Value |
|---|---:|
| **R² out-of-regime** | **0.980** |
| MAE out-of-regime | 0.645 |
| MAE inside the envelope | 0.495 |
| Interval coverage (95% nominal) | 94% |
| Error ratio outside the envelope | 7.8× |
| Inputs | **2** — `1000/Temp9`, `PressureC1` |

Trained on one operating campaign, evaluated on another with **one third the reboiler duty**. The
relationship learned is thermodynamic, not operational — which is why it survives the regime change.

---

## ⭐ What the project found

### The physics changed the input, not the model

Antoine's equation makes vapour pressure a function of the inverse of temperature:

```
log₁₀ Psat = A − B / (C + T)        the physics
feature     =     1 / T             what the model receives
```

Same model, same data, same split. Only the shape of the input changed:

| feature form | R² | MAE |
|---|---:|---:|
| raw temperature + pressure | 0.950 | 0.953 |
| **1/T + pressure** | **0.980** | **0.645** |

A 32% reduction in error, with no additional variable. A more complex form (Rankine, with an added
log term) did not help — Antoine is the right shape for this system, and that is a result in itself.

### Two inputs beat twenty-one

The dataset offers 21 real measurements after removing derived columns. Every one was tested:

| inputs | R² | MAE |
|---|---:|---:|
| all 21 raw columns | 0.884 | 2.520 |
| 7 hand-picked | 0.960 | 1.162 |
| 2 raw | 0.950 | 0.953 |
| **2 · Antoine form** | **0.980** | **0.645** |

Using every column makes it worse. With 163 training rows, 21 inputs is enough freedom to fit the
training campaign rather than the physics.

`Temp9` is the strongest single input among all 21, and `Temp9 + PressureC1` the strongest of all 210
pairs — neither was chosen by intuition. And a detail worth noting: `PressureC1` alone scores
**−0.497**, worse than predicting the mean, yet paired with `Temp9` reaches 0.950. Pressure means
nothing without temperature — exactly what a vapour-pressure relation predicts.

### The system knows when to refuse

Every response carries three fields, not one:

```json
{"vapour_pressure": 38.8, "interval_95": [35.84, 41.75], "in_envelope": true}
```

**Prediction interval** — from 500 bootstrap refits. Claimed 95%, measured 94% on the held-out
campaign.

**Operating envelope** — Mahalanobis distance, so it flags implausible *combinations*, not only
individually extreme values. A temperature of 460 is normal and a pressure of 235 is normal; their
combination may never have occurred.

And it was tested: error on flagged points is **7.8× worse** than on accepted ones. Most projects add
an envelope check and never verify it identifies the right samples.

### Sensor positions are not disclosed — so a reference column was built

The variable names are coded: `Temp9` is a temperature, but its position on the column is not given.
A 25-stage column was written from scratch — seven real hydrocarbons, Wilson K-values, Wang–Henke
solution — and swept over 50 operating cases with pressure held fixed.

Quality information increases monotonically toward the bottom, so `Temp9` is most likely a
lower-section temperature. The exact stage cannot be identified — the near-perfect reboiler
correlation is partly definitional, and with seven discrete components the bottom stream is purer
than a real column's.

📄 `docs/EDA.md` — what was found in the data
📄 `docs/VALIDATION.md` — where the model fails, and why

---

## Run the API

```bash
python -m chemai.api.run
```

Interactive documentation at `http://localhost:8000/docs`.

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 450, "pressure": 228}'
```

```json
{
  "vapour_pressure": 38.8,
  "interval_lower": 35.84,
  "interval_upper": 41.75,
  "in_envelope": true,
  "distance": 1.26,
  "warning": null
}
```

Outside the validated range the service still answers, but says so:

```json
{
  "vapour_pressure": 9.73,
  "in_envelope": false,
  "warning": "input outside validated operating envelope - estimate unreliable"
}
```

**There is deliberately no endpoint returning a bare number.** A caller who sees only a value
will trust it unconditionally.

---

## Design decisions

**Two features, not seven.** The extra five contribute under 0.005. Fewer sensors means less
calibration, less maintenance, fewer failure points.

**The inverse-temperature form.** Antoine's equation makes vapour pressure a function of `1/T`.
Using that form instead of raw temperature reduces residual error by ~40% with no new variable.

**Scaling is not optional.** `1000/T` has 165× less variance than pressure; without a scaler,
Ridge suppresses it and R² falls from 0.98 to 0.52.

**Split by operating campaign, never randomly.** A random split would place samples from the
same campaign on both sides and report a score that does not reflect a new regime.

**The envelope detector was tested, not assumed.** Error on flagged points is 7.8× worse than
on accepted ones — it identifies exactly where the model fails.

---

## Known limitations

1. **Under-predicts high values** — 4.8× worse error on the top decile, with systematic bias.
   Not suitable as an off-specification alarm.
2. **253 rows total.** Small-data regime; regularised linear models only.
3. **Coded variables.** Sensor positions are not disclosed; a reference column model narrows
   `Temp9` to the lower section, but the exact stage is unidentified.
4. **Two operating campaigns.** Generalisation shown across one regime change, not many.
5. **The exhaustive pair search over 210 combinations is itself selection on the test set.** The
   ranking is consistent with the physics; 0.950 is not an unbiased estimate.

Full detail in `docs/VALIDATION.md`.

---

## Data

Industrial distillation tower dataset from [OpenMV.net](https://openmv.net/info/distillation-tower).
**Not committed** — place `distillation-tower.csv` in `data/`.

---

**What this is:** a deployed prototype — a complete cycle from raw data to a service anyone can call.

**What it is not:** a plant system. No historian connection, no authentication, no monitoring, no
automatic retraining, and no field validation.
