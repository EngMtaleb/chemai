# Project 3 — RO Fouling and Cleaning

**Which RO train do we clean, in which week, and why?**

A seawater reverse-osmosis plant cleans a train (a CIP) when its feed-channel pressure drop has
risen enough. Too late, and fouling hardens and the next cleaning recovers less. Too early, and the
plant spends chemicals, downtime and membrane life for nothing. What the plant needs on Monday is
not a curve but a short table: **one row per train — an action, a week, the evidence, a confidence
and an owner** — or a plain statement that the data cannot support an answer.

This project builds that table from the pressure drop a plant already logs: a physics-based
normalisation, a forecast of when each train reaches its review level with a P10–P90 range and a
flag when it is looking too far ahead, and a weekly plan with the rules written down.

**Live:** [chemai-tr2c.onrender.com/ro/](https://chemai-tr2c.onrender.com/ro/) — worked examples
whose cause is known, or upload your own `day,dp_norm` CSV.

---

## ⭐ The headline: on the real plant, the forecast did not win

Tested once, after a written declaration, on the only public plant record of this kind we could
find (three trains, 18 cleanings — Su, Meng & Chang 2026): 213 weekly decisions with a known
answer, from 11 cleaning cycles, of which 61 were followed by a crossing within 28 days.

| "Will this train reach the review level within 28 days?" | Caught (of 61) | False alarms | Recall | Precision |
|---|---:|---:|---:|---:|
| the frozen forecast (P50 date within 28 days) | 13 | 5 | 0.21 | 0.72 |
| **a simple rule: the last 7 days sit within 0.03 of the level** | **26** | **2** | **0.43** | **0.93** |
| both, as the weekly plan now uses them (chosen after this test) | 31 | 6 | 0.51 | 0.84 |

The record has gaps: over the two years, a third of the weekly rows (32 %) had too few recorded
days to forecast at all, and the plan says so (`data`) instead of guessing.

Level forecasts 14 days ahead were **no better than assuming the level stays where it is** (mean
error 0.034 for both). The declaration had said this could happen, and named one reason: the
deposit's pressure drop carries a vendor normalisation. A second reason was found afterwards: its
cycles are short and quiet, so a train close to the level crosses on its day-to-day swings before
any trend matters.

The margin rule is now part of the plan. That change was made after seeing the real result, so the
last row is optimistic. It was checked on simulated plants — the development and test plants
reused, plus new, quieter ones — where it raised recall by 2–9 points (2 on the new plants) at
nearly the same precision. A second plant is what would settle it.

---

## Results

| | |
|---|---:|
| Real plant: trains · cleanings · complete cycles | 3 · 18 · 15 |
| Simulated plants: development · test · checks | 30 · 30 · 45 |
| Median error, the dates it flags vs the rest (real plant) | **59 vs 12.5 days** |
| Median error of the crossing date, test plants · real plant | 19 days · 28 days |
| P10–P90 coverage: test plants · real plant · quieter simulation | 0.78 · 0.89 · 0.69 |
| The operators' cleanings the plan had at "review" or "plan" the week before (rules as adopted after the test) | 13 of 18 |
| Tests | 269 (whole repository) |

Simulation is used for everything that needs a known answer: the true crossing day without the
cleaning. **The real deposit is an external test set, spent once.** No forecast setting was tuned
on it; one plan rule was added after seeing it, and is labelled that way.

---

## What it found

### Normalisation decides the answer — more than any model

When production changes, the pressure drop changes with it. A plant has to divide that out before
it can see fouling, and the exponent it divides by matters:

| Normalisation | Fouling it invents when nothing fouls (SD) | Forecast error |
|---|---:|---:|
| linear flow ratio | 12.6 % | 32.7 days |
| vendor exponent 1.5 | 4.8 % | 28.0 days |
| **the plant's own exponent, measured from its own flow changes** | **2.7 %** | **20.0 days** |

The review level is +15 %. A vendor normalisation moves the signal by up to 8 % with nothing
fouling. Physics (friction in a spacer channel) says the exponent is 1.7–1.8; the plant can measure
it from day-to-day production changes to within 0.03, and `/ro/normalise` does.

**And the transmitter matters as much:** taking the pressure drop as the difference of two pressure
transmitters instead of one differential transmitter raised the forecast error from 19 to 30 days.

### The forecast is blind on the floor and late on acceleration

- Of 151 real crossings, **85 came while the forecast said "not in sight"**: the pressure drop was
  sitting at its clean baseline, and a trend that has not started cannot be extended.
- On accelerating, biofilm-like fouling a straight line is **late by a median 23 days**
  (simulation). Plan by the P10 date for a train with that history.
- The flag works on both: dates it flags — further ahead than twice the days the line has seen,
  or from a window mostly at the baseline — missed by a median **59 days on the real plant**, the
  rest by 12.5.

### The gap law is right, and worth a day or two

As fouling closes the channel the pressure drop accelerates (ΔP ∝ gap^−2.2). But between 1.00 and
1.30, where plants clean, it bends less than 0.015 from a straight line. Fitting through it gained
one to two days, in simulation and on the real plant; not adopted, kept as an option.

### Three real cleanings had no hydraulic reason

Replaying the weekly plan over two years — with the margin rule added after the test, so this is
optimistic — 13 of the operators' 18 cleanings came when the plan said "review" or "plan". Three came with the last week's DP_norm at 1.04–1.11. Before one of them, salt passage had been
about 60 % above its baseline for a month — a reason a pressure-drop plan cannot see. Before the
other two, none of the four published indicators had moved. The data do not say why they were done.

### Beside the published rule: it acts at the level; the plan gives notice

The deposit's authors propose an objective trigger — clean when DP_norm reaches 1.20 — and show it
would clean less often than the operators or the vendor rules. That rule acts once the level is
reached. The weekly plan tries to say it in advance. Side by side on the same real decisions
(`step8_head_to_head.py`):

| Level 1.20 (the published trigger) | Published rule | Weekly plan |
|---|---:|---:|
| Crossings it saw coming (7 with a decision before them) | 0 — it acts at the level | **6** |
| Notice before the crossing, median | — (acts ~4 days after, at the next weekly check) | **22.5 days** (2–38) |
| Warnings that did not come true within 28 days | none — it never warns | 3 of 16 (precision 0.81) |

The two are complementary, not rivals: the threshold says *when it is time*; the plan says *how
long until then, and which train first*, so a cleaning can be booked with crew, chemicals and the
other trains in mind. Seven crossings rank nothing, and the plan's margin rule was added after the
real test (above), so its column is optimistic. On simulated plants, where the answer is known, the
plant's own normalisation exponent also removes most of the fouling a vendor exponent invents —
an error that a fixed 1.20 trigger inherits as much as any forecast.

---

## How it works

```
normalise  →  cycle  →  forecast  →  weekly plan
 (physics)   (cut at    (line, P10-P90,  (rules in a fixed order,
             cleanings)  extrapolation    an action and an owner
                         flag)            per train)
```

1. **Normalise.** ΔP ∝ Q^(2−m) μ^m: the flow exponent (2 − m ≈ 1.75) and the viscosity exponent come
   from channel friction, and the plant's own exponent can be fitted from its own flow changes.
2. **Cut at cleanings.** Every score uses the cleaning cycle as its unit, never the day.
3. **Forecast.** A robust straight line through the last six weeks of the current cycle; plausible
   lines drawn around it for the P10–P90 date; a flag beyond twice the window seen; "not enough
   data" said plainly, never guessed.
4. **Plan.** Manufacturer limit → too little data → at the review level → within 0.03 of it or a
   crossing inside four weeks → rising further out → flat or falling. A cleaning that removed less
   than a quarter of the rise is flagged as a procedure problem, not a scheduling one. Two trains
   are never planned into one week if that leaves fewer than two running.

---

## Run it

```bash
pip install -e ".[dev,viz]"
pytest tests -q

python projects/p03_ro_fouling/step1_audit.py              # download + md5, read the deposit
python projects/p03_ro_fouling/step2_physics.py            # normalisation, exponent fit, gap law
python projects/p03_ro_fouling/step3_simulator.py          # simulated cycles against real ones
python projects/p03_ro_fouling/step4_forecast_dev.py       # every choice, development plants
python projects/p03_ro_fouling/step5_forecast_test.py      # test plants, once
python projects/p03_ro_fouling/step6_external_test.py      # the real deposit, once
python projects/p03_ro_fouling/step7_matched_simulator.py  # post hoc: a quieter simulator
```

**Notebooks** — the same steps cell by cell, runnable in Google Colab with nothing installed:

| | |
|---|---|
| [01 · Read the deposit](notebooks/01_read_the_deposit.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMtaleb/chemai/blob/main/projects/p03_ro_fouling/notebooks/01_read_the_deposit.ipynb) |
| [02 · The physics of the pressure drop](notebooks/02_physics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMtaleb/chemai/blob/main/projects/p03_ro_fouling/notebooks/02_physics.ipynb) |
| [03 · A plant whose truth is known](notebooks/03_simulator.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMtaleb/chemai/blob/main/projects/p03_ro_fouling/notebooks/03_simulator.ipynb) |
| [04 · The forecast, its range and its flag](notebooks/04_forecast.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMtaleb/chemai/blob/main/projects/p03_ro_fouling/notebooks/04_forecast.ipynb) |
| [05 · The real test and the weekly plan](notebooks/05_real_test_and_plan.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMtaleb/chemai/blob/main/projects/p03_ro_fouling/notebooks/05_real_test_and_plan.ipynb) |

**Service**

```bash
python -m chemai.api.run      # http://localhost:8000/ro/
```

| Endpoint | Answers |
|---|---|
| `POST /ro/plan` | every train: action · week · evidence · confidence · owner |
| `POST /ro/forecast` | one train: date to the review level, P10–P90, extrapolation flag |
| `POST /ro/normalise` | logged flows, pressures, temperatures → DP_norm, with the plant's own exponent |
| `GET /ro/health` | the settings, the limits, the data source |

```bash
curl -X POST localhost:8000/ro/forecast -H "Content-Type: application/json" -d '{
  "cycle": {"name": "Train A", "day": [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42],
            "dp_norm": [1.050,1.054,1.061,1.064,1.071,1.073,1.080,1.085,1.088,1.094,
                        1.099,1.103,1.109,1.112,1.118]},
  "review_level": 1.15}'
```

```json
{"status": "crossing", "level_now": 1.118, "rate_per_week": 0.011,
 "days_to_level": 20.0, "days_lo": 18.8, "days_hi": 21.4,
 "extrapolation_ratio": 0.51, "in_envelope": true, "warnings": [], "...": "..."}
```

Send the same cycle with one reading a week and the answer is `"status": "insufficient"` — six
points cannot carry a date, and the service says so instead of guessing.

**Data** (not committed — `step1_audit.py` downloads it and checks the published md5):
Su, Y.H., Meng, F.C. and Chang, P.Y. (2026) *Auditable clean-in-place decision support from
routine SWRO SCADA.* Membranes 16(8), 257,
[doi:10.3390/membranes16080257](https://doi.org/10.3390/membranes16080257). Data:
[doi:10.5281/zenodo.21055948](https://doi.org/10.5281/zenodo.21055948), CC-BY-4.0.

---

## ⚠️ Limits

1. **On the real plant the date forecast did not beat a simple margin rule**, and its level
   forecasts were no better than persistence. The plan's combined rule is scored on that plant
   after the fact.
2. **One plant**: three trains, 15 complete cycles. Not a validation.
3. **Blind while the pressure drop sits at its clean baseline**; late on accelerating fouling.
4. **The P10–P90 range is rough**: 0.78, 0.89 and 0.69 coverage in three settings.
5. **Hydraulic only.** Scaling and membrane damage show first in permeate flow and salt passage.
6. **Normalisation decides the answer.** Measure the plant's own exponent; prefer a differential
   transmitter.
7. **Upsets** — a bloom, a pretreatment failure — change the rate overnight.
8. **Gaps.** A third of the real weekly plans could not forecast for lack of recorded days.
9. **The review level and manufacturer limits are inputs**, not findings: replace the defaults
   (1.15; 1.0 bar per element, 3.4 bar per vessel) with the plant's own.

---

## The validation record

[`VALIDATION.md`](VALIDATION.md) holds 17 sections in the order the work was done: what the deposit
turned out to be (three sources disagreed about its baselines), the physics, the simulator and where
it differs from the real plant, every development choice, the test plants, the declaration and the
external test, the rule added after it, a check on a quieter simulator, the eighteen errors that
became tests, and a side-by-side with the published trigger.

**What this is:** a deployed prototype, complete from a public deposit to a service anyone can call,
tested once on a real plant with the result reported as it came out.
**What it is not:** a plant system. No historian connection, no authentication, no monitoring, no
field validation, and no engineer has yet acted on one of its dates.
