# Project 2 — Control Loop Performance

**Which ten loops do I fix this week, and who fixes them?**

A refinery runs hundreds of PID loops. A sticking valve makes one oscillate, the oscillation spreads
through the unit, and by Monday morning ten loops look faulty. Nine of them are victims. The common
repair — retuning the controller — does not fix a sticking valve: it only slows the limit cycle down,
which is how weeks get spent on a mechanical problem.

This project diagnoses loops from what a plant historian actually stores: setpoint, measurement and
controller output. It answers with **the fault, the evidence, a confidence band, the action, and the
owner** — or it says the data cannot support an answer.

**Live:** [chemai-tr2c.onrender.com/loops/](https://chemai-tr2c.onrender.com/loops/) — four worked
examples whose fault is known, or upload your own CSV.

---

## Results

| | |
|---|---:|
| Real plant loops assessed (SACAC + ISDB, duplicates removed) | **141** |
| Labelled loops used as an external test set | 76 |
| Stiction caught by the adopted rule, precision 0.88 | **12 of 14** |
| Eastman plant-wide root cause, found among 30 loops unaided | **tag 22**, 13× the next priority |
| Cascade pairs discovered from structure alone, across 3 plants | 8 |
| Tests | 196 (whole repository) |

Simulation is used for training and for anything needing a known answer. **Plant data is an external
test set, spent once.** Nothing was retuned after seeing how it scored.

---

## ⭐ What it found

### The standard index rewards the wrong thing

The Harris index is the textbook measure of loop performance. On simulated loops it rates
**tight-tuned pressure and level loops above healthy ones** (0.60–0.62 against 0.43–0.45), because
minimum-variance control *is* aggressive control. Raising the controller gain raises the index —
until the loop is one small process change away from instability. A 20% gain change that the healthy
loop absorbs makes the "better" loop unstable.

A real averaging level loop, labelled healthy by the expert who published it, scores **0.013** — the
worst possible mark for a surge drum doing exactly its job.

**So Harris ranks; it does not diagnose.** It is not used in the diagnosis at all.

### Half of all stiction is not in the data

Simulated sticky valves, split by how far the valve's slip jump stands above the load noise:

| Stiction | Recall |
|---|---:|
| Buried (jump below the noise) | **0.50** |
| Visible (jump above 2× the noise) | 0.83 |

Below the noise the fingerprint is not in OP or PV at all, so **no algorithm recovers it**. The fix
is not a better model — it is logging the valve position, which modern positioners already measure
and historians usually discard. This is the project's strongest engineering recommendation, and it is
now a number rather than an opinion.

### A trained classifier lost to one threshold

A five-class random forest on three features, cross-validated by loop, reaches 0.83 on simulation.
On plant data it catches 11 of 14 stiction cases against the baseline's 12, with more false alarms.

The cause was measured: **Harris is three times lower on plant data than in simulation** (0.07 vs
0.22), because a plant rarely knows its dead time and the minimum estimate must be used. The model
had learned that a low Harris means sluggish tuning — so it called eight real stiction loops
"sluggish".

**The baseline is what ships.** A model that does not beat one threshold on one feature has not
earned its complexity.

### The historian can turn a right answer into a wrong one

Twelve known-stiction loops, compressed the way a plant historian compresses:

| Compression | Stiction found | **Read as tuning** |
|---|---:|---:|
| none | 9 | 1 |
| logged every 5 s instead of 1 s | 1 | 3 |
| deadband of 1 × std | **0** | **8** |

A slower logging rate mostly yields *undetermined* — the report admits it does not know. **A deadband
yields a confident wrong answer**, which sends the control engineer to a loop whose valve is stuck.
Corners are the stiction fingerprint, and compression rounds corners first.

Working limit: **ten samples per oscillation cycle**, and ten cycles in the record. Below that the
service refuses to diagnose.

### One plant, one fault, ten victims

Thirty Eastman loops recorded together; the published root cause is tag 22, a sticking valve on a
level loop. Without being told anything, the analysis finds **eleven loops sharing one 114-minute
oscillation** and puts tag 22 first, at 13× the priority of the next loop.

The weekly report then stops listing eleven faults and says: *one propagation event, this is the
candidate source, the other ten are probably victims — no work orders until the source is repaired.*

**And the ranking's limits were measured too.** In a simulated plant the source and its nearest
victim tie to within 0.001. On a second real case the ranking points at the wrong loop, for a reason
that is now understood: a non-linear source spreads its power into harmonics, so *the very
fingerprint that identifies a sticky valve counts against it here.* The source list is a candidate
list for an engineer to confirm, never a verdict.

---

## How it works

```
data quality  →  indices  →  diagnosis  →  plant view
   (week 2)     (weeks 1-2)   (weeks 2-3)   (weeks 4-5)
```

1. **Data quality first.** Five traps that have nothing to do with the valve: a frozen sensor, a loop
   in manual, a saturated valve, a quantised transmitter, a moving setpoint — plus an archive too
   coarse to carry a waveform. A frozen sensor looks like the calmest loop in the plant while the
   integral drives the process away.
2. **Indices.** Oscillation regularity (is there a cycle at all?) and the curve-fitting shape index
   (triangular, from a valve, or sinusoidal, from tuning). Loop type selects the signal: fast loops
   are read on OP, level loops on PV, because a tank integrates a square wave into a triangle.
3. **Diagnosis.** A regular oscillation with a triangular waveform is stiction; data-quality flags
   outrank the shape; everything else is *undetermined*.
4. **Plant view.** Cascade structure (a slave's setpoint *is* its master's output) and shared
   oscillation across loops recorded together.

Each step has a teaching notebook that builds it from first principles before any code enters the
package.

---

## Run it

```bash
pip install -e ".[dev,viz]"
pytest tests -q

python projects/p02_control_loops/week1.py                  # simulator · Harris study
python projects/p02_control_loops/week2_data_quality.py     # five traps, on every real loop
python projects/p02_control_loops/week2_oscillation.py      # is it oscillating at all?
python projects/p02_control_loops/week2_shape.py            # triangle or sine — the baseline
python projects/p02_control_loops/week2_classifier.py       # classifier vs baseline
python projects/p02_control_loops/week3_report.py           # the weekly report
python projects/p02_control_loops/week4_cascade.py          # cascade structure and source
python projects/p02_control_loops/week5_propagation.py      # plant-wide propagation
python projects/p02_control_loops/week6_compression.py      # what the historian throws away
```

**Service**

```bash
python -m chemai.api.run      # http://localhost:8000/loops/
```

| Endpoint | Answers |
|---|---|
| `POST /loops/analyse/loop` | one loop: diagnosis · evidence · confidence · action · owner |
| `POST /loops/analyse/plant` | a unit: the ranked report, the propagation event, cascade candidates |
| `GET /loops/health` | the diagnoses it can make, and the limits it works under |

**Data** (not committed):

- SACAC PID data repository → `data/sacac/`
- ISDB, Jelali & Huang (2010), `isdb10.mat` → `data/isdb/`
  (`sites.ualberta.ca/~bhuang/ISDB.zip` — cite the book)

---

## ⚠️ Limits

1. **Half of buried stiction is unrecoverable** from OP and PV. Log the valve position.
2. **Ten samples per cycle and ten cycles per record**, or no diagnosis is given.
3. **Stiction and saturation only.** Positioner hysteresis, air-supply leaks and wrong valve sizing
   have no labelled data to test against.
4. **Cascade and propagation results are candidates**, confirmed in the field, not verdicts.
5. **Cost ranking needs the plant.** Severity is comparable only within one plant in its own units,
   and location weights (1 utility · 2 energy · 3 specification or safety) are written by an
   engineer. Without them the report ranks on diagnosis alone and says so.
6. **Cascade loops are diagnosed, not scored.** A cascade-specific Harris index is out of scope.
7. **Mixed faults blur every tool here** — stiction *and* tight tuning together reads as one or the
   other.

---

## The validation record

`VALIDATION.md` holds 22 sections written as the work happened: what was decided before seeing
results, what the data then said, and what had to change. It includes the numbers behind every claim
above, three conflicts found between published labels and the data they describe, and the measures
that were tried and rejected.

**What this is:** a deployed prototype, complete from raw data to a service anyone can call, tested
against 141 real loops with published expert labels.
**What it is not:** a plant system. No historian connection, no authentication, no field validation,
and no engineer has yet confirmed a single one of its diagnoses on a real valve.
