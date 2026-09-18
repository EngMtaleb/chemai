# chemai

Reusable toolkit for chemical engineering AI projects.
Physics-formed features, calibrated uncertainty, and an operating-envelope check.

---

## Project 1 — Distillation Soft Sensor

Estimates laboratory vapour pressure from two process measurements.

| Metric | Value |
|---|---:|
| **R² out-of-regime** | **0.980** |
| MAE out-of-regime | 0.645 |
| MAE inside envelope | 0.495 |
| Interval coverage (95% nominal) | 94% |
| Features | **2** (`1000/Temp9`, `PressureC1`) |

Trained on one operating campaign, evaluated on another with **one third the reboiler duty**.
The relationship is thermodynamic, not operational — which is why it survives the regime change.

📄 `docs/EDA.md` — what was found in the data
📄 `docs/VALIDATION.md` — where the model fails, and why

---

## Project 2 — Control Loop Performance *(in progress — week 6 of 7)*

PI loop simulator for flow, pressure and level, with valve stiction, tight and sluggish tuning and
external oscillation — and the **valve position (MV) recorded**, which plant data lacks.

```bash
pip install -e ".[viz]"
python projects/p02_control_loops/week1.py                 # simulator + Harris study
python projects/p02_control_loops/week2_data_quality.py     # data-quality checks on every real loop
python projects/p02_control_loops/week2_oscillation.py      # is the loop oscillating at all?
python projects/p02_control_loops/week2_shape.py            # triangular or sinusoidal? (baseline)
python projects/p02_control_loops/week2_classifier.py       # classifier vs baseline
python projects/p02_control_loops/week3_report.py           # the weekly report
python projects/p02_control_loops/week4_cascade.py          # cascade: structure and source
python projects/p02_control_loops/week5_propagation.py      # plant-wide propagation
python projects/p02_control_loops/week6_compression.py      # what the historian throws away
```

**Service** — the diagnosis is served alongside the soft sensor, under `/loops`:

```bash
python -m chemai.api.run          # http://localhost:8000/loops/
```

`/loops/` is a one-page demo: four worked examples whose fault is known, or upload a CSV with
`sp,pv,op`. `POST /loops/analyse/loop` diagnoses one loop; `POST /loops/analyse/plant` takes a whole unit and
finds what they share — ten faults are often one fault and nine victims. Every answer carries its
evidence, a confidence band, an action and an owner; `GET /loops/health` serves the limits.

Data (not committed): SACAC repository in `data/sacac/`; ISDB `isdb10.mat` in `data/isdb/`
(from `sites.ualberta.ca/~bhuang/ISDB.zip` — cite Jelali & Huang, 2010).

Findings so far: the Harris index rates tight-tuned pressure and level loops **above** healthy ones;
half of all stiction that sits below the load noise is unrecoverable from OP and PV alone; and a
five-class classifier trained on simulation does **not** beat a single-feature baseline on plant data.
📄 `projects/p02_control_loops/VALIDATION.md`

**Explicit limits**
- Diagnoses stiction and valve saturation. Other actuator faults (positioner hysteresis, air-supply
  leaks, wrong valve sizing) are out of scope: no labelled data exists to test them against.
- Cascade loops are **diagnosed** — which loop is the source — but not **scored**. Assessing the joint
  performance of master and slave with a cascade-specific Harris index is advanced research that
  would take weeks, and is out of scope.

---

## Structure

```
chemai/
├── config.py        every constant in one place
├── data/            loading · derived-column detection · campaign split
│                    SACAC and ISDB loaders · control-loop simulator · archive compression
├── features/        the form the physics requires · loop performance · data quality · shape · cascade · propagation
├── models/          estimator + uncertainty + envelope
├── evaluation/      metrics, including the ones usually skipped · weekly loop report
└── api/             FastAPI services: soft sensor, and loop diagnosis under /loops
```

---

## Quick start

```bash
pip install -e ".[dev]"
pytest tests -q                              # 196 tests, no data needed
python projects/p01_soft_sensor/train.py     # needs data/ populated
```

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

Full detail in `docs/VALIDATION.md`.

---

## Data

Industrial distillation tower dataset from [OpenMV.net](https://openmv.net/info/distillation-tower).
**Not committed** — place `distillation-tower.csv` in `data/`.

---

*Implementation assisted by AI tools. Engineering framing, interpretation and validation are the author's.*
