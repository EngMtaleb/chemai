chemai

Reusable toolkit for chemical engineering AI projects. Physics-formed features, calibrated uncertainty, and systems that say where they stop working.

Eng. Mohammed Taleb — chemical engineer · LinkedIn

	
Live	chemai-tr2c.onrender.com
Soft sensor API	/docs
Loop diagnosis	/loops/

Free hosting — the first request after idle takes up to a minute to wake.

Project 1 — Distillation Soft Sensor

Laboratory vapour pressure takes 4 to 8 hours. This estimates it from two process measurements, so the operator stops steering the one thing he can see to control the one thing he cannot.

	
R² on a different operating campaign	0.980
Interval coverage (95% nominal)	94%
Error ratio outside the validated envelope	7.8×
Inputs	2

What it found: writing temperature in the form Antoine's equation requires — 1/T instead of T — cut error by 32% with no new variable. Two inputs beat all twenty-one. And the envelope detector was tested: error on the points it flags is 7.8× worse, so the refusal is real.

📄 Full write-up · docs/VALIDATION.md

Project 2 — Control Loop Performance

A sticking valve makes a loop oscillate; the oscillation spreads; ten loops look faulty. This says which one to fix, and who fixes it — or says the data cannot support an answer.

	
Real plant loops tested against published labels	141
Stiction caught by the adopted rule (precision 0.88)	12 of 14
Eastman root cause found among 30 loops, unaided	tag 22, 13× the next candidate
Stiction below the load noise, unrecoverable from OP and PV	~50%

What it found, and most of it is negative: the Harris index rates tight-tuned loops above healthy ones. A five-class classifier trained on simulation does not beat a single-feature baseline on plant data — and the cause was measured, not guessed. A historian deadband turns stiction into "tuning": a confident wrong answer, which sends the control engineer instead of maintenance.

📄 Full write-up · 22-section validation record

Quick start
bash
pip install -e ".[dev]"
pytest tests -q                              # 197 tests, no data needed
python -m chemai.api.run                     # http://localhost:8000
bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 450, "pressure": 228}'
json
{"vapour_pressure": 38.8, "interval_lower": 35.84, "interval_upper": 41.75,
 "in_envelope": true, "distance": 1.26, "warning": null}

No endpoint returns a bare number. A caller who sees only a value will trust it unconditionally, so every answer carries its interval, its confidence, or the reason it cannot be given.

Data is not committed. See each project's write-up for where to obtain it.

Structure
chemai/
├── config.py        every constant in one place, with the reason beside it
├── data/            loaders · campaign split · control-loop simulator · archive compression
├── features/        physics-formed features · loop indices · data quality · shape · cascade · propagation
├── models/          estimator · uncertainty · envelope · loop classifier · serialisation
├── evaluation/      metrics, including the ones usually skipped · the weekly loop report
├── pipeline.py      raw loop data in, diagnosis and owner out
└── api/             FastAPI: soft sensor, and loop diagnosis under /loops
projects/            one folder per project: scripts, validation record, write-up
tests/               197 tests — they run without any dataset
How this is built

Every limit is published, not hidden. GET /loops/health serves the five things the diagnosis cannot do. The demo page prints them under the result.

Negative results are kept. A classifier that lost to a one-feature baseline, a non-linearity measure that ranked the known root cause last, an index that flatters aggressive tuning — each is recorded with the number that shows it.

Every error found became a test. coefficient_sign_is_physical fails if the model inverts against thermodynamics. split_is_not_random fails on leakage. rejects_celsius_temperature catches the silent unit error. test_a_deadband_turns_stiction_into_tuning pins a wrong answer that looks confident.

A model carries its provenance. The serialised estimator records the library version that fitted it, and /health reports a mismatch — a warning printed to a log nobody reads is not a safeguard.

What these are: deployed prototypes, complete from raw data to a service anyone can call. What they are not: plant systems. No historian connection, no authentication, no monitoring, no automatic retraining, and no field validation.

Implementation assisted by AI tools. Engineering framing, interpretation and validation are the aut
