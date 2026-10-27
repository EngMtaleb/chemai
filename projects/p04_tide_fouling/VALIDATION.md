# Project 4 — validation notes

Every claim in README.md traces to a section here, and every number here to a
file in `results/` or a test in `tests/`.

## 1. The record, as found

Public UF pilot record, Zenodo 4630769 (CC0; also Dryad doi:10.5068/D1310B),
md5 `317ba90d029d86fd7d69208c53b3c4d6`. An ultrafiltration unit pre-treating
seawater for a reverse-osmosis pilot. 180 run files in the depositors'
`training/` (150) and `test/` (30) folders. Source: `results/step1_audit.json`.

| finding | count | what was done |
|---|---|---|
| csv with a logged clock (Excel day number) | 149 + 1 without header | used |
| files with no clock (sheets, or csv saved from sheets) | 30 | not usable against a clock |
| clocked files whose first sample is not on the day their name says | 6 (one a year off, one with day and month swapped) | excluded, not guessed |
| malformed lines (39 fields where 18 belong) | 102 lines in 5 files | skipped and counted, never repaired |
| clocked runs shorter than 6 h | 37 | excluded |
| **kept** | **107 runs, 5,105.8 h, 2012-06-23 → 2015-04-01** | |

- **The clock is local civil time.** In all 34 kept runs of 48 h or more, the
  daily temperature maximum falls between 12 and 19 h (median hour 16). A clock
  logged in UTC would put it near midnight.
- **Turbidity is often not feed turbidity.** The tag reads −2.5 (no signal) in
  63 kept runs and about 0.03 NTU (filtrate) in 25. Only 19 runs, all in 2013,
  read the feed (median ≥ 0.20 NTU): 1,517.9 h.
- **The depositors' split is not separated in time.** Two `test/` files sit
  inside continuous `training/` episodes (episodes found by chaining files
  whose start follows the previous end within 30 min): one starts 28 minutes
  after a training run ends, the other is followed by a training run 10
  seconds after it ends. A model scored on that split has seen the minutes
  next to its test data. (An earlier version of this note said three files;
  the episode table shows two.)

## 2. Filtration segments

Source: `results/step2_segments.json`; code `chemai/features/uf_segments.py`;
tests `tests/test_uf_segments.py`.

- Three elements. In 2012 all three filter together and a backwash stops them
  all; from March 2013 two filter while the third is backwashed and rested, and
  the pair changes every 18–28 minutes.
- A segment is a stretch of steady filtration (total flow within ±15 % of the
  run median, TMP > 0.5 psi) with one set of elements online, a change counting
  only after it holds 60 s. The first 180 s (switching and backwash-pulse
  transients) and the last 60 s are dropped; at least 10 minutes and 5 binned
  points must remain.
- Fouling rate = slope of ln R against time on 1-minute medians (2-minute
  medians where the logger sampled every 120 s), R = TMP/(μ(T)·Q). The
  viscosity term removes the TMP swing a temperature change causes; the test
  `test_fouling_slope_is_recovered_and_not_fooled_by_temperature` checks it.
- 14,998 segments in 106 runs (one run never filtered steadily), 86 episodes.
  Fouling rate median 13.2 %/h, interquartile 8.7–18.6 %/h, 3.7 % negative.
  A single segment's slope is noisy: its standard error is a median 37 % of its
  size. The test works on thousands of segments, not on any one.

## 3. Power, before the test

Source: `results/step3_power.json`; `chemai/evaluation/tide_test.py:power_curve`.
An M2 wave of known size is added to AR(1) noise with each run's own residual
spread and persistence, and the full declared test (999 surrogates) is run on
it, 200 times per size. The data's real alignment with the tide is not used.

| response | 2 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|
| fouling rate (median 13.2 %/h) | 3 % | 43 % | 99.5 % | 100 % |
| feed turbidity (median 0.51 NTU) | 6.5 % | 38 % | 67 % | 97.5 % |
| pump speed (median 1.15) | 100 % | 100 % | 100 % | 100 % |
| pump discharge pressure (median 3.58 psi) | 100 % | 100 % | 100 % | 100 % |

Size = 5–95 % range of the wave as a share of the response's median; cell =
share of trials found at p ≤ 0.01. The hydraulic tide that section 5 finds is
smaller than any size on this grid: 0.2 % of median pump speed and 0.9 % of
median discharge pressure.

## 4. The declared test, as run

Declaration: commit `1989858` (DECLARED_BEFORE_RUN.md), pushed before step 4
ran. Source: `results/step4_tide_test.json`.

| test | variance explained | null 95th pct | p | verdict |
|---|---|---|---|---|
| **H1 fouling rate** | 0.101 % | 0.138 % | **0.244** | fails (rule: p ≤ 0.01) |
| C1 pump speed | 0.048 % | 0.086 % | 0.349 | fails |
| C2 pump discharge pressure | 0.467 % | 0.404 % | 0.015 | fails, narrowly |
| S1 feed turbidity (2013) | 1.84 % | 1.88 % | 0.063 | fails |
| placebo periods on fouling | 0.060 % | 0.096 % | 0.330 | calibration holds |
| confirmation, fit 2012–13 → score 2014–15 | −1.34 % (worse than nothing) | −1.11 % | 0.248 | fails |

Declared outcome **3**: no tide effect at this intake larger than the minimum
detectable size. From section 3, a tide-locked wave whose 5–95 % range is 10 %
of the median fouling rate (≈ 1.3 %/h) would have been found in 99.5 % of
trials; the fitted tide component's range is 0.83 %/h, below its own null's
95th percentile (0.98 %/h). Calibration held, so the null is trusted.

Reported, not decisive: the depositors' split (fit on `training/`, score on
`test/`) explains −0.06 % (p = 0.47). No daylight-saving correction: p = 0.32.
No covariates: p = 0.14. No winsorising: p = 0.28. M2 phase fitted on
2012–2013 alone: 293°; on 2014–2015 alone: 32° — a tide does not change its
phase between years.

**Checked end to end on the real data.** A known M2 wave added to the real
fouling rates is found by the same pipeline: a wave of 10 % of the median at
p = 0.001 (amplitude recovered 0.68 %/h against 0.67 planted), one of 5 % at
p = 0.029 (missed at the declared line, as section 3 predicts 57 % of the
time). The pooled variance explained was also recomputed by a direct full
least-squares fit, without the projection shortcut: 0.1014 %, the same.

## 5. Exploratory checks — not in the declaration

Run after step 4, to understand the controls. Treat as hints, not results.

- **Phase coherence per episode** (37 episodes of 48 h or more; one M2
  amplitude and phase fitted per episode on the base-model residuals; Rayleigh
  test on the phases, each episode counted once). The declared test weights
  episodes by their variance, so one noisy episode can swamp a small steady
  signal; this check does not.

  | response | mean resultant length | Rayleigh p | mean direction | 2012–13 / 2014–15 directions |
  |---|---|---|---|---|
  | fouling rate | 0.16 | 0.38 | — | 280° / 59° (disagree) |
  | pump speed | 0.39 | 0.003 | 355° | 316° / 14° |
  | pump discharge pressure | 0.40 | 0.002 | 166° | 134° / 180° |

- **Reading.** Pump speed and discharge pressure lock to the M2 phase in both
  halves of the record and point roughly opposite ways — what a suction head
  that rises and falls with the tide does to a flow-controlled pump. The tide
  reaches the pilot's hydraulics, faintly: an M2 amplitude of about 0.01 psi
  in discharge pressure, where a pump drawing straight from the sea would see
  1.4 psi of suction head for every metre of tide. The feed seems to arrive
  through storage or a header that damps the tide. The fouling rate shows no
  such lock.

## 6. What the result is, and is not

- **Is**: at this pilot, with feed delivered as it was, UF fouling is not locked
  to the tide by more than about 10 % of its median rate. Planning backwash and
  coagulant here gains nothing measurable from the tide.
- **Is not**: a statement about intakes that draw straight from the sea. The
  hydraulics suggest this pilot's feed was buffered; an open intake could
  differ. The same test runs on any timestamped plant log.
- Turbidity (2013 only, 11 long episodes) stays undecided: p = 0.063 declared,
  0.07 exploratory.

## 7. Part 2 — the sun, as declared and run

Declaration: commit `6d3dd6e` (DECLARED_BEFORE_RUN_2.md), pushed before step 6
ran. Power: `results/step5_sun_power.json`. Results: `results/step6_sun_tests.json`.

| test | number | p | verdict |
|---|---|---|---|
| **H2** fouling rate follows the time of day | variance explained 0.028 % (null 95th pct 0.092 %) | **0.746** | no daily pattern detected |
| C3 positive control: temperature follows the time of day | 43.8 % (null 9.8 %) | 0.001 | the clock and the test work |
| H2 placebo periods | 0.060 % (null 0.097 %) | 0.330 | calibrated |
| H2 confirmation, fit 2012–13 → score 2014–15 | −0.27 % | 0.897 | nothing to confirm |
| H2 with temperature in the base model | — | 0.385 | same |
| **Solar-hours contrast** (09–15 h against the rest) | −0.8 % of the median fouling rate, 95 % interval **−2.5 % to +1.2 %** | — | not relevant (line: ±10 %) |
| **H3** first 3 h after a restart (81 restarts, 2,503 segments) | +3.1 % of the median, 95 % interval **−3.9 % to +10.0 %** | **0.382** | no restart effect detected; above ~10 % ruled out |
| H3 placebo, pretend restart at 24 h | −1.3 % | 0.842 | calibrated |
| H3 in 2012–13 / 2014–15 | +0.34 / +0.25 %/h | 0.74 / 0.72 | no replication needed — nothing to replicate |
| H3 idle 2–12 h / over 12 h (exploratory) | +1.63 / +0.41 %/h | 0.38 / 0.42 | — |

**Checked end to end on the real data.** A daily wave of 10 % of the median
added to the real fouling rates is found at p = 0.001 (5 %: p = 0.032); an
early-hours step of 10 % added to the real restart windows is found at
p = 0.0007.

**What it means for a plant run on the sun.** On this UF pilot neither fouling
worry about solar operation showed up, and the intervals say how large an
effect could still hide:

- No daylight effect was detected: the hours 09–15 foul within −2.5 % to
  +1.2 % of the rest of the day.
- No restart effect was detected either: the first 3 h after a restart are
  within −3.9 % to +10.0 % — a penalty above about 10 % is ruled out, a smaller
  one is not; a price of 20 % would have been caught 98 % of the time.
- The warm-water bonus is real but small here: the daily temperature swing is
  0.7 °C, and the hours 09–15 average only 0.13 °C warmer than the rest — a
  0.3 % viscosity gain. Not a reason to run in daylight, not a reason against.

So, on this record, a seawater UF pre-treatment run on solar hours, with daily
stops, at the same flux it runs at around the clock, showed no sign of fouling
faster. The price of going solar is then membrane area and storage — sized in
part 3 — rather than fouling. This covers the UF only: the RO trains are not in
the record, and their daily stops need their own test. The limits of section 7
of the second declaration stand: one pilot, and its restarts were the
operators' choice.

## 8. Part 3 — sizing a plant that runs on the sun

Code: `chemai/models/solar_plan.py`; tests: `tests/test_solar_plan.py`; run:
`step7a_uf_energy.py`, then `step7_solar_plan.py`; results:
`results/step7a_uf_energy.json`, `results/step7_solar_plan.{parquet,json}`.
This part is a **design study on a synthetic year**, not a measurement. What
comes from the record and what is assumed:

| input | value | from |
|---|---|---|
| fouling per m³ in daylight-only, stop-start operation at design flux | taken as round-the-clock: no difference detected (a restart penalty below ~10 % not excluded) | part 2 (H2, H3), UF only |
| UF energy | 0.26 kWh per m³ of product: 0.083 kWh hydraulic per m³ of UF feed (median of 106 runs; filtration pressure × feed flow plus backwash pressure × backwash flow), ÷ pump efficiency 0.7, ÷ RO recovery 0.45 | the record + assumptions |
| seawater RO energy | 3.0 kWh/m³ — the plant uses 3.26 kWh/m³ in all | literature range 2.5–3.5 |
| weather | synthetic: day length 10.4–13.6 h, clear-day 6.2 kWh/kWp ± 16 % by season, daily cloud draw; mean 5.26 kWh/kWp/day ("sunny"), 4.73 ("cloudier") | assumption, no site |
| RO under daily stop-start | flushed at each stop | literature (lab study, 2025: a daily rinse recovered permeability) — not tested here |

**Method.** Modules switch on and off at their design flux, so production can
follow the sun without changing the flux. For each design (production
capacity 1–3× the average demand, product-water tank 2–24 h, PV 0.3–1.3 kWp
per m³/day), a linear programme over 8,760 hours finds the least grid energy
any controller could achieve (perfect foresight); a simple rule with no
forecast is replayed beside it. 192 designs × 2 weathers (capacity 1.0× is not paired with tanks over 4 h: it runs all day and cannot fill them).

**Correction (after this section was first written).** The first version
added the UF figure per m³ of UF *feed* (0.12 kWh/m³) to the RO figure per m³
of *product*. Per m³ of product it is 0.26 kWh/m³ — 0.37 if the backwash water
were drawn from the RO feed (the pilot kept filtering at full flow while one
element backwashed, so it is taken here as not). The plant's energy went from
3.12 to 3.26 kWh/m³, and every design was rerun. The conclusions held: the
90 % design below reached 94 % before and 93 % now. `step7a_uf_energy.py` now
regenerates `uf_volumes.parquet`, which had been written by hand-run code.

**Smallest designs reaching a solar share** (sunny year; per m³/day of demand;
fewest kWp first, then least capacity, then least tank):

| solar share | capacity | tank | PV | reached | PV curtailed |
|---|---|---|---|---|---|
| 50 % | 1.5× | 4 h | 0.5 kWp | 55 % | 31 % |
| 75 % | 2.5× | 12 h | 0.5 kWp | 76 % | 6 % |
| 90 % | 3.0× | 16 h | 0.7 kWp | 93 % | 18 % |
| 95 % | 3.0× | 16 h | 0.9 kWp | 97 % | 33 % |

Cloudier year: 90 % needs 3.0×, 16 h, 0.9 kWp (93 %); 95 % needs 1.1 kWp (96 %).

- **Round-the-clock capacity cannot go far on the sun**: at 1.0× capacity the
  plant must run all night, and the solar share stops near 46 % however much
  PV is added. Capacity and a tank are what buy the rest.
- **A forecast is worth little once the tank is big enough**: the simple rule
  is within 0.1–0.2 percentage points of the perfect-foresight plan in the
  median design. With a small tank foresight pays: the median gap is 8 points
  with a 2 h tank, 7 with 4 h and 1.5 with 8 h, and nil from 12 h up (a few
  12 h designs still lose up to 8; at 16 h the largest gap is 1 point) —
  knowing tomorrow's sun replaces storage.
- **Add modules, do not push flux.** Getting 2.5× capacity by running the same
  membrane 2.5× faster would raise fouling per m³ by 2.5^(n−1): ×1.2 at n = 1.2,
  ×1.6 at n = 1.5, ×2.5 at n = 2 (fouling rate ∝ flux^n). The record cannot pin
  n — flux was changed mostly between campaigns — so this is a range, not a
  measurement; any n above 1 says the same thing.
- **For every 1,000 m³/day**, the 90 % design (700 kWp, a 670 m³ tank, a
  3,000 m³/day plant running about 8 h a day) cuts grid electricity from
  3.26 to 0.23 kWh/m³: about 1.1 GWh a year, 440–770 t of CO₂ at a grid
  intensity of 0.4–0.7 kg/kWh.

**What this is not**: a cost optimum (no prices are used: capacity, tanks and
PV are traded in physical units only), a site design (the year is synthetic),
or evidence about the RO membrane under daily cycling (only the UF
pre-treatment was tested).

## 9. The moon as a power source

Run: `step8_sun_and_moon.py`; results: `results/step8_sun_and_moon.{parquet,json}`.
Same design study, with a generic tidal-stream turbine beside the PV: current
from M2 and S2 (so spring and neap), power ∝ speed³ up to the rated speed,
capacity factor 0.35 (typical for tidal stream). It delivers at night and on
cloudy days, on a timetable known years ahead. Part 1 found no tide in this
membrane's fouling, part 2 no sign that it minds when it runs — so, in this
design study, the plant can follow either supply.

| renewable share | PV only | + tidal 0.10 kW per m³/day (≈ 26 % of the energy) | + tidal 0.15 kW (≈ 39 %) |
|---|---|---|---|
| 90 % | 2.5× capacity, 12 h tank, 0.9 kWp | 2.0×, 8 h, 0.9 kWp | 2.0×, 8 h, 0.5 kWp |
| 95 % | 3.0×, 16 h, 0.9 kWp | 2.5×, 12 h, 0.7 kWp | 2.0×, 12 h, 0.9 kWp |
| 99 % | not reached up to 0.9 kWp | 3.0×, 16 h, 0.9 kWp | 2.5×, 16 h, 0.9 kWp |

(Smallest tank first, then least capacity, then least PV.) A tidal turbine
needs a strong current at the site — most sites have none; where one exists,
it buys a smaller plant and a smaller tank.

## 10. The service — the same test on anyone's log

Code: `chemai/api/sun_app.py` (mounted at `/sun`), `series_test` in
`chemai/evaluation/tide_test.py`; tests: `tests/test_sun_api.py`.

`POST /sun/test` takes a timestamped signal (`t`, `value`, optional `run`) and
runs the declared method on it. A run is the caller's label, or else a gap
over 6 h starts one. Each run is cut into pieces of at most 14 days, counted
from its own start, and every piece is shifted on its own in the null — whole
weeks for the tide, 1–23 h for the clock. The base model is a spline per piece
and, for the tide, the time of day. The placebo periods (11, 13.5, 27, 5.3 h)
run beside every answer, and the verdict is "locked" only when the main p is
at most 0.01 **and** the placebos are quiet. Fewer than 100 points, 72 hours or
4 pieces is answered "insufficient", not guessed.

**One change from the pilot analysis.** In the service the placebos always
carry the time of day in their base, for the clock test too. In the declared
sun test (section 7) they did not, and it made no difference: the pilot had
no daily wave. On a log that has one, the wave leaks into the 27-hour placebo
whenever runs start near multiples of nine days (the beat of 24 and 27 hours);
without the change, a real daily wave was flagged as untrustworthy
(placebo p = 0.005 on the nine-day test log below).

**Check on made-up logs** (20 seeds each; 10 runs of 6 days, 30-minute
samples, red noise; a wave of amplitude 0.25 — about the noise's own standard
deviation; 199 surrogates, so the smallest possible p is 0.005):

| log | test | called "locked" | median p | placebo p ≤ 0.01 |
|---|---|---:|---:|---:|
| no wave | tide | 0 / 20 | 0.56 | 0 |
| daily wave | tide | 0 / 20 | 0.56 | 0 |
| tide wave (M2) | tide | 20 / 20 | 0.005 | 0 |
| daily wave | clock | 20 / 20 | 0.005 | 0 |
| no wave | clock | 0 / 20 | 0.63 | 0 |
| tide wave | clock | 0 / 20 | 0.95 | 0 |

A daily rhythm — the one a solar plant has by design — never came out as a
tide. What the service does not do: a power analysis. "Not locked" on a short
or noisy log is not evidence of absence; the response gives the share of the
variance a wave would have needed to beat chance (`null_partial_r2_q95`), and
section 3 shows how the pilot's power was measured before its test.

`GET /sun/design` reads the design sweep of sections 8 and 9 (624 designs,
generated by `make_sun_designs.py` from the step 7 and 8 results). The
per-1,000 m³/day figures are the same numbers scaled; CO₂ on the page uses
0.4–0.7 kg per grid kWh.
