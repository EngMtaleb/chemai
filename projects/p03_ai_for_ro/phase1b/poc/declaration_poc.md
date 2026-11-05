# Phase 1B, proof of concept — a simulated plant record with every column, and several models on it

**Approved by the owner on 3 November 2026 at 22:40 (UTC+3).** Written the same evening, before any POC data were generated or
any model was run; its SHA-256 was recorded at approval, before anything ran. Nothing here changes after that; any later change
is listed as a change, with its reason, next to the original.

## Why

- Phase 1: on the real record, no method beat *no change*; the likely reason is that the record holds the indicators, not the
  causes of fouling.
- Phase 1B (declared 28 October 2026) first asked a simulator to behave like this plant (the gate). No setting of 800 passed, and
  it stopped as declared. Its question — how much logging the causes would help plants like this one — stays open. This proof of
  concept does not answer it.
- The owner's question (3 November 2026): if a plant logged everything, what would the whole workflow look like? This shows it,
  end to end, on a simulated plant record with the columns a seawater RO plant typically logs — including columns that have
  nothing to do with fouling — and noise.

## What it can and cannot say

- It shows the steps. Its numbers describe this simulator under the assumptions below. They are not expected results for any real
  plant; no number from it is offered as one, and none enters a recommendation (the deck's “What to log” page keeps “no promised
  gain”).
- We plant the causes in the simulator ourselves, so a model that is given them is expected to gain. A simulator flatters
  machine learning: in the repo's earlier ML-ladder project, ML beat a simple rule on simulated plants, and on the real plant it
  was a trade. Every figure carries **SIMULATION · PROOF OF CONCEPT**.
- It is not the map phase 1B stopped: no calibration, one setting fixed here in advance, no claim about plants like the real one.

## The simulated plants

- Simulator version 2 (`phase3b/sim_v2.py`), unchanged, on the base `chemai/data/ro_sim.py`, unchanged: real feed temperature,
  the fouling driver Z, the flux factor and the clock from day 0 switched on; attempt 2's slow movement of the ΔP reading off
  (`wander_sd = 0`); everything else as in the base.
- **Fixed here, not tuned:** σ_Z = 0.75, the middle of phase 1B's declared range (0–1.5). The feed-quality instruments see half
  of the driver, with half its spread as noise: s = 0.5, r = 0.5, the middle of phase 1B's grid.
- **Three plants, one per fouling law** — linear (seed 4101), biofilm (4102), asymptotic (4103) — drawn as phase 1B drew its
  plants (`sample_plant(default_rng(seed), cfg, law)`); seeds never used before. Each: 3 trains, 772 days; a random window of the
  daily sea temperature (NASA POWER, as in phase 1B).
- The new columns below are drawn from a fourth random stream (`default_rng([seed, 4])`), so the plant itself is draw for draw
  what version 2 makes without them.
- The table is generated once and saved with its SHA-256 before any model runs.

## The table: one row per train per recorded day

| group | columns | what moves them |
|---|---|---|
| **Causes** (drive fouling here) | feed temperature (°C); permeate flux (LMH); SDI15; turbidity (NTU) | real sea temperature; the flux the operator runs; SDI and turbidity see half of the fouling driver Z |
| **Operation** | feed, permeate and concentrate flow (m³/h); recovery; feed TDS (mg/L); permeate pressure (bar); running; cleaning day | demand shared across the trains that run; the operator's cleanings |
| **Results** (fouling shows here) | feed and concentrate pressure (bar); ΔP from a differential transmitter (bar); permeate TDS (mg/L); the four indicators below | the simulator's physics |
| **No link** (logged, play no part here) | feed pH; ORP after dechlorination (mV); free chlorine after dechlorination (mg/L); doses of sodium bisulfite, antiscalant and ferric coagulant (mg/L); cartridge-filter ΔP (bar) | their own slow movement, set-point steps and noise |

The fouling state, the true ΔP, the driver Z, the upsets and every other hidden quantity are never logged and never enter a model
(a test checks the column list). Feed quality is one intake: SDI, turbidity and the no-link columns are plant-wide, the same row
value for every train on a day.

**The four indicators**, built as the deposit's are (rises over the median of the first 14 recorded days; never reset at a
cleaning; clipped 0–1), from measured values only: `dp_pct` (ΔP normalised to the reference flow and 25 °C viscosity, as
`sim_v2.deposit_view`), `flux_pct` (loss of 25 °C specific flux = flux / (NDP × TCF)), `sp_pct` (rise of 25 °C salt passage),
`ndp_pct` (rise of net driving pressure). NDP from the measured pressures and the osmotic pressures of the measured TDS
(`chemai.features.ro_physics`).

## Noise and gaps

- **As the base simulator already makes them:** flows ±0.5 %; feed TDS ±1 %; permeate TDS ±2 %; feed temperature ±0.2 °C;
  permeate pressure ±0.02 bar; feed and concentrate pressure ±0.03 bar plus a zero drift (random walk, 0.004 bar a day); the ΔP
  transmitter ±0.8–3 % (per train); 5 % of train-days are outliers (noise × 4); trains stop in blocks, and a day is recorded only
  when at least two trains run.
- **SDI and turbidity:** phase 1B's construction, L + r·e with L = √s·Z + √(1 − s)·V (s = 0.5, r = 0.5); SDI-like and NTU-like
  scales for readability.
- **The no-link columns — every level ASSUMED, anchored where a public source states one:**
  - feed pH 8.1 (standard seawater, FilmTec manual) + 0.1 × a slow random signal (unit spread, 7-day correlation) ± 0.02 daily;
  - ORP 180 mV + 20 × a slow signal ± 5 mV daily — below the 250 mV alarm in LG Chem's feed guidelines;
  - free chlorine max(0, 0.02 + 0.01 × a slow signal ± 0.005) mg/L — below the 0.1 mg/L limit (FilmTec; LG Chem);
  - set-point doses, each held between steps (a new set point on average every 30–90 days) with ±3 % pump noise: sodium
    bisulfite 2–4 mg/L (LG Chem: 3.0 mg per 1.0 mg of free chlorine); antiscalant 1.5–3 mg/L (a seawater product sheet:
    typically 1–5 ppm); ferric coagulant 1–4 mg/L as Fe (plants in Gaid, Canet & Craig, IDA 2009: up to 4, and below 7);
  - cartridge-filter ΔP from 0.2 bar after a change, rising to the change-out at 0.7 bar (10 psi, the example set point in
    Netsol Water's guide) in 30–90 days; ±0.01 bar.
  - In a real plant some of these carry news about the feed (operators raise the coagulant with raw-water turbidity; cartridges
    clog faster in bad water). This simulator does not model those links, so here they carry none — said on the page.

## The forecast: exactly phase 1's steps

- **Pairs:** (train, day *t*) → the same train on day *t + h*, h = 1, 3, 7, 14; both days recorded; no cleaning in (t, t + h]
  (the table's own cleaning days).
- **Targets:** `dp_pct` and `flux_pct`.
- **Tests:** phase 1's rolling origin — test blocks 550–599, 600–649, 650–699, 700–771, each trained on every earlier pair of its
  plant — and the unseen train (trained on the other two trains).
- **Scores and the bar:** MAE (percentage points) and skill = 1 − MAE / MAE(no change); a method beats *no change* only if its
  pooled skill is at least +5 % and positive in at least 3 of the 4 blocks. Errors are pooled over the three plants; each plant
  is reported too.

## Two records, the same methods

- **Record 1, “like the real record”:** phase 1's inputs — the four indicators today and 1, 3, 7 and 14 days back, the 14-day
  slopes, days since the last cleaning, the smoothed state and the fouling-law forecast. (The count of 10-min samples is not
  simulated and is left out.)
- **Record 2, “everything logged”:** record 1 plus every other logged column — today, its 7-day mean, and 1, 3, 7 and 14 days
  back (the nearest earlier day within 2 days).
- **Methods, settings unchanged from phase 1 and addition 1:** *no change* (the rival); the fouling law; XGBoost; the hybrid
  (law + XGBoost on its error); the LSTM; the RNN — 5 seeds each. The networks read record 2's columns day by day in their
  17-day sequence. *No change* and the law use only the indicator itself, the same in both records.

## Checks

- **Negative control:** record 2 with SDI and turbidity shifted in time (circular, a random 60–300 days per plant). XGBoost's skill
  7 and 14 days ahead must fall back to within 2 points of its skill without those two columns. If not, something leaks, and
  nothing is reported until it is found.
- **Which columns the model used** (descriptive): XGBoost on record 2, rolling test, ΔP 7 days ahead — permutation importance
  on the test rows, by column group (causes, operation, results, no link).

## Reported whatever the result

Every method × record × target × horizon × test, pooled and per plant; the control; the column groups' importance.

## In the deck (note 15)

Pages 37–38 (the gate and the failure) leave; their files stay in the repo. Two pages take their place: the table, and the
results.

## Sources

- DuPont, *FilmTec™ Reverse Osmosis Membranes Technical Manual*: free chlorine < 0.1 ppm (section 1.5); standard seawater pH 8.1
  (table 5); feed turbidity < 1 NTU (section 2.5.1).
- LG Chem, *Technical Applications Bulletin 106*, feed water guidelines: free chlorine ≤ 0.1 mg/L; ORP alarm 250 mV, high-high
  300 mV; 3.0 mg SBS per 1.0 mg free chlorine; SDI ≤ 5.
- American Water Chemicals, *Pureflux SW product data sheet* (2022): seawater antiscalant, typically 1–5 ppm.
- Gaid, Canet & Craig, *Comparative Seawater RO Pretreatment Evaluation Using Existing Plant and Pilot Plant*, IDA (2009):
  ferric doses (Gold Coast plant up to 4 mg/L as Fe; Fujairah below 7 mg/L as Fe), seawater pH 7.6–8.3.
- Netsol Water, *When to replace cartridge filters*: replacement at a preset differential, such as 10 psi.
- Phase 1 declaration (28 October 2026); addition 1 declaration (networks); phase 1B declaration (28 October 2026).
