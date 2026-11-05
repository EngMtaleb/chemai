# Phase 1B, proof of concept — report

Run 3–4 November 2026 as declared in `declaration_poc.md` (approved 3 November 2026, 22:40 UTC+3; SHA-256 `3986ddbd8924fc542192b1f1cd2ad2341886eae739add657cac7e75afafb46cb`). **SIMULATION · PROOF OF CONCEPT:** every number below describes this simulator under the declared assumptions, not any real plant.

## In one paragraph

Three simulated plants (one per fouling law; 3,973 train-days, 25 logged columns with noise) ran phase 1's steps twice: on record 1 (the four indicators, as the real record has them) and on record 2 (every logged column). ΔP seven days ahead, rolling test, three plants pooled: *no change* 5.78 points; XGBoost 5.38 on record 1 and 4.89 on record 2. Over all 16 cells (ΔP and flux × 1, 3, 7, 14 days × both tests) the methods passed phase 1's bar this many times, record 1 → record 2: XGBoost 4 → 9; hybrid 0 → 2; LSTM 2 → 9; RNN 3 → 12. XGBoost passed even on record 1 here — on the real plant no method did for ΔP: the simulator is easier than the plant. Permutation importance: XGBoost leaned mostly on the results group; columns with no link moved its error about as much as the causes. The declared negative control missed its bar in one cell of four (flux 14 days ahead, +2.64 points; bar: within 2). As declared, nothing was reported until the cause was found: there is no leak — SDI and turbidity carry a yearly cycle, which a circular shift moves but keeps; with the cycle taken out before the same shift, the gap is +0.25 to +1.17 points over three sets of seeds (post hoc).

## What ran

- The table: `make_table.py` → `data/table.csv` (SHA-256 `6dcb3b7734916a48b1cd8b4a39f936ce91e09c67be43e884e6e6af0930173eec`) and `data/cleanings.csv` (`f0fbcfe3ecfa0ddee6d3bd0852720551d6b72b34acbe085fff67d7286f9baef2`), made once before any model ran; hidden quantities never enter a model (`poc_lib.load` checks the column list).
- Trees: `poc_trees.py` — no change, the fouling law, XGBoost and the hybrid (phase 1's `p3lib`, unchanged), both records, both tests; the negative control; permutation importance.
- Networks: `poc_nets.py` — addition 1's LSTM and RNN (`p3add`, unchanged; five seeds), both records, every fold. Only the input sizes follow the record (no count of 10-min samples; record 2 carries every logged column in the 17-day sequence).
- Scores: `make_poc_data.py` → `out/poc_data.js` (the deck's numbers), `out/report_tables.md`, `out/scores_*.csv`.

## Results

### Scores, pooled over the three plants

#### rolling · dp_pct · MAE in points (skill against no change, %; ✓ = passes the bar)

| method | record | 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|
| no change | both | 3.81 (+0.0) | 4.83 (+0.0) | 5.78 (+0.0) | 6.19 (+0.0) |
| fouling law | both | 4.16 (-9.0) | 4.73 (+2.0) | 5.71 (+1.2) | 6.64 (-7.2) |
| XGBoost | record 1 | 3.76 (+1.4) | 4.55 (+5.6 ✓) | 5.38 (+7.0 ✓) | 6.17 (+0.4) |
| XGBoost | record 2 | 3.83 (-0.3) | 4.57 (+5.4 ✓) | 4.89 (+15.5 ✓) | 5.68 (+8.3 ✓) |
| hybrid | record 1 | 3.87 (-1.4) | 4.69 (+2.8) | 5.50 (+4.8) | 6.31 (-1.9) |
| hybrid | record 2 | 3.83 (-0.3) | 4.66 (+3.5) | 5.14 (+11.1 ✓) | 5.90 (+4.7) |
| LSTM | record 1 | 3.56 (+6.6 ✓) | 4.45 (+7.7 ✓) | 5.57 (+3.6) | 5.96 (+3.8) |
| LSTM | record 2 | 3.75 (+1.6) | 4.47 (+7.5 ✓) | 5.03 (+13.1 ✓) | 5.74 (+7.2 ✓) |
| RNN | record 1 | 3.54 (+7.3 ✓) | 4.38 (+9.2 ✓) | 5.41 (+6.4 ✓) | 5.86 (+5.3) |
| RNN | record 2 | 3.80 (+0.5) | 4.43 (+8.2 ✓) | 4.92 (+14.9 ✓) | 5.63 (+9.0 ✓) |

#### rolling · flux_pct · MAE in points (skill against no change, %; ✓ = passes the bar)

| method | record | 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|
| no change | both | 3.17 (+0.0) | 4.07 (+0.0) | 5.28 (+0.0) | 4.80 (+0.0) |
| fouling law | both | 3.90 (-23.0) | 4.52 (-11.0) | 5.66 (-7.3) | 6.38 (-33.1) |
| XGBoost | record 1 | 3.25 (-2.5) | 4.05 (+0.4) | 4.68 (+11.3 ✓) | 4.72 (+1.5) |
| XGBoost | record 2 | 3.06 (+3.5) | 3.79 (+6.8 ✓) | 4.37 (+17.2 ✓) | 4.23 (+11.8 ✓) |
| hybrid | record 1 | 3.61 (-14.0) | 4.34 (-6.8) | 5.13 (+2.8) | 5.52 (-15.2) |
| hybrid | record 2 | 3.58 (-13.0) | 4.21 (-3.5) | 5.18 (+1.8) | 5.51 (-14.9) |
| LSTM | record 1 | 3.11 (+1.9) | 4.19 (-2.9) | 5.66 (-7.2) | 4.86 (-1.3) |
| LSTM | record 2 | 3.12 (+1.7) | 3.79 (+6.7 ✓) | 4.64 (+12.0 ✓) | 4.37 (+8.9 ✓) |
| RNN | record 1 | 3.04 (+4.1) | 3.99 (+2.0) | 5.12 (+2.9) | 4.96 (-3.5) |
| RNN | record 2 | 3.10 (+2.2) | 3.74 (+8.1 ✓) | 4.53 (+14.1 ✓) | 4.28 (+10.8 ✓) |

#### unseen_train · dp_pct · MAE in points (skill against no change, %; ✓ = passes the bar)

| method | record | 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|
| no change | both | 3.81 (+0.0) | 4.83 (+0.0) | 5.78 (+0.0) | 6.19 (+0.0) |
| fouling law | both | 4.16 (-9.0) | 4.73 (+2.0) | 5.71 (+1.2) | 6.64 (-7.2) |
| XGBoost | record 1 | 4.26 (-11.6) | 4.91 (-1.8) | 5.96 (-3.0) | 6.64 (-7.3) |
| XGBoost | record 2 | 4.21 (-10.3) | 4.63 (+4.0) | 5.33 (+7.9 ✓) | 6.03 (+2.6) |
| hybrid | record 1 | 4.27 (-12.0) | 5.10 (-5.7) | 6.00 (-3.8) | 6.73 (-8.7) |
| hybrid | record 2 | 4.08 (-6.9) | 4.81 (+0.3) | 5.42 (+6.2 ✓) | 6.23 (-0.6) |
| LSTM | record 1 | 3.80 (+0.4) | 4.70 (+2.6) | 5.89 (-1.8) | 6.31 (-1.8) |
| LSTM | record 2 | 3.82 (-0.1) | 4.60 (+4.6) | 5.21 (+10.0 ✓) | 5.97 (+3.6) |
| RNN | record 1 | 3.76 (+1.6) | 4.62 (+4.3) | 5.87 (-1.6) | 6.26 (-1.1) |
| RNN | record 2 | 3.85 (-1.0) | 4.47 (+7.3 ✓) | 5.06 (+12.4 ✓) | 5.75 (+7.1 ✓) |

#### unseen_train · flux_pct · MAE in points (skill against no change, %; ✓ = passes the bar)

| method | record | 1 d | 3 d | 7 d | 14 d |
|---|---|---|---|---|---|
| no change | both | 3.17 (+0.0) | 4.07 (+0.0) | 5.28 (+0.0) | 4.80 (+0.0) |
| fouling law | both | 3.90 (-23.0) | 4.52 (-11.0) | 5.66 (-7.3) | 6.38 (-33.1) |
| XGBoost | record 1 | 3.35 (-5.7) | 4.11 (-0.9) | 4.63 (+12.2 ✓) | 4.53 (+5.5) |
| XGBoost | record 2 | 3.15 (+0.6) | 3.91 (+4.0) | 4.48 (+15.1 ✓) | 4.39 (+8.5 ✓) |
| hybrid | record 1 | 3.83 (-20.9) | 4.37 (-7.5) | 5.20 (+1.4) | 5.39 (-12.4) |
| hybrid | record 2 | 3.77 (-18.9) | 4.33 (-6.5) | 5.31 (-0.7) | 5.66 (-18.0) |
| LSTM | record 1 | 3.15 (+0.4) | 4.14 (-1.8) | 5.31 (-0.6) | 4.83 (-0.6) |
| LSTM | record 2 | 3.09 (+2.5) | 3.86 (+5.0) | 4.56 (+13.5 ✓) | 4.50 (+6.2 ✓) |
| RNN | record 1 | 3.12 (+1.7) | 3.99 (+1.9) | 5.14 (+2.6) | 4.66 (+2.8) |
| RNN | record 2 | 3.08 (+2.7) | 3.74 (+8.1 ✓) | 4.69 (+11.2 ✓) | 4.32 (+10.0 ✓) |

### Each plant (rolling test, ΔP, 7 days): skill against no change, %

| plant | method | record 1 | record 2 |
|---|---|---|---|
| P1 | XGBoost | +3.8 | +20.2 ✓ |
| P1 | hybrid | +2.4 | +13.7 ✓ |
| P1 | LSTM | -2.1 | +26.2 ✓ |
| P1 | RNN | +1.7 | +26.4 ✓ |
| P2 | XGBoost | +6.5 ✓ | +15.2 ✓ |
| P2 | hybrid | +0.4 | +12.8 ✓ |
| P2 | LSTM | +0.7 | +11.1 ✓ |
| P2 | RNN | +3.3 | +10.8 ✓ |
| P3 | XGBoost | +8.9 ✓ | +13.7 ✓ |
| P3 | hybrid | +10.2 ✓ | +8.2 ✓ |
| P3 | LSTM | +9.1 ✓ | +9.1 ✓ |
| P3 | RNN | +11.4 ✓ | +13.6 ✓ |

## The negative control

Declared: record 2 with SDI and turbidity shifted in time (circular, 60–300 days per plant); XGBoost's skill 7 and 14 days ahead must fall back to within 2 points of its skill without those two columns; if not, nothing is reported until the cause is found.

Skill against *no change*, %, rolling test, three plants pooled (shifts: P1 299 days, P2 236 days, P3 170 days):

| target · h | as run | without SDI and turbidity | shifted | shifted − without | as run − without |
|---|---|---|---|---|---|
| dp_pct · 7 d | +15.51 | +13.70 | +13.95 | **+0.25** | +1.80 |
| dp_pct · 14 d | +8.28 | +6.35 | +5.80 | **-0.55** | +1.92 |
| flux_pct · 7 d | +17.21 | +17.69 | +18.71 | **+1.02** | -0.47 |
| flux_pct · 14 d | +11.79 | +10.41 | +13.05 | **+2.64** | +1.38 |

**Verdict as declared: missed** — in one cell of four: flux 14 days ahead, +2.64 points (bar: within 2). For ΔP the shift took the columns' gain away, as it should.

### The investigation

1. **The code** (`ctrl_check1.py`): the shifted build is identical to the as-run build in every column except the twelve SDI and turbidity inputs, in all three plants; the shift only moves values in time. No target, no hidden quantity and no future value of the plant's own record enters through it.
2. **The seeds** (`ctrl_check2.py`): the missed cell re-run with the declared seeds (0–4; it reproduces the run's +2.64: yes) and with two other sets of five seeds. Nothing else changes.
3. **The yearly cycle** (`ctrl_check3.py`): SDI and turbidity carry a yearly cycle (on the log scale, 15–27 % of their variance), and a circular shift moves that cycle but keeps it. The same cell with the cycle taken out before the same shift.

Flux 14 days ahead, XGBoost's skill, %:

| seeds   |   shifted |   without |   shifted, no yearly cycle |   shifted − without |   shifted, no yearly cycle − without |
|:--------|----------:|----------:|---------------------------:|--------------------:|-------------------------------------:|
| 0-4     |     13.05 |     10.41 |                      11.33 |                2.64 |                                 0.92 |
| 5-9     |     13.6  |     11.08 |                      12.25 |                2.52 |                                 1.17 |
| 10-14   |     12.66 |     11.18 |                      11.43 |                1.48 |                                 0.25 |

**Finding.** The code shows no leak (step 1). The gap is not the seeds' doing: it stays at +1.48 to +2.64 points over three sets of five seeds (step 2). It is the yearly cycle: with the cycle taken out before the same shift, the gap is +0.25 to +1.17 — within the bar for every set (step 3). The shifted columns still told the model the time of year, which helps flux two weeks ahead; the declared control should have taken the cycle out before shifting. The declared verdict stands as written (missed, one cell); steps 2 and 3 were designed after the miss and are reported as such. For ΔP, the cells the deck shows, the shift took the gain away as it should. Nothing in the results changes: the shifted columns enter no reported forecast.

## Which columns XGBoost used

| group     |   increase_pp |   base_pp |   increase_pct |
|:----------|--------------:|----------:|---------------:|
| results   |        0.8942 |    4.8854 |        18.3033 |
| operation |        0.1578 |    4.8854 |         3.2298 |
| causes    |        0.1535 |    4.8854 |         3.1417 |
| no link   |        0.1215 |    4.8854 |         2.4865 |

Top columns (descriptive):

| column            |   increase_pp |   base_pp |   increase_pct |
|:------------------|--------------:|----------:|---------------:|
| dp_bar            |        0.1461 |    4.8854 |         2.9902 |
| flux_lmh          |        0.1061 |    4.8854 |         2.1726 |
| recovery          |        0.0666 |    4.8854 |         1.3639 |
| cf_dp_bar         |        0.0595 |    4.8854 |         1.2178 |
| orp_mv            |        0.0352 |    4.8854 |         0.7205 |
| free_chlorine_mgl |        0.0346 |    4.8854 |         0.7077 |
| turbidity_ntu     |        0.0325 |    4.8854 |         0.666  |
| feed_flow_m3h     |        0.0171 |    4.8854 |         0.3497 |
| perm_tds_mgl      |        0.0128 |    4.8854 |         0.2617 |
| perm_pressure_bar |        0.0101 |    4.8854 |         0.2062 |

Read with care: the no-link columns play no part in this simulator, yet permuting them moves the error about as much as the causes — slow signals and set-point steps can line up with fouling over a short record by chance. The results group holds the indicators' own past (record 1's inputs) and the measured pressures and ΔP.

## Notes

- The no-link columns come from `default_rng([seed, 4])` with the plant's seed (4101–4103), as declared; the plant's own draws use their streams, so the plant is draw for draw what version 2 makes without these columns.
- The deck refuses a partial run: `poc_data.js` carries `final`, true only with all three plants and the networks.
- Before anything goes to the repo: the declaration names suppliers and two plants in its sources (a product sheet's maker, a supplier's guide, plants in a published comparison). The deck names only the membrane makers' manuals; the owner decides on the declaration's text.

## Files

- `declaration_poc.md` · SHA-256 `3986ddbd8924fc542192b1f1cd2ad2341886eae739add657cac7e75afafb46cb`
- `make_table.py` · SHA-256 `8b49ff93cae5e690337e41727be7f92e7c06bc0e841af424c7297a7098729d8d`
- `poc_lib.py` · SHA-256 `277a0759766c5159e50232013f7b94445399962853f672cb429e08d298dd4219`
- `poc_trees.py` · SHA-256 `c3f2bd5e0f4ccc5ae4ff8431efcb03ee79272c290205b33d8afd31f85ea8b1e5`
- `poc_nets.py` · SHA-256 `808433844c1b8aee3378491a231bc1b49bca06cac89c64edfdf403b4d737626d`
- `make_poc_data.py` · SHA-256 `33f7d095e2094b034d5cfec376d47145a754afe5a78fde63c3b49b73574909ab`
- `ctrl_check1.py` · SHA-256 `eb83e81bceda30466d1bf524b41232d5d8b987d2adf8644172d33ed8e2fd08ee`
- `ctrl_check2.py` · SHA-256 `e244a53e5180d40e4833b9d13b6a1d0d6384a5b7146a420f61587253ab95ee23`
- `ctrl_check3.py` · SHA-256 `0a12c6b54def2c5c8b5dbf9b34461864c7444abdd261252929fcca52b0e00aba`
- `out/pred_trees.csv` · SHA-256 `de6e355e1d3ba93a7f0cb137075c543852505e17118b2332ec6c229cc10ac091`
- `out/pred_nets.csv` · SHA-256 `c28283abcaa300876b433fda3fdae5246e2bc64d39ee44426003ba6276e3d1f3`
- `out/pred_control.csv` · SHA-256 `6f73935f59175c6508f758aeff18360e2abb70e283054316a09e98e7c6d6c3b0`
- `out/poc_data.js` · SHA-256 `fc72a5f87f88159037aae4080d0d3be86dcb008da40312ce1f4f81fbbc79401e`
