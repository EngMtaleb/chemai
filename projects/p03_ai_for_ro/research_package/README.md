# The research package — phases 2B and 2C

The deck's phases 2B (the whole plant, modelled) and 2C (the whole plant, run by the sun) rest on this package, and its source
register names the files each page uses, with their SHA-256. The files are kept at the paths the deck names, unchanged, so each
SHA-256 can be checked here.

| Folder | What it is | Deck pages |
|---|---|---|
| `physics/` | The whole-plant model: solution–diffusion with concentration polarisation, a segmented mass and salt balance along seven 8-inch elements, one-point calibration to the manufacturer's datasheet, and the study runner (cases, operating grid, sensitivities, numerical checks). The original design: 3 trains × 2 vessels × 7 elements. | 2B, and the code pages |
| `energy/` | The control study on that design: a fixed schedule, a daylight-and-tank rule and a rolling MILP optimisation, in four 72-hour windows of 2023 and their sensitivity cases (33 runs), with water and energy accounted separately. | 2C (its configuration) |
| `bigplant/big/` | The design the deck shows: **12 trains × 110 vessels × 7 elements** (132,000 m³/day), with the number of trains as an input. `CHANGES_vs_original.diff` is the whole change to the package's code; at 3 trains it reproduces the original runs exactly. Its 33 runs are in `energy/results/`; a repeat of the MILP runs and a run with the tie-breaks taken literally are kept as summaries. | 2B, 2C |
| `industrial_data/` | A reanalysis of the public plant record: causal, past-only features and a whole-train holdout. Its Ridge and gradient-boosting candidates did not pass its screen; persistence had the lowest validation error in all eight tasks. | the code page on past-only features |

## Run it

Python 3.11 or later, with `numpy`, `pandas`, `scipy` and `matplotlib` (versions in `energy/requirements.txt`). No network is
needed except for `industrial_data/`.

```bash
python physics/run_study.py                 # the small design: cases, operating grid, checks
python energy/run_study.py                  # its 33 control runs (writes energy/results/)
python bigplant/big/physics/run_study.py    # the 12-train design
python bigplant/big/energy/run_study.py     # its 33 control runs
```

The MILP calls stop at 3 seconds each: a repeat run moved grid import by up to 4 % and changed no pass or fail (see
`bigplant/big/energy/results_repeat_milp_summary.json`).

`industrial_data/run_study.py` reads the public deposit: download the record's ZIP
(`ESWRT_reproducibility_deposit_v1.3-20260630.zip`) from [zenodo.org/records/21055948](https://zenodo.org/records/21055948),
unzip it into `industrial_data/zenodo_deposit/`, and run the script; it checks every file against the deposit's own
`SHA256SUMS.txt`. Its `DATA5.json` is the deck's own copy of the record, which it compares with the deposit.

## Data

- **Photovoltaic output:** JRC PVGIS 5.3, PVGIS-ERA5, 2023, at 29.3759 N, 47.9774 E (a fixed 30° south-facing array, 14 %
  losses; the 4.5 kWp series is scaled to each design). The request and the raw file's SHA-256 are recorded in
  `energy/data/solar_source.json`; `energy/data/solar_hourly.csv`, made from that file, is the hourly series the runs read. Modelled from weather —
  not metered at a real installation.
- **The membrane:** the FilmTec™ SW30HRLE-440i product data sheet (Form No. 45-D00965-en): one calibration point and the
  element's absolute limits.
- **The plant record** (`industrial_data/` only): Su, Meng & Chang (2026), *Membranes* 16(8) 257; data
  doi:10.5281/zenodo.21055948, CC BY 4.0.

## Its limits, as the package states them

- The modelled plant and the plant behind the public record are different systems. The record's `dp_pct` and `flux_pct` are
  derived scores; they are not converted into pressures or into the model's degradation factors.
- The plant's duty is imposed; the solver finds the pressure and checks the limits. Fitting one manufacturer's point is
  calibration, not field validation. 500 mg/L is a study limit for total salt, not a drinking-water standard; no boron,
  pathogens or ion-specific scaling are modelled.
- The panels' output is modelled from weather. Demand is known and constant; the dispatch sees only the previous day's solar
  forecast. The MILP and the rules are not learned AI. Four 72-hour windows do not establish annual savings, finances or
  off-grid reliability.

*FilmTec™ and SW30HRLE are trademarks of affiliates of DuPont de Nemours, Inc., named only to identify the public documents
used.*
