# Experiments 9–12 — after Week 8

Four experiments, each run once under a protocol written before its run. They use the same 58
labelled real loops as Week 8 (26 sticky valves, the episode's 22 inside them), which had been seen
before — so their numbers describe, they do not prove. Results and verdict: `../VALIDATION.md` §24.

| | Question | Protocol |
|---|---|---|
| 9 | Train on all the simulation (480 runs), test once on the 58 loops: models → tuning → ensembles → physics → the rule. Addendum: a neural network. | `exp9/PROTOCOL_exp9.md` |
| 10 | Networks that read CO and CV directly (1D-CNN, LSTM), trained on simulation. | `exp10/PROTOCOL_exp10.md` |
| 11 | Some real loops join the training: five parts, every loop tested once by models that never saw it or its valve. | `exp11/PROTOCOL_exp11.md` |
| 12 | Seven model families (XGBoost and LightGBM among them), default and tuned; an ensemble of the seven; S alone; the networks with the physics (hybrid, rule as teacher). | `exp12/PROTOCOL_exp12.md` |

One check was made after the experiment-12 results were seen; it is labelled as such and is not a
try against the bar: `exp12/POSTHOC_exp12.md`. `exp12/paired_exp12.py` (also after the runs) says
how far one score difference can be trusted with 58 loops; it changes no result.

## Run

From the repository root, after `week8_physics.py` (it writes the feature tables these read), with
the archives in `data/` (see the project README). In this order:

```bash
E=projects/p02_control_loops/experiments
python $E/exp9/run_exp9.py && python $E/exp9/run_exp9_ann.py && python $E/exp9/check_exp9.py
python $E/exp10/prep_exp10.py && python $E/exp10/run_exp10.py && python $E/exp10/check_exp10.py
python $E/exp11/folds_exp11.py && python $E/exp11/run_exp11_features.py && python $E/exp11/run_exp11_dl.py
python $E/exp12/prep_exp12.py && python $E/exp12/run_exp12_features.py && python $E/exp12/run_exp12_dl.py
python $E/exp12/run_exp12_posthoc.py && python $E/exp12/paired_exp12.py
```

`EXP10_SMOKE=1`, `EXP11_SMOKE=1` and `EXP12_SMOKE=1` run a tiny version as a code check (one epoch,
or tiny grids), and nothing is saved. The run scripts hide the archive numbers in this mode; the post
hoc check, written after the results were seen, prints its one-epoch numbers.

Beyond the package: `xgboost` and `lightgbm` (experiment 12), and `keras` with the `jax` backend
(the networks). The runs used Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0,
Keras 3.15.1 and JAX 0.12.0.1.

## What is committed

The protocols, the code and the run logs (`*.log`, `exp12/paired_exp12.txt`). The inputs (`*.npz`,
`*.csv`) and the results (`*.json`) are made by the scripts and are not committed, like the archives
themselves.
