# Experiment 9 — train on all the simulation, test on the whole labelled archive

Written: 2026-10-25, at Mohammed's request, BEFORE any result of this experiment.
Nothing in this file changes after the run.

## Why

The episode trained on 120 simulated plants and kept 120 new plants as a simulated final exam.
Here the two simulated sets are merged for training, and the real archive is the only test.
Then the same sequence as the episode: models → tuning → ensemble → physics-informed → the rule.

## What has been seen before (stated, not hidden)

The 58 labelled real loops (and the 22 of the episode inside them) were tested before: the Week 2
classifier, the five-model plant check and the Week 8 physics run. Plant data also shaped three
choices (frozen-sensor threshold, no regularity gate, shape threshold check). So the archive
numbers below DESCRIBE — they do not prove.

## Data

- **Training:** all simulated plants — `random_state = 42` (120 plants) and `random_state = 2026`
  (120 plants): 240 plants, 480 runs, five conditions. Features exactly as computed by
  `week8_physics.py` (`results_week8_dev_features.csv`, `results_week8_test_features.csv`).
  The runs of one plant always stay together (groups = plant).
- **Test (the archive):** the 58 real loops with a stated label (26 sticky valves), stiction against
  the rest. Also reported: the 22 loops of the episode inside them (14 sticky valves).
- **Unlabelled real loops:** used only by "rule as teacher", with the rule's labels where it is
  clear — as in Week 8. No archive label is ever used for training, tuning or selection.

## Stages, in this order

- **A · Models, default settings.** Logistic regression, SVM (RBF), decision tree, random forest
  (300 trees), gradient boosting (histogram). Balanced class weights. The episode's features:
  Harris (true delay in simulation), regularity, shape, shape-missing flag, cycles.
- **B · The same models, tuned.** Grid search on the training data only, GroupKFold(4) by plant,
  objective = stiction F1 (stiction against the rest). Grids:
  LR C ∈ {0.01, 0.1, 1, 10, 100}; SVM C ∈ {0.3, 1, 3, 10, 30} × gamma ∈ {scale, 0.03, 0.1, 0.3};
  tree depth ∈ {2, 3, 4, 6, 8, none} × min leaf ∈ {1, 3, 5, 10};
  forest depth ∈ {none, 4, 8} × min leaf ∈ {1, 3, 5} × max features ∈ {sqrt, all};
  boosting rate ∈ {0.03, 0.1} × depth ∈ {2, 3, none} × iterations ∈ {100, 300} × L2 ∈ {0, 1}.
- **C · Ensembles of the five tuned models.** Soft voting (equal weights), and stacking (a logistic
  regression on the five models' group-out-of-fold probabilities). Both fixed now.
- **D · Physics-informed.** The same five families, tuned the same way, with the Week 8 physics
  features: Harris computed with the plant delay (2 samples), S read from the trend, the wait
  fraction, a missing flag. Then: their soft-voting ensemble; the hybrid (the rule where it is
  clear, the physics-informed model with the best simulation score elsewhere); the rule as teacher
  (a forest with the tuned physics-informed forest's settings, trained on the simulation plus the
  rule-labelled unlabelled real loops).
- **E · The rule.** Regularity > 1 and shape > 0.6. Nothing is trained.

## Reported for every method

- **Simulation score:** nested cross-validation on the 480 runs — outer GroupKFold(5) by plant,
  everything (tuning, ensembles) redone inside each outer training part; stiction F1 out of fold.
  (Hybrid: the rule's clear zones plus the selected model's out-of-fold predictions. Rule as
  teacher: not computed — its pseudo-labels are always in training.)
- **Archive (58):** caught of 26, false alarms, precision, recall, F1, and a 95 % interval for F1 by
  resampling the 58 loops (2000 times). **Archive (22):** caught of 14, false alarms, F1.
- All methods are reported, not only the best. "Best" inside a stage is chosen by the simulation
  score only.

## Reading rule (fixed now, the same bar as before)

A method clearly beats the rule on the archive only if, on the 58 loops, caught − false alarms
exceeds the rule's by MORE than two loops, and on the 22 loops it is not worse by more than one.
With 26 sticky valves, a difference of one or two loops is within noise.

## Addendum 1 — a neural network (added 2026-10-25 after the main run, at Mohammed's request)

Written before the network is run. Nothing above changes.

- **Method:** a feed-forward neural network (multilayer perceptron, scikit-learn), standardised
  inputs, Adam, up to 1500 iterations, random_state 0.
- **Tuned like stage B:** GroupKFold(4) by plant, objective stiction F1. Grid: hidden layers ∈
  {(16), (32), (32, 16), (64, 32)} × L2 penalty ∈ {0.0001, 0.001, 0.01, 0.1}.
- **Two versions:** the episode's features (as stage B) and the physics features (as stage D).
- **Same reporting, same bar.** Also: the final settings refitted with seeds 0–4, to show how much
  the archive result depends on the seed.
- It is added after the main results were seen, so it is one more family checked against the same
  bar — not a new chance to win.
