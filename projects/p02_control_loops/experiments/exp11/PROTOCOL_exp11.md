# Experiment 11 — some real loops join the training

Written: 2026-10-25, at Mohammed's request, BEFORE any result of this experiment.
Nothing in this file changes after the run.

## Question

If part of the labelled archive is added to the simulation for training, and the rest is used for
testing, what happens to each family — ML, tuning, ensemble, physics-informed, ANN, CNN, LSTM?

## Design: every real loop tested once, by a model that never saw it or its valve

- **One split would be too small** (about 29 test loops, 13 sticky valves). Instead the 58 labelled
  loops are split into 5 parts. Each part is tested by models trained on the simulation plus the
  other 4 parts. Every loop gets exactly one prediction, from a model that did not see it.
- **Recordings that share a valve or a plant stay together** (GroupKFold by source). Groups:
  the 8 oil-gas stiction recordings (DB-1…7 and the full record) · the 12 chemical DB tuning
  recordings · the 7 Horch (2003) loops (one mill) · ISDB buildings · ISDB chemicals 1–12 ·
  ISDB chemicals 71–76 · ISDB pulp & paper · ISDB metals · and three single loops (chemical
  stiction 2018, power 2018, Thornhill 2003). 11 groups, 58 loops, 26 sticky valves.
- The fold assignment is computed once and saved; every method uses the same folds.

## Training data per fold

- **Simulation:** all 480 runs (240 plants), five classes, as in experiments 9 and 10.
- **Real loops of the other 4 parts,** with their expert label mapped onto the simulated classes:
  stiction, tuning_sluggish, tuning_tight, external_oscillation, healthy; "no_oscillation" →
  healthy. "no_stiction" and "tuning" have no simulated class and are left out of five-class
  training (they are still tested). The rule-as-teacher forest, which is binary, uses them all.
- **Weight:** the real loops together weigh as much as the 480 simulated runs (feature models:
  each real loop repeated k = round(480 / number of real training loops) times; networks: sample
  weights). Without this, 40 loops would vanish among 480 runs.
- **Groups for any inner split:** simulated plant, and the real source group above.

## Methods (each run twice: simulation only, and simulation + real loops)

1. ML — random forest, default settings (the episode's features).
2. Tuning — random forest tuned (grid and objective as experiment 9).
3. Ensemble — soft voting of the five tuned families (LR, SVM, tree, forest, boosting).
4. Physics-informed — random forest tuned on the physics features.
5. Hybrid — the rule where it is clear, method 4 elsewhere.
6. Rule as teacher — binary forest with method 4's settings; training data plus the 11
   rule-labelled unlabelled real loops.
7. ANN — neural network tuned on the physics features (grid as experiment 9, addendum 1).
8. CNN and 9. LSTM — on CO and CV, inputs and settings exactly as experiment 10; the archive
   call is the mean over seeds 0, 1, 2.
10. The rule — nothing is trained; it is the reference (12 of 26, 2 false alarms).

## Reported

For every method and both conditions, on all 58 loops: caught of 26, false alarms, precision,
recall, F1 with a 95 % interval (2000 resamples of loops), and the change from simulation-only to
simulation + real. All methods are reported.

## Reading rule (the same bar)

A method clearly beats the rule only if caught − false alarms on the 58 exceeds the rule's by MORE
than two loops. Archive loops were seen before; now their labels are also used for training,
strictly inside the cross-validation. The numbers describe, they do not prove.
