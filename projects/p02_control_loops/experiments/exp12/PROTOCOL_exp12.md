# Experiment 12 — more model families, and the networks with the physics

Written: 2026-10-25, at Mohammed's request (round-10 notes), BEFORE any result of this experiment.
Nothing in this file changes after the run.

## Questions

1. Experiment 11 ran one forest, the same forest tuned, and an ensemble of five tuned families. Do the
   other common families — including the boosting libraries XGBoost and LightGBM — do better, with
   default settings and tuned?
2. The networks gained the most from real loops. Does the valve's physics help them the way it was
   tried with the forest: as a **hybrid** (the rule decides where its call is clear, the network
   decides the rest) and with the **rule as teacher** (the 11 rule-labelled unlabelled real loops join
   the training)?
3. For the physics introduction page: how far does **S alone** go on this split?

## Design — identical to experiment 11

- The same 58 labelled real loops (26 sticky valves), the same five parts (`../exp11/folds_exp11.csv`),
  the same 480 simulated runs, the same weighting (real loops of the four training parts together
  weigh as much as the simulation), the same groups for every inner split, the same scorer (stiction
  F1), the same 11 pseudo-labelled loops (`week8_physics.clear_zone` on the unlabelled real loops).
- Two conditions, as before: A = simulation only (teacher variants: simulation + the 11 pseudo loops);
  B = simulation + the real loops of the other four parts. The pages show condition B.
- Every method runs once. All methods are reported, including the ones that do worse.

## Methods

Feature models — the episode's five numbers (`BASE`), five conditions out, scored as stiction or not:

| family | default settings | tuned (GridSearchCV, inner GroupKFold 4, stiction F1) |
|---|---|---|
| logistic regression | C = 1 (scaled inputs) | as experiment 11 |
| SVM | C = 1, gamma "scale" (scaled inputs) | as experiment 11 |
| decision tree | sklearn defaults | as experiment 11 |
| random forest | 300 trees (= experiment 11, method 1) | as experiment 11 (= method 2) |
| gradient boosting (sklearn, histogram) | sklearn defaults | as experiment 11 |
| XGBoost | library defaults (100 trees, depth 6, rate 0.3) | n_estimators 100/300 · max_depth 2/3/6 · learning_rate 0.03/0.1/0.3 |
| LightGBM | library defaults (100 trees, 31 leaves, rate 0.1) | n_estimators 100/300 · num_leaves 7/15/31 · learning_rate 0.03/0.1 · min_child_samples 5/20 |

All families use balanced class weights (XGBoost through balanced sample weights), random_state 0.

- **Ensemble (7)** — soft voting of the seven tuned families. Replaces experiment 11's five-family
  ensemble on the page; the five-family vote is recomputed only as a reproduction check.

Physics with the networks (physics features `PHYS` for the ANN; CO and CV windows for CNN and LSTM,
inputs and settings exactly as experiments 10 and 11):

- **ANN** — as experiment 11, method 7 (reproduction check: it must give the same calls).
- **ANN hybrid** — the rule's clear zone where it is clear, the ANN's call elsewhere.
- **ANN rule as teacher** — a binary ANN with the tuned ANN's settings, trained exactly like the
  experiment-11 forest teacher: the binary training table plus the 11 pseudo-labelled loops.
- **CNN hybrid, LSTM hybrid** — the rule's clear zone where it is clear, the experiment-11 network
  call (mean over seeds 0, 1, 2) elsewhere. Nothing is retrained.
- **CNN rule as teacher, LSTM rule as teacher** — the same networks with a two-class output (sticky or
  not), trained like the forest teacher: simulated windows (weight 1), the windows of all labelled real
  loops of the four training parts (together weighing as much as the simulation), and the windows of
  the 11 pseudo-labelled loops (weight 1, like a simulated window). Early stopping on 15 % of the
  simulated plants; seeds 0, 1, 2; the call is the mean probability.
- **S alone** — S read from the trend (`s_travel`), one threshold chosen on the training side
  (37 quantiles, the one with the best stiction F1 on the training rows), applied to the test part.

## Reported

For every method and both conditions, on all 58 loops: caught of 26, false alarms, precision, recall,
F1 with a 95 % interval (2000 resamples of loops); the default-to-tuned change for each family.

## Reading rule — the same bar, and one warning

A method clearly beats the rule only if caught − false alarms exceeds the rule's (12 − 2 = 10) by MORE
than two loops. This experiment was designed after experiment 11's results were seen, and it adds
more than twenty methods on the same 58 loops: with that many tries, one can clear the bar by luck.
Any clearance here is a hypothesis for new labelled loops, not a replacement for the rule.

## Amendment 1 — before any run (2026-10-25)

Per-loop weight of the pseudo-labelled windows: a simulated run gives 12 training windows, while a
pseudo-labelled loop gives about 38. So that each pseudo-labelled loop weighs as much as one simulated
run — exactly as in the forest teacher, where each pseudo-labelled loop is one row — its windows get the
weight 12 ÷ (its number of windows), not 1. Nothing had been run when this was written.
