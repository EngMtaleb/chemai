# Experiment 10 — deep learning on the raw signals (LSTM and 1D-CNN)

Written: 2026-10-25, at Mohammed's request, BEFORE any result of this experiment.
Nothing in this file changes after the run.

## Question

Would a network that reads CO and CV directly — instead of the four or eight numbers the project
computes — do better on the real archive?

## Data (as in experiment 9)

- **Training:** all simulated runs, `random_state = 42` and `2026`, the same generator and settings
  as Week 8 (`generate_dataset(8, 2)`): 240 plants, 480 runs, five conditions. Runs of one plant
  stay together in every split.
- **Archive:** the 58 real loops with a stated label (26 sticky valves), stiction against the rest;
  also the 22 loops of the episode. Loaded exactly as in Week 8 (same loaders, same filters).
  These loops were seen before, so the archive numbers describe — they do not prove.
- No archive label is used for training, settings or selection. There is no hyperparameter tuning
  at all: the settings below are fixed now.

## Input (the same rule for simulation and archive)

1. Two channels: CO, and CV − SP. Each is linearly detrended.
2. Time is rescaled so that one oscillation period spans 32 points. The period comes from the
   project's own estimator (autocorrelation of SP − CV). If it is missing or outside
   [4 samples, record length / 3], the record length / 12 is used instead.
   Why: without it, one window covers half a cycle on one loop and fifty on another, and a failure
   would only show a units problem.
3. Windows of 192 points (six periods), each channel z-scored inside its window (a flat channel
   becomes zeros). A record shorter than one window is zero-padded at the start.
4. Training: 12 windows per run, drawn at random (seed 0) from all windows at stride 96.
   Prediction for a record: up to 64 windows, evenly spaced; the class probabilities are averaged.
5. Loop type is NOT an input (project rule: it selects signals, it is never a feature).

## Networks (Keras 3, JAX backend)

- **LSTM:** LSTM(64, sequences) → LSTM(32) → Dense(32, relu) → Dropout(0.2) → Dense(5, softmax).
- **1D-CNN:** Conv1D(32, 7) → MaxPool(2) → Conv1D(64, 5) → MaxPool(2) → Conv1D(64, 3) →
  global average pooling → Dropout(0.2) → Dense(5, softmax). ReLU, "same" padding.
- **Training:** five classes, cross-entropy, Adam (1e-3), batch 64, at most 30 epochs, early
  stopping on validation loss (patience 5, best weights kept). Validation = 15 % of the training
  plants, chosen by plant.

## Reported

- **Simulation score:** GroupKFold(5) by plant, out of fold, one seed (0); stiction F1.
- **Archive:** the final network trained on all 480 runs with seeds 0, 1 and 2; the archive call is
  the mean probability over the three seeds. Each seed's counts are also reported.
  Caught of 26, false alarms, precision, recall, F1 with a 95 % interval (2000 resamples of the 58
  loops); the 22 loops: caught of 14, false alarms.

## Reading rule (the same bar)

A network clearly beats the rule (12 of 26, 2 false alarms on the 58; 12 of 14, 2 on the 22) only
if, on the 58, caught − false alarms exceeds the rule's by MORE than two loops, and on the 22 it is
not worse by more than one.
