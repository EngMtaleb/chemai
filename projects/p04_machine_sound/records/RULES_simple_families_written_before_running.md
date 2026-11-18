# Rules · The four simple families on the three public tests (notebooks 31, 32, 33)

Written on 2026-11-16, before any of the three notebooks ran on real files.

## Why
The deck presents five families of detectors. Families 1 to 4 were scored only on four MIMII pumps
(`results/mimii`). Family 5 alone was scored on DCASE 2020, 2022 and 2025. So the deck cannot say what
the simple families reach on those tests. These notebooks measure it.

## What is run
One notebook per test: 31 · DCASE 2020 Task 2, 32 · DCASE 2022 Task 2, 33 · DCASE 2025 Task 2.
The files are the ones notebooks 25, 27, 28 and 30 read (same Zenodo records, same MD5s), read by the
same loaders. A clip is its first 10 s, one channel.

A unit is a machine ID (2020), a section (2022) or a machine type (2025). Each family is fitted on the
unit's normal training clips only (all of them, source and target) and scores that unit's test clips.

| Family | Features | Detector | Setting (as on the pumps) |
|---|---|---|---|
| F1 · a limit on one number | the clip's loudness: mean log-mel over bands and time | distance from the training mean, in training standard deviations | none |
| F2 · distance from the normal cloud | 128 clip statistics (mean and standard deviation of 64 log-mel bands) | StandardScaler, PCA 95 %, T² + SPE over their 99th training percentiles | `Maha(0.95)` |
| F3 · isolation | the same 128 numbers | Isolation Forest | 200 trees, seed 0 |
| F4 · autoencoder | 5 consecutive log-mel frames (320 inputs) | reconstruction error, mean over the clip | MLP 64-64-8-64-64, 50 epochs, batch 512, seed 0, first 90 % of the rows |

Log-mel: 64 bands, n_fft 1,024, hop 512, as `code/p04_mimii_features.py`. The classes are those of
`code/p04_mimii_models.py`, unchanged.

One difference from the pump page, fixed here: on the pump page "loudness alone" was scored as it is
(louder = more anomalous). A detector does not know the direction of a fault, so F1 is the two-sided
distance from the usual loudness. The one-sided reading is reported beside it (`F1_raw`) and is not a family.

## What is not done
No setting is tuned on any DCASE clip. No family is changed after a result is seen. No fusion of the
simple families, and none with family 5. The evaluation labels are read once, after every score file is
written and its SHA-256 recorded.

## The numbers
- 2020: per machine AUC and pAUC (max_fpr 0.1); per type their means; the result is the mean AUC over the six types.
- 2022 and 2025: per section AUC (source), AUC (target), pAUC; the result is the official score, the harmonic mean of every section's three values (2025: the eight evaluation types).
The metric functions are those of notebooks 27, 28 and 30, which the official evaluators reproduced.

## What will be said
For each test, the best simple family's evaluation result is set beside family 5's recorded result
(2020: 0.9747; 2022: 0.7063; 2025: 0.5314, and the expert alone 0.5922) and beside the official baselines.
- If a simple family's result is at or above family 5's on a test, the deck says so with the number, on that test's page and on the conclusion page.
- If it is below, the deck gives the difference.
Either way all four results are published for all three tests, including any that is below 0.5.

## Limits known before running
- F4 here is the pump runs' scikit-learn autoencoder, not the official DCASE baseline code; the official baselines' published scores are shown beside it, not replaced by it.
- One seed per family (seed 0), as on the pumps.
