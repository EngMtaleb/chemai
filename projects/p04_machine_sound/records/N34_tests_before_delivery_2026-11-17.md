# Notebook 34 · tests before delivery (2026-11-17)

Notebook: `notebooks/P04_N34_Array_Simple_Families_Colab.ipynb`, built by `notebooks/build_n34_array_simple_families.py`.
Rules: `RULES_array_simple_families_written_before_running.md`. It has **not** run on real MIMII files (my environment cannot reach Zenodo).

All tests ran on synthetic data: `nbtest_s34/make_synth.py` writes three zips shaped like MIMII's (2 pumps, 16 normal and
4 abnormal clips each, 8 channels, 2 s, a source with known per-microphone delays, noise at three levels). `nbtest_s34/run.sh`.

| Test | Result |
|---|---|
| Runs end to end, all cells | no error |
| Array physics smoke test (notebook 11's own) | delays recovered within 0.05 sample; passes |
| Delays measured from the synthetic training clips vs the delays written into the files | largest difference 0.04 sample |
| One microphone at 6 dB vs the project's own pump code (`code/p04_mimii_features.py` + `code/p04_mimii_models.py`) on the same zip | scores of F2, F3, F4 and loudness identical (largest difference 0.0), same clips in the same order |
| Resume: two finished units and two local log-mel files removed, run again | finished levels skipped; 2 of 18 networks retrained; results, answers and delays identical to the first run |

Not tested here: the real files, the run time, and gates G0a / G0b (they compare with records of real runs, so they can only be
checked on Colab; the notebook stops if they fail).

**Run time, estimated, not measured:** sound and families 1–3 about 1.5 h; the autoencoder 36 networks × 7.9 min (the average
notebook 31 measured) ≈ 5 h. My estimate for notebooks 31–33 was too low by a factor of 2–3, so treat this as a rough figure.
