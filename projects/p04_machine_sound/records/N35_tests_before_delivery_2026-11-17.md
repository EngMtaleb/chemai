# Notebook 35 · tests before delivery (2026-11-17)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU, with stand-in models.

- **Notebook:** `notebooks/P04_N35_Pooling_First_Shot_DCASE2025_Colab.ipynb` (15 cells, 13 of them code; 1,536 lines of code).
- **Builder:** `notebooks/build_n35_pooling_first_shot.py`. **Rules:** `RULES_pooling_first_shot_2025_written_before_running.md`.
- **The reading, written before any result:** `notebooks/analyse_n35.py` (A0 to A5 of the rules).
- **Tests:** `results/mimii_beats/tests_n35/` (`run35.sh`, `check35.py`), on notebook 30's own fixtures and runner.

## What comes from notebook 30, which ran on the real files
Cells 1 to 9 are notebook 30's, to the byte, except the output folder's name in cell 1 (two checked substitutions): setup,
constants, the BEATs code and both checkpoints with their hashes, LoRA, ECHO, and the data cell with its gates G0(a) and G0(b).
Cells 10 to 13 are new: 301 lines.

## What was tested
| Test | Result |
|---|---|
| Notebook 30's test run first (it makes the six trainings notebook 35 reads), then notebook 35, from nothing | both finished; notebook 35 in 57 s |
| The eleven poolings against a slow reference written from the formulas (runs in the notebook itself) | largest difference 3.7e-07 |
| G0(q): the mean pooling through each trained head against notebook 30's saved embeddings | cosine 1.0 for all six trainings |
| G0(r): candidate `AT|F|mean` against notebook 30's saved score files, development and evaluation | Spearman 1.0, relative difference 0.0, all six types |
| `check35.py`: the trained model's pooled tokens differ from the frozen model's (LoRA was loaded); the two frozen encoders differ; no NaN | passed |
| `check35.py`: scores E0, E1, Ev of three frozen candidates on three types and their α*, recomputed with plain numpy from the pooled files | largest difference 6.4e-08; α* identical |
| The bank: 122 candidates, the same in every type; shapes; every score finite | passed |
| Resume: two finished passes put aside (one frozen, one trained), run again | only those two passes ran; the bank is identical (difference 0) |
| Notebook 30's folder missing | stops in cell 10 with a message naming the folder and a missing file |
| `analyse_n35.py` on the synthetic bank with the fixtures' ground truth | ran; A0's development score equals the notebook's own table |

## What was not tested
- Any GPU, Drive, the real files, the real BEATs models, real clip lengths (the fixtures' clips are 2 s; the pooling code does not depend on the length).
- The run time. The estimate (about 1.5 hours on a T4) rests on notebook 24's measured 6.5 minutes per pass on a T4 and is not a measurement of this notebook.
- Reading about 3,000 arrays from Drive when the bank is built: expected to take a few minutes; not measured.
- Whether `AT|F|mean` reproduces 0.6228 on the real evaluation set: gate G0(r) checks it against notebook 30's saved scores on Colab, and `analyse_n35.py` stops if it does not.
