# P04 · Notebook 23 · External judgement. Rules written before running

Written 2026-11-10, after notebook 22b's results (`N22b_outputs_2026-11-10.md`) and before notebook 23 existed. This is the program's last stage ("External judgement", `PROGRAM_full_written_before_running.md`). Each track's final system is frozen: nothing in this notebook changes a system or chooses between systems.

## Questions
1. **T1 · MIMII per noise level.** At 6, 0 and −6 dB, for each machine type and each track, does the frozen system's Split-A AUC reach arXiv 2309.07500's number?
2. **T3 · DCASE 2020 Task 2, evaluation set (MIC only).** MIC's recipe, unchanged, is retrained on DCASE 2020's own training lists and scored once. Is its mean AUC over the four MIMII types at least the rank-1 team's 95.49?
3. **Confirmation:** valves and slide rails are in every table. They never chose anything.
4. **A control (reported; no decision).** Notebook 22b's MIC gain came from fusing a second encoder. Does it come from the second encoder, or from having twice as many models?

## The frozen final systems (notebook 22b, at −6 dB)
| Track | Input | System | Development | Confirmation |
|---|---|---|---|---|
| **ARR** | MWF | notebook 21's fine-tuned BEATs iter3+ (R0 + PD, seed 0), BEAM | 0.9794 | 0.9652 |
| **MIC+** | N2N-CH0 | frozen BEATs iter3 (PretrainedSED's copy), BEAM | 0.8453 | 0.8563 |
| **MIC** | CH0 | the mean of two systems' scores: notebook 21's fine-tuned BEATs iter3+ and notebook 22b's fine-tuned BEATs iter3, each R0 + XC with seeds 0–2, global LDN1 on F with ECHO | 0.8562 | 0.9425 |

- **They are read from Drive:**
  - `P04_StrongFineTune` (notebook 21);
  - `P04_NewEncoders` (notebook 22b);
  - `P04_DenoiseScore` (notebook 20: denoisers, frozen embeddings, ECHO's);
  - `P04_TwinFrontEnds` (notebook 19: MWF weights);
  - `P04_BEATs` (both BEATs checkpoints).
- Nothing in them is retrained. The new trainings below are DCASE 2020's retraining (Part 2) and the control (Part 3).

## Part 1 · MIMII per noise level (all three tracks; T1)
- **Levels:** 6, 0 and −6 dB.
  - MIMII's four 0 dB zips are new here: about 33 GB.
  - Their MD5s are the Zenodo record's, as in notebook 19: pump 488748295c3f60b25de07b58fe75b049, fan 6354d1cc2165c52168f9ef1bcd9c7c52, valve 178478eb0d11c79080a35562bfdeee71, slider 4d674c21474f0646ecd75546db6c0c4e.
- **Each level is evaluated on its own:**
  - The references and the test clips are both at that level.
  - Splits A, B, B+1, B+3 and B+10 are notebook 11's, with the fold rule.
- **Each track's system is applied unchanged at every level:**
  - **ARR:** MWF with notebook 19's saved weights, then notebook 21's PD model and BEAM. These are the −6 dB weights. Notebook 21 already applied them to the 6 dB twins, for PD's teacher.
  - **MIC+:** notebook 20's microphone-0 denoisers, then frozen BEATs iter3 and BEAM. The denoisers were trained on −6 dB inputs.
  - **MIC:** microphone 0; the twelve fine-tuned models (2 encoders × 3 seeds × 2 folds); ECHO computed on that level's microphone 0.
- At −6 dB the systems are recomputed from their files (gate G0(f)).
- **T1** is judged cell by cell: 3 levels × 4 types for each track, Split-A AUC against arXiv 2309.07500, Table 1.

  | Type | 6 dB | 0 dB | −6 dB |
  |---|---|---|---|
  | Pump | 0.9891 | 0.9115 | 0.8977 |
  | Fan | 0.9682 | 0.9260 | 0.8778 |
  | Valve | 0.9978 | 0.9830 | 0.9794 |
  | Slide rail | 0.9974 | 0.9750 | 0.9441 |

  - It is like for like (one microphone) only on MIC. Even there it is indicative, because the paper's split differs from ours.
- **Also reported:** B+10 at each level, and the development and confirmation scores at each level.

## Part 2 · DCASE 2020 Task 2 (MIC only; T3)
### Data (MD5s from the Zenodo record pages, read 2026-11-10)
| Set | Zenodo record | Files used (MD5) |
|---|---|---|
| Development | 3678171 | `dev_data_fan.zip` (649bdfc06263ae7a838963f43b6641e6), `dev_data_pump.zip` (90e7091ef722b7238a7f1009365779cd), `dev_data_slider.zip` (da24a757719f0d94d5aa2d646bbfdc86), `dev_data_valve.zip` (34d2672f55bb041589ef79c10dc89934) |
| Additional training | 3727685 | `eval_data_train_{type}.zip`, as in notebooks 21–22b |
| Evaluation | 3841772 | `eval_data_test_fan.zip` (1eb9356a768cadfd0f2e59a5c57e578b), `eval_data_test_pump.zip` (23a8f8f924be218c69df67fe07360348), `eval_data_test_slider.zip` (0193c769073840332ce3aad84b1ccaa2), `eval_data_test_valve.zip` (f69c551d2088691050d20cccca9d631c) |
| Ground truth | 3951620 | `eval_data_list.csv` (f16aaa185df9906e588864aaf7cb58ea) |

- **Only the four MIMII types are used.** ToyCar and ToyConveyor are not.
- **Machines:**
  - Development: IDs 00, 02, 04 and 06, each with training and test clips.
  - Evaluation: IDs 01, 03 and 05. Their training clips are the additional training dataset, and their test clips are the evaluation set.

### Retraining: MIC's recipe, unchanged, on DCASE 2020's own lists
- **Classes and clips:**
  - One class per development machine, with all of its `train/` clips.
  - One class per evaluation machine, with its first 600 additional-training clips by name. That is exactly XC's cap.
- **No test clip of either set is used for training,** as DCASE 2020's rules require.
- **One model per seed, no folds,** because every DCASE machine has its own training clips.
  - Seeds 0, 1 and 2 for each encoder (BEATs iter3+ and BEATs iter3): six trainings.
  - Settings: notebook 21's, unchanged.
- **ECHO:** frozen, run on the DCASE clips.

### Scoring
- **Per machine:**
  - References: all of its normal training clips. For an evaluation machine that is all of its additional-training clips, about 1,000.
  - Tests: its test clips.
- **The scorer is MIC's.** For each encoder:
  - global LDN1 on F;
  - the three seeds unit-normalised and concatenated (÷√3);
  - ECHO concatenated (÷√2).

  The system's score is the mean of the two encoders' scores.
- **Metrics:** exactly as the official evaluator computes them (github.com/y-kawagu/dcase2020_task2_evaluator, `evaluator.py`, read 2026-11-10):
  - AUC per machine: `sklearn.metrics.roc_auc_score`.
  - pAUC per machine: `roc_auc_score(..., max_fpr=0.1)`, the standardised partial AUC.
  - Per type: the arithmetic mean over its machines.
  - Labels: the third column of `eval_data_list.csv`, matched by file name.

### Order, and the one-time scoring
1. **The six trainings, then the development set.** Its AUC and pAUC are reported per machine and per type.
2. **Gate G-DEV, before the evaluation set is touched:**
   - Every type's development AUC must be at least the DCASE 2020 baseline's: fan 0.6583, pump 0.7289, slide rail 0.8476, valve 0.6628. These are the means of its 10 trials, from the task page.
   - A failure means a broken pipeline. The run then stops, and the evaluation set is not scored.
3. **The evaluation set is scored once:**
   - The ground truth is downloaded and read only in this step.
   - The per-clip scores, in the official CSV format (`anomaly_score_<type>_id_<xx>.csv`), and the results are saved to Drive at once.
   - A later run reads them and scores nothing again.
   - If an error is found after this step, any second scoring is reported as a second scoring.

### T3
- **T3:** the mean AUC over the four types is at least 0.9549.
  - That is Giri_Amazon_task2_2, rank 1: fan 94.54, pump 93.65, slide rail 97.63, valve 96.13 (the program's table).
  - pAUC is reported alongside: Giri 84.30, 81.73, 89.73, 90.89; mean 86.66.
- **Per type,** against the five teams of the program's table.
- **What T3 is and is not:**
  - The task page says the AUC averaged over machine types should not be used, because difficulty differs by type.
  - Its official ranking averages AUC and pAUC ranks per type, then over all six types.
  - So T3 is the program's summary, not an official rank. The per-type table is the fair view.

## Part 3 · The control (reported; no decision)
- **The control system:** notebook 21's MIC system with six seeds. That is BEATs iter3+, R0 + XC, seeds 0–5, global on F with ECHO.
  - Seeds 3, 4 and 5 are trained here, on both folds (6 trainings), with the training code that trained BEATs iter3 in notebook 22b.
  - So the control and the fusion differ only in the second half's encoder.
- **It is compared with the final MIC fusion** (three seeds of each encoder):
  - on MIMII at −6 dB: development, confirmation and per machine;
  - on DCASE 2020's development set, with three more trainings (seeds 3–5 on the DCASE lists).
- It is not run on the evaluation set, which is scored once, by the final system.
- **How it will be read (fixed now):**
  - If the fusion beats six seeds by at least 0.01 on MIMII's development score, the second encoder adds more than the extra models.
  - Otherwise notebook 22b's gain cannot be told apart from ensemble size.
  - The DCASE development set is a second view of the same question.

## Gates
- **G0(a):** the clip lists equal notebook 20's, at all three levels.
- **G0(b):** notebook 20's frozen results and notebook 21's systems, as in notebooks 22 and 22b.
- **G0(c):** frozen BEATs on our streams against notebook 20's embeddings (cosine ≥ 0.999).
- **G0(d):** notebook 21's adapters reproduce its embeddings (cosine ≥ 0.999).
- **G0(f), new:** notebook 22b's final systems, recomputed in this session, give 0.9794 / 0.9652, 0.8453 / 0.8563 and 0.8562 / 0.9425, within 0.0005.
- **G0(g), new: ECHO.**
  - Its weights' SHA-256 equals notebook 15's.
  - Recomputed on the first 32 −6 dB microphone-0 clips of each type, it matches notebook 20's saved embeddings (cosine ≥ 0.999).
  - This ECHO is the one used at 0 and 6 dB and on DCASE.
- **G0(h), new:** the BEATs iter3 file in Drive is the one notebook 22b used (SHA-256 ba0f139a46b3a22f4ff78873db92214146cddfaad12db7c026f6dd758df1be6a).
- **G1:** the training loss falls in every training.
- **G-DEV:** above.
- A failure of any G0 gate, or of G-DEV, stops the run.
- **Side check (reported):** DCASE's development clips are MIMII's microphone 0 at −6 dB. Notebook 19 showed it for pumps; here 10 clips of each type are checked.

## Reported, no decision
- The per-level table (T1) for all tracks, with B+10 and the development and confirmation scores.
- T2, unchanged from notebook 22b, recomputed at −6 dB.
- DCASE 2020:
  - the development set, per machine and per type;
  - the evaluation set, per machine and per type;
  - AUC and pAUC for both.
- The control.
- Training and pass times.

## Expected (written now; not decisions)
- **T1:**
  - At 6 dB the published numbers are 0.968–0.998. Expect misses there on every track, at least on valves and slide rails.
  - At −6 dB, MIC is above on all four types and ARR on all but valves (0.9744 against 0.9794).
- **T3: uncertain.** On MIMII's −6 dB Split A, MIC's mean over the four types is 0.9483, below 0.9549. The DCASE machines differ, and the retrained models learn seven machines per type.
- **The control:** no expectation.

## Caveats written now
- **DCASE 2020 compliance (MIC):** BEATs, BEATs iter3 and ECHO are outside DCASE 2020's list of allowed models. T3 is a post-challenge comparison.
- **MIMII at 0 and 6 dB uses systems built at −6 dB.** The PD and XC models, notebook 20's denoisers and the MWF weights are applied unchanged. That is what "frozen" means here. A system built for each level would probably do better at 6 dB.
- **Published MIMII numbers use another split:** T1 is indicative, even on MIC.
- **The evaluation machines' training clips:** MIC's MIMII models already used 600 of each as XC classes. These are training clips only, which DCASE 2020 allows.
- **The control's seeds 3–5:**
  - They come from notebook 22b's training code, which follows notebook 21's recipe.
  - GPU training is not bit-reproducible. So the six seeds are six draws of the same recipe, not six exact repetitions of notebook 21's runs.
- **Earlier comparisons:** notebooks 20–22b made many. This notebook makes none.

## Outputs
- **The Drive folder `P04_ExternalJudgement`** (new) holds:
  - the trainings and their embeddings;
  - the DCASE evaluation scores (official CSV format);
  - `external_judgement.json`;
  - the pictures;
  - `p04_n23_results.zip`.

## Compute (G4)
- **Downloads:**
  - about 100 GB of MIMII (three levels);
  - 4.4 GB of DCASE development data;
  - 0.95 GB of DCASE evaluation data;
  - the additional training data, as before.
- **15 trainings:**
  - 6 for the control on MIMII's folds;
  - 9 on the DCASE lists: 6 for the system, 3 for the control.
- **Passes:**
  - about 32 encoder passes on MIMII at 0 and 6 dB;
  - 9 on DCASE;
  - ECHO at 0 and 6 dB and on DCASE.
- **Time:** expect about 4–5 hours. The notebook prints its own estimate.

## Next (not part of this notebook)
- The write-up, deck, README and repository come only after notebook 23, and only with the user's approval.

## Sources
- `PROGRAM_full_written_before_running.md` (targets T1–T3, the five teams, arXiv 2309.07500 Table 1).
- DCASE 2020 Task 2 page: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
  - Baseline development results (mean of 10 trials).
  - Averaging per machine type, and the ranking method.
  - Data naming.
  - Test data never used for training; the additional training data may be.
- Zenodo 3678171 (development), 3727685 (additional training), 3841772 (evaluation), 3951620 (ground truth `eval_data_list.csv`).
- Official evaluator: https://github.com/y-kawagu/dcase2020_task2_evaluator (`evaluator.py`: `roc_auc_score`, `max_fpr=0.1`, arithmetic mean over machine IDs).
