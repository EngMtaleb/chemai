# P04 · Notebook 27 · ToyCar and ToyConveyor: the kept system on two machine types that no notebook has read. Rules written before running

Written 2026-11-12, after notebook 26's results (`N26_outputs_2026-11-12.md`) and before notebook 27 existed. The user chose this step from the three that notebook 26's report laid out.

## Why
- **Every number of the program rests on recordings that were read more than once.**
  - MIMII's 12 cells: three readings (notebooks 23, 24, 26).
  - DCASE 2020's four MIMII types: their development set chose the fusion (notebook 25), and their evaluation set was scored twice (notebooks 23 and 25).
- **DCASE 2020 Task 2 has two more machine types, ToyCar and ToyConveyor.** No notebook of this project has opened them (notebook 23's record: "not run here"; notebook 25's rules: "not in this project").
- **They are another kind of machine:** toys on a bench, from the ToyADMOS recordings. The recipes were built on MIMII's fans, pumps, valves and slide rails.
- **With them the challenge's six machine types are complete** for this system, so it can be set beside the challenge's teams on the whole task.

## Questions
1. **Do the two recipes carry over?** Built unchanged on these two types' own training lists, how does the kept system do on their evaluation sets, scored once?
2. **Does fusing still help there?** The kept system against each of its two parts.
3. **The six types together,** against the five teams of the program's table.

## What is fixed: nothing is chosen in this notebook
- **The system is the kept one** (notebooks 25 and 26): the frozen recipe and the second view, fused by form ZP.
- **No setting changes for these types.** The frozen recipe keeps notebook 21's settings; the view keeps notebook 25's.
- **No form, weight, seed or epoch is selected on these types.** Their development set is read here for the first time, and it selects nothing.
- **Nothing here changes the kept system or any earlier number.**

## Data (MD5s from the Zenodo record pages, read 2026-11-12)
| Set | Zenodo record | Files used (MD5) |
|---|---|---|
| Development | 3678171 | `dev_data_ToyCar.zip` (4dec75ca8d9f666aa9e4c1894a740501), `dev_data_ToyConveyor.zip` (03b6aa1bfd09a39d53af0ba39f71fa91) |
| Additional training | 3727685 | `eval_data_train_ToyCar.zip` (a5d4ccd498af70b04dd74b644b820006), `eval_data_train_ToyConveyor.zip` (0e808ac0b761e9fb13d5e0c65caff00d) |
| Evaluation | 3841772 | `eval_data_test_ToyCar.zip` (bea2bdd612f616be8f2b8eb087b32c7c), `eval_data_test_ToyConveyor.zip` (ccfcc9847c7d3404d7ee9d6d9e0b2ba2) |
| Ground truth | 3951620 | `eval_data_list.csv` (f16aaa185df9906e588864aaf7cb58ea), as in notebooks 23–25 |

- **The same three pages list the four MIMII types' files** with the MD5s that notebooks 23 and 25 checked on real downloads. That is the check on this reading of the pages.
- **Machines** (as the view's reference code lists them; the notebook reads them from the files and stops if they differ):
  - ToyCar: development IDs 01, 02, 03, 04; evaluation IDs 05, 06, 07.
  - ToyConveyor: development IDs 01, 02, 03; evaluation IDs 04, 05, 06.
- **Clip length.** ToyConveyor's clips are 10 s long and ToyCar's about 11 s. Every system here reads the first 10 s of a clip (160,000 samples), as the view's reference code does (`y[..., :sr * 10]`). A clip shorter than 10 s stops the run.
- **Labels.** Development test clips carry them in their names. The evaluation labels are in the ground-truth file, which is read only in the evaluation step.
- **No test clip of either set is used for training,** as DCASE 2020's rules require.

## Part 1 · The frozen recipe on these types
This is notebook 23's Part 2, with the machine types changed and nothing else.
- **Classes and clips:**
  - one class per development machine, with all of its `train/` clips;
  - one class per evaluation machine, with its first 600 additional-training clips by name (XC's cap).
  - That is 13 classes: 7 for ToyCar and 6 for ToyConveyor.
- **One model per seed, no folds.** Seeds 0, 1 and 2 for each encoder (BEATs iter3+ and BEATs iter3): six trainings of 10 epochs, with notebook 21's settings.
- **ECHO:** frozen, run on every clip.
- **The scorer is MIC's.** For each encoder: global LDN1 on F, the three seeds unit-normalised and concatenated (÷√3), ECHO concatenated (÷√2). The score is the mean of the two encoders' scores.
- **References:** all of a machine's normal training clips. For an evaluation machine that is all of its additional-training clips.

## Part 2 · The second view on these types
This is notebook 25's view, with the machine types changed and nothing else.
- **Model and training:** TASTgram-MFN with Noisy-ArcMix, the settings of github.com/soonhyeon/Noisy-ArcMix at the commit notebook 25 pinned; 300 epochs; seeds 0, 1 and 2.
- **Classes and clips:** one class per machine (13), with all development `train/` clips and all additional-training clips of the two types.
- **Hold-out:** 5 % of those clips, one draw for the three seeds (`np.random.default_rng([27, 0])`), saved in Drive. The saved model of a seed is the epoch with the lowest hold-out loss.
- **Score:** the paper's score (the cross-entropy for the clip's own machine class, margin on), averaged over the three seeds.

## The kept system: form ZP
- **ZP = log(s_F) / σ_F + log(s_P) / σ_P.** s_F is the frozen recipe's score and s_P the view's; both are floored at 1e-6.
- **The spreads use no test clip** (notebook 25's rule):
  - σ_F: the frozen recipe's score on the additional-training clips its trainings did not use (those after the first 600 of an evaluation machine, by name), against the 600 they used.
  - σ_P: the paper's score on the view's held-out clips of the type.
  - A spread is 1.4826 × the median absolute deviation of the log score. Each needs at least 30 clips. If one cannot be formed, the run stops.

## Order, and the one-time scoring
1. **Gate G0,** then the six trainings of Part 1 and the three of Part 2 (gate G1).
2. **The development set:** AUC and pAUC per machine and per type for the kept system and its two parts.
3. **Gate G-DEV,** before the evaluation set is touched.
4. **The evaluation set is scored once.**
   - Three sets of scores are written before any label is read: the kept system's, and each of its two parts'. All are in the official CSV format.
   - **The kept system's is the result.** The two parts are there for question 2.
   - The ground truth is then read, in this step only. The results are saved to Drive at once; a later run reads them and scores nothing again.
   - If an error is found after this step, any second scoring is reported as a second scoring.
5. **The official evaluator** (github.com/y-kawagu/dcase2020_task2_evaluator, the commit notebooks 24 and 25 pinned) is run on each set of score files. If its numbers differ from the notebook's, the evaluator's are the result.

- **Metrics,** as the official evaluator computes them: AUC and pAUC (`max_fpr = 0.1`) per machine with `sklearn.metrics.roc_auc_score`; per type, the arithmetic mean over its machines.

## Gates
- **G0(a):** every zip has the MD5 above.
- **G0(b):** the files are what the rules assume.
  - The machine IDs are the ones listed above.
  - Every development machine has training clips and test clips of both kinds. Every additional-training clip is a normal clip. Evaluation test names carry no label.
  - Every clip is 16 kHz, one channel, at least 10 s. The counts and lengths are recorded.
- **G0(c):** the BEATs code files, both BEATs checkpoints and ECHO's weights are notebook 23's (SHA-256).
- **G0(d):** the view has the reference's sizes: 313 frames, a last kernel of 8 × 20, and 13 classes here.
- **G1:** the training loss falls in all nine trainings, and each seed's saved view has a hold-out loss below its first epoch's.
- **G-DEV:** on the development set, each of the three systems has an AUC of at least 0.7877 on ToyCar and at least 0.55 on ToyConveyor.
  - 0.7877 is the DCASE 2020 baseline's ToyCar AUC there (the task page, mean of 10 trials).
  - ToyConveyor's line is not the baseline's 0.7253. Systems that learn machine identity are weak on that type: one of the program's five teams had 0.6521 there. The gate is there to catch a broken pipeline, not to judge the system.
- A failure of any gate stops the run, and the evaluation set is then not scored.

## How the evaluation result will be read (fixed now)
The yardsticks, evaluation set, AUC / pAUC % (the DCASE 2020 results page, read 2026-11-12):

| System | Official rank | ToyCar | ToyConveyor |
|---|---|---|---|
| Giri_Amazon_task2_2 | 1 | 94.34 / 89.73 | 91.19 / 73.34 |
| Daniluk_SRPOL_task2_4 | 3 | 93.52 / 83.87 | 90.51 / 77.56 |
| Primus_CP-JKU_task2_2 | 5 | 86.37 / 83.83 | 88.28 / 79.15 |
| Vinayavekhin_IBM_task2_2 | 7 | 93.16 / 87.69 | 87.41 / 72.03 |
| Zhou_PSH_task2_3 | 12 | 95.60 / 91.30 | 73.61 / 64.06 |
| DCASE 2020 baseline | 93 | 80.14 / 66.17 | 85.36 / 66.95 |

- The same page's rows give these teams' four MIMII types exactly as the program's table has them.
- **Question 1, per type, one of three words** for the kept system's AUC:
  - **with the teams:** above at least 3 of the 5 teams;
  - **above the baseline:** at least the baseline's AUC, and above fewer than 3 teams;
  - **below the baseline.**
  - pAUC is reported beside it against the same rows, with no word.
- **Question 2, per type,** the kept system's AUC less the frozen recipe's, and less the view's, on the evaluation set:
  - **helps** if it is at least 0.01 above the better of its two parts;
  - **costs** if it is more than 0.01 below the better part;
  - **no clear difference** otherwise.
  - The same is reported on the development set.
- **Question 3, the six types.** The four MIMII types are the kept system's evaluation results of notebook 25 (AUC / pAUC): fan 0.9980 / 0.9894, pump 0.9812 / 0.9202, slide rail 0.9992 / 0.9960, valve 0.9919 / 0.9652.
  - **T3′:** the mean AUC over the six types is at least 0.9458. That is the rank-1 team's mean: (94.54 + 93.65 + 97.63 + 96.13 + 94.34 + 91.19) / 6. With notebook 25's four types, the two toy types must average 0.8523.
  - **Among the six systems** (this one and the five teams): its rank per type for AUC and for pAUC, and its mean rank over the 12 columns, beside each team's.
  - What these are: the task page says the AUC averaged over machine types should not be used, and the official ranking ranks all 117 submitted systems per type. T3′ and the rank among six are the program's summaries, not an official rank.

## Reported, no decision
- **Development set,** per machine and per type, AUC and pAUC:
  - the kept system and its two parts;
  - the view by LDN1 and by B1;
  - each encoder of the frozen recipe on its own, and ECHO alone;
  - the kept system with each seed of the view alone.
- **Published development numbers beside them** (AUC / pAUC %, all six types trained together):
  - Noisy-ArcMix with TASTgram-MFN (arXiv 2310.06364, Table 1): ToyCar 96.76 / 90.11, ToyConveyor 77.90 / 67.15.
  - The DCASE 2020 baseline (the task page): ToyCar 78.77 / 67.58, ToyConveyor 72.53 / 60.43.
  - AnoPatch with BEATs (arXiv 2406.11364, Table 1; the mean of AUC and pAUC): development ToyCar 96.10, ToyConveyor 73.20; evaluation ToyCar 96.00, ToyConveyor 83.74.
- **Evaluation set,** per machine and per type, for the three systems; the official evaluator's result files.
- **The spreads** with their clip counts.
- **Every clip's score:** the development test clips under every system, the evaluation score files, and the clips the spreads rest on. The label rows of the two types are saved beside them, so that every evaluation number can be recomputed from the results zip.
- **Training logs and times.**

## Expected (written now; not decisions)
- **ToyCar:** I expect it to carry over. Development AUC about 0.95–0.98 for the kept system (the view's paper has 96.76), and "with the teams" on the evaluation set (the five have 86.37–95.60).
- **ToyConveyor is where I expect the weakness.**
  - Systems that learn machine identity are weak there: the view's paper has 77.90 on the development set, and Zhou_PSH has 73.61 on the evaluation set.
  - The plain autoencoder baseline has 85.36 on the evaluation set, and the three top teams of the table, which add density models, have 88.28–91.19.
  - AnoPatch, a fine-tuned BEATs, reaches a mean of AUC and pAUC of 83.74 on the evaluation set, above Giri's 82.27. So the frozen recipe may do better there than the view.
  - I expect the kept system below those three teams. I do not know on which side of the baseline's 85.36 it falls.
- **The fusion:** no loss expected on ToyCar. On ToyConveyor I have no expectation; the view may pull the frozen recipe down there, as it did on pump id_02.
- **T3′:** it needs ToyConveyor at about 0.75 if ToyCar is at 0.95. I expect it reached, without confidence.

## Caveats written now
- **13 classes.** The view's paper trains the six types together (41 classes). Here the two toy types are trained on their own, as the four MIMII types were (28 classes). The frozen recipe likewise.
- **DCASE 2020 compliance:** BEATs, BEATs iter3 and ECHO are outside DCASE 2020's list of allowed models. This is a post-challenge comparison, as before.
- **ToyCar's last second is not read** (above).
- **The six-type summary mixes two kinds of evidence.** The two toy types are scored here once. The four MIMII types come from notebook 25, which was a second scoring of their evaluation set.
- **The five teams are the program's selection** (ranks 1, 3, 5, 7 and 12), not the whole results table.
- **Each training runs once.** GPU training is not bit-reproducible.
- **Three sets of scores are written on the evaluation set.** Only the kept system's is the result; the parts answer question 2 and cannot replace it.
- **No embedding of an earlier notebook is recomputed here.** There is none for these types. The forward passes are notebook 23's code, and gate G0(c) checks the files they load.

## Outputs
- **The Drive folder `P04_ToyTypes`** (new) holds the nine trainings, their embeddings and features, the evaluation score files, `toy_types.json`, the picture and `p04_n27_results.zip`.
- **It reads** `P04_BEATs` (both BEATs checkpoints). No other folder of the project is needed.

## Compute
- **Downloads:** about 6.4 GB from Zenodo; ECHO from Hugging Face.
- **Nine trainings:** six of the frozen recipe (10 epochs on about 10,600 clips) and three of the view (300 epochs on about 13,000 clips).
- **Time:** about 1.5–2 hours on G4, from notebook 23's and notebook 26's measured step times. A T4 would take many times longer (notebook 24: 7 times for the frozen recipe); it was never measured for the view.

## Next (not part of this notebook)
- The result is recorded as it is, whichever way it falls, and the next step is decided with the user.
- The write-up, deck, README and repository come only with the user's approval.

## Sources
- `N26_outputs_2026-11-12.md`; `RULES_external_judgement_written_before_running.md` (Part 2: the frozen recipe on DCASE 2020's lists); `RULES_second_view_written_before_running.md` (the view, form ZP and its spreads).
- DCASE 2020 Task 2 page: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds (data, rules, baseline development results, the ranking method).
- DCASE 2020 Task 2 results: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds-results (the five teams and the baseline on the evaluation set).
- Zenodo 3678171, 3727685, 3841772, 3951620.
- Noisy-ArcMix: arXiv 2310.06364, Table 1; code at github.com/soonhyeon/Noisy-ArcMix (clips cut to 10 s; the machine IDs).
- AnoPatch: arXiv 2406.11364, Table 1.
- Official evaluator: https://github.com/y-kawagu/dcase2020_task2_evaluator.
