# P04 · Rules for notebook 28 · the kept system, as it is, under a change of operating conditions (DCASE 2022 Task 2)

**Written 2026-11-12, before the notebook existed.** One gate (G-DEV) was changed the same day, before the notebook ran; the change is marked where it stands. No recording of DCASE 2022 Task 2 has been read by any notebook of this project.

## Why
- **Notebook 27** tested the kept system once on two machine types no notebook had read. It carried over to both. In those recordings, as in every recording of the program so far, the test clips come from the conditions the training clips come from.
- **A plant does not hold its conditions.** Speed, load, the microphone and the background change, and normal recordings of the new conditions are few at first.
- **DCASE 2022 Task 2 was built for that question.** Every section has 990 normal training clips from its source conditions and 10 from its target conditions. Half of its test clips come from the target conditions, and a system is not told which.
- **This stage measures. It improves nothing.**

## Questions
1. Where does the kept system, as it is, stand on the evaluation set's official score, against the challenge's 31 teams and its two baselines?
2. What does the change of conditions cost it: the source conditions against the target conditions?
3. Does the fusion of the two recipes help here?

## What is fixed: nothing is chosen in this notebook
- **The kept system** is the frozen recipe (MIC) plus the second view (TASTgram-MFN trained with Noisy-ArcMix), fused by form ZP with the view's paper score. Notebooks 25 and 26 kept it.
- **Both recipes run with their settings unchanged:** notebook 21's for the frozen recipe, notebook 25's for the view.
- **No form, weight, seed or epoch is chosen on these recordings.** Their development set is read for the first time, after the nine trainings, and it selects nothing.
- **One mapping has to be made, and it is made here: a section is a machine.**
  - In DCASE 2020 each machine had its own training clips and its own test clips, and each machine was one class for both recipes.
  - In DCASE 2022 the unit with its own training clips and its own test clips is the section. So each section is one class.
  - The domain of a clip (source or target) and the attributes in the file names are used nowhere: not in training, not in scoring. Neither recipe has a place for them.
- **The seven machine types are trained together** (42 classes), as the four MIMII types were (28) and the two toy types (13).

## Data (MD5s from the Zenodo record pages, read twice on 2026-11-12)
| Set | Zenodo record | Files (MD5) |
|---|---|---|
| Development | 6355122 | `dev_bearing.zip` (6381a00f9efc0ced779c8ad847e4ff59), `dev_fan.zip` (a1a9b488934a82426bacc933d87aacde), `dev_gearbox.zip` (c165dfef8c404256bd719c6fe1f7036f), `dev_slider.zip` (8c3a5466cf53e54872fd94998a67bfac), `dev_ToyCar.zip` (1ccca43751353e516ee7d6d55457c705), `dev_ToyTrain.zip` (17e5b8cae15ccb630a5c533551bc93b5), `dev_valve.zip` (1da37b2e82942dfba720984541e2ef60) |
| Additional training | 6462969 | `eval_data_bearing_train.zip` (e5c41a4cddd1a5ccfe32fbd182e84d73), `eval_data_fan_train.zip` (ee94f9843b86550259a0a24e99fe02dc), `eval_data_gearbox_train.zip` (122bb02c6d32c281aae8bfcb3b6a2d1d), `eval_data_slider_train.zip` (9fe49f109f491471720762fa8eb366b6), `eval_data_ToyCar_train.zip` (fde0d2e49a542c5e3e38f9ecd8329496), `eval_data_ToyTrain_train.zip` (e66a77da55eafe090f4f0645855fac2f), `eval_data_valve_train.zip` (e2396a438d65e6c61fc0698df72cbf66) |
| Evaluation | 6586456 | `eval_data_bearing_test.zip` (47518a98084260c11fa61ce9a0c54b7c), `eval_data_fan_test.zip` (5b1398524c46a67badf5e94b903c3d84), `eval_data_gearbox_test.zip` (768c162d3d2bb5147127a2965d54c2ca), `eval_data_slider_test.zip` (b2901b49bc001d6446d470260d32db64), `eval_data_ToyCar_test.zip` (3c52961e75d60623a40b9dcc47ec1009), `eval_data_ToyTrain_test.zip` (7df19e597a47b7cc1028969b6330be0f), `eval_data_valve_test.zip` (372c723bb3112340a880c282f5171775) |

- **Ground truth of the evaluation set:** the official evaluator's repository, `github.com/Kota-Dohi/dcase2022_evaluator`, at commit de8719b7d35280ff3a39487c9d1003413eef7818.
  - 21 label files and 21 domain files, one of each per evaluation section. Their combined SHA-256 (each file's path, a zero byte and its content, in path order) is d61ad20294be2e127d53ec2bdc1e9f3d1fb4e5cb932f97137ba7e231f2c1f35e.
  - The evaluator script, `dcase2022_task2_evaluator.py`: SHA-256 31bae0d9fa981b4ed3e92414bf62cd06adf75e0afba3f1a2d3d4b9408c5fdf4a.
  - These files were opened once here, to learn their layout and their counts: 200 rows per section, 100 anomalous, 100 from the target domain. No system's score was near them.
- **Machine types:** ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve.
- **Sections:** development 00, 01 and 02; additional training and evaluation 03, 04 and 05. The notebook reads them from the files and stops if they differ.
- **Clips, as the task page documents them:**
  - Training, per section: 990 source-domain and 10 target-domain clips, all normal.
  - Development test, per section: 100 normal and 100 anomalous clips, half of each from each domain. Their names carry the label and the domain.
  - Evaluation test, per section: 200 clips. Their names carry neither. The labels and the domains are in the ground truth, which is fetched only in the evaluation step.
  - In all: 42,000 training clips and 8,400 test clips.
- **Clip length:** 10 s, as documented. Notebook 27's rule stays: every system reads the first 160,000 samples of a clip, and a clip shorter than 10 s stops the run.
- **No test clip of either set is used for training, for a reference or for a spread.** No clip's score depends on another test clip, as the challenge's rules require.

## Part 1 · The frozen recipe on these recordings
This is notebook 23's Part 2 with the recordings changed and nothing else.
- **Classes and clips:** each development section with all its 1,000 training clips, and each evaluation section with its first 600 additional-training clips by name (XC's cap).
  - That is 42 classes and 33,600 clips.
  - File names put the source domain before the target domain. So the 600 clips of an evaluation section are source-domain clips, and its 10 target-domain training clips are not in the frozen recipe's training. They are among its references.
- **Trainings:** seeds 0, 1 and 2 for each of the two encoders (BEATs iter3+ and BEATs iter3): six trainings of 10 epochs.
- **ECHO** is frozen and embeds every clip.
- **Score:** MIC's scorer, unchanged. Global LDN1 per encoder on the three seeds' head features with ECHO concatenated; the system's score is the mean of the two encoders' scores.
- **References:** all of a section's normal training clips, 990 from the source domain and 10 from the target domain.

## Part 2 · The second view on these recordings
This is notebook 25's view with the recordings changed and nothing else.
- **Classes and clips:** the same 42 classes, with every training clip of the two sets: 42,000 clips.
- **Hold-out:** 5 % of them, one draw for the three seeds (`np.random.default_rng([28, 0])`), saved in Drive.
- **Trainings:** 300 epochs; seeds 0, 1 and 2. The saved model of a seed is the epoch with the lowest hold-out loss.
- **Score:** the paper's score, averaged over the three seeds: the cross-entropy for the clip's own section class, with the margin on.

## The kept system: form ZP
ZP = log(s_F) / σ_F + log(s_P) / σ_P, where s_F is the frozen recipe's score and s_P the view's.
- Each σ is the spread (1.4826 × the median absolute deviation) of that system's log score on normal clips it did not train on, per machine type. No test clip enters.
- **σ_F:** the additional-training clips after the first 600 of each evaluation section by name, scored against the 600 used. As documented that is 390 source-domain and 10 target-domain clips per section.
- **σ_P:** the view's held-out clips of the type.
- Each needs at least 30 clips. If one cannot be formed, the run stops.

## Order, and the one-time scoring
1. Gate G0, then the six trainings of Part 1 and the three of Part 2 (gate G1).
2. The development set, for the kept system and its two parts.
3. Gate G-DEV, before the evaluation set is touched.
4. **The evaluation set is scored once.**
   - Three sets of score files are written before any label is read: the kept system's, the frozen recipe's and the view's. The kept system's is the result. The two parts are there for questions 2 and 3.
   - The ground truth is fetched only after those files are written.
   - The results are saved to Drive at once. A later run reads them and scores nothing again.
5. **The official evaluator** (the pinned file, unmodified) is run on the three sets. If its numbers differ from the notebook's, the evaluator's are the result.
   - The evaluator also needs a decision file per section. The notebook writes a decision of 0 for every clip. Decisions play no part in the official score, and no threshold is set in this stage.

## Metrics (the official evaluator's)
- **Per section, three values:**
  - AUC (source): the section's normal clips of the source domain against all its anomalous clips;
  - AUC (target): its normal clips of the target domain against all its anomalous clips;
  - pAUC: all its clips, with `max_fpr = 0.1`.
- **Per machine type:** the harmonic mean of its nine values.
- **The official score Ω:** the harmonic mean of all 63 values (7 types × 3 sections × 3).
- The development set has the same three sections per type, so its Ω is formed the same way.

## Gates
- **G0(a):** every zip has the MD5 above.
- **G0(b):** the files are what the rules assume.
  - The seven types; sections 00–02 in the development set and 03–05 in the other two.
  - Every training clip is normal. Every training section has at least 600 source-domain clips before its first target-domain clip by name, and at least one target-domain clip.
  - Every development section has normal and anomalous test clips of both domains.
  - Evaluation test names carry no label and no domain, and each section's clips are numbered 0000 to 0199.
  - Every clip is 16 kHz, one channel, at least 10 s. The counts and lengths are recorded.
- **G0(c):** the BEATs code files, both BEATs checkpoints and ECHO's weights are notebook 23's (SHA-256).
- **G0(d):** the view has the reference's sizes: 313 frames, a last kernel of 8 × 20, and 42 classes here.
- **G1:** the training loss falls in all nine trainings, and each seed's saved view has a hold-out loss below its first epoch's.
- **G-DEV, a sanity gate before the evaluation set:** on the development set, each of the three systems (kept, frozen alone, view alone by the paper's score) has a source-domain line of at least 0.55. The source-domain line is the harmonic mean of the 21 sections' AUC (source).
  - The source domain is the conditions a system trained on. A system at chance has about 0.50 there (0.495 with a spread of 0.011, from 2,000 draws of random scores with the documented clip counts). Both baselines have more than 0.63. A value below 0.55 means a broken pipeline, not a weak system.
  - **Changed on 2026-11-12, before the notebook ran and before any recording was read.** The gate was first written on the official score, at 0.50. That was wrong for this data: under a change of conditions a working system's AUC (target) can fall below 0.50 (the autoencoder baseline's is 34.81 on ToyCar and 23.35 on ToyTrain), so an official score near 0.50 does not show a broken pipeline. The official evaluator, run on random scores against the evaluation ground truth while testing the notebook, gave 0.49996.
- A failure of any gate stops the run, and the evaluation set is then not scored.

## How the evaluation result will be read (fixed now)
**Yardsticks:** the challenge's team ranking, the best system of each of its 31 teams (official score, %):

| Place | Team's best system | Official score |
|---|---|---|
| 1 | Liu_CQUPT_task2_4 | 70.97 |
| 2 | Kuroyanagi_NU-HDL_task2_2 | 68.22 |
| 3 | Guan_HEU_task2_4 | 68.04 |
| 4 | Deng_THU_task2_4 | 67.62 |
| 5 | Venkatesh_MERL_task2_3 | 67.57 |
| 6 | Wei_HEU_task2_2 | 67.12 |
| 7 | Morita_SECOM_task2_1 | 66.83 |
| 8 | Bai_JLESS_task2_1 | 63.95 |
| 9 | Verbitskiy_DS_task2_1 | 63.83 |
| 10 | Wilkinghoff_FKIE_task2_1 | 63.72 |
| 11 | Nejjar_ETH_task2_1 | 63.35 |
| 12 | Du_NERCSLIP_task2_1 | 63.30 |
| 13 | Tozicka_NSW_task2_1 | 62.50 |
| 14 | Yamashita_GU_task2_3 | 62.39 |
| 15 | Almudevar_UZ_task2_2 | 60.60 |
| 16 | Liu_BUPT_task2_2 | 59.77 |
| 17 | Gou_UESTC_task2_4 | 59.17 |
| 18 | PENG_NJUPT_task2_1 | 58.47 |
| 19 | Li_CTRI_task2_1 | 58.13 |
| 20 | Cohen_Technion_task2_2 | 56.76 |
| 21 | Hu_NJU_task2_1 | 55.07 |
| 22 | CHO_SG_task2_1 | 54.47 |
| 23 | Narita_AIT_task2_2 | 53.85 |
| 24 | Kodua_ITMO_task2_1 | 52.96 |
| 25 | Jalalia_AIT_task2_1 | 51.34 |
| 26 | Jinhyuk_SNU_task2_1 | 51.03 |
| 27 | Siang_NTHU_task2_1 | 50.59 |
| 28 | Kazakova_ITMO_task2_1 | 49.90 |
| 29 | LEE_KNU_task2_1 | 49.27 |
| 30 | Li_JAIST_task2 | 48.94 |
| 31 | Zorin_AIRI_task2_1 | 43.17 |

The two baselines: MobileNetV2 54.02 (official rank 68) and the autoencoder 52.94 (official rank 75).

- **Question 1, the word for the kept system's official score Ω on the evaluation set:**
  - **"above the first team":** Ω is above 70.97.
  - **"with the leading teams":** Ω is at least 66.83, the seventh team's. The first seven teams stand 2.88 points clear of the eighth.
  - **"in the upper half":** Ω is at least 59.77, the sixteenth team's, the middle of the 31.
  - **"above the baselines":** Ω is at least 54.02.
  - **"below the baselines"** otherwise.
  - Its place among the 31 teams is reported with the word: one more than the number of teams with a higher score.
  - Scores are compared in % of two decimals, as the results page prints them.
- **Question 2, the shift.** Reported with no word, for the kept system and its two parts on the evaluation set:
  - the harmonic mean of the 21 sections' AUC (source), and of their AUC (target), and the difference;
  - beside the autoencoder baseline's 64.23 and 45.13 (the evaluator's own example output, whose official score is the results page's 52.94).
- **Question 3, the fusion,** on Ω, on the evaluation set and on the development set:
  - "helps" if the kept system is at least 0.01 above the better of its two parts;
  - "costs" if it is more than 0.01 below the better part;
  - "no clear difference" otherwise.

## Reported, no decision
- **Development set:** per section, per type and overall, for
  - the kept system and its two parts;
  - the view by LDN1 and by B1;
  - each encoder of the frozen recipe on its own, and ECHO alone;
  - the kept system with each seed of the view alone.
- **Published development numbers beside them** (the task page; averages of five runs). The harmonic mean of a baseline's 21 published per-type harmonic means is its development Ω:
  - the autoencoder baseline: 52.61 (source 68.74, target 41.91, pAUC 53.73);
  - the MobileNetV2 baseline: 55.94 (source 63.80, target 50.04, pAUC 55.65).
- **Evaluation set:** per section, per type and overall for the three systems, and the official evaluator's result files.
- **The spreads** with their clip counts.
- **Every clip's score:** the development test clips under every system, the evaluation score files, and the clips the spreads rest on. The ground-truth rows are saved beside them, so that every evaluation number can be recomputed from the zip.
- **Training logs and times.**

## Expected (written now; not decisions)
I have no measurement of either recipe under a change of conditions, so these are weak.
- **The frozen recipe** scores a clip by its nearest reference, in units of that reference's own neighbourhood. The 10 target-domain references of a section are there, so a target-domain test clip has something near it. I expect it to hold up in part: better on the source domain than on the target domain, and "in the upper half".
- **The view** scores a clip by how surely the network names its section. A normal clip from conditions seen in 10 training clips of 1,000 may look as unfamiliar as an anomaly. I expect the view to be the weaker part, and weakest on the target domain.
- **The fusion:** I expect "no clear difference" or "costs". If the view is weak on the target domain, the calibrated sum lets that through, as it did on the machines of notebooks 26 and 27 where the view was weak.
- **The kept system:** "in the upper half", without confidence. "With the leading teams" is possible; those teams used the attributes and the domains, which this system ignores. I would be surprised by "above the first team".

## Caveats written now
- **The system ignores what the challenge's leading systems used:** the attributes in the file names, and handling made for the two domains. That is what "as it is" means here.
- **This is a post-challenge comparison.** BEATs and ECHO are later than the challenge and are not on its list of allowed resources.
- **The pre-trained encoders learned from general audio, not from machine recordings:** BEATs from AudioSet, ECHO from AudioSet and MTG-Jamendo (Table 2 of its paper).
- **The recordings are unread; some kinds of machine are familiar.** Fans, slide rails and valves were in DCASE 2020 too (MIMII there, MIMII DG here), and toy cars (ToyADMOS there, ToyADMOS2 here). The kept system's form was selected on DCASE 2020's fans, pumps, slide rails and valves.
- **An evaluation section's 10 target-domain training clips are not in the frozen recipe's training** (above). They are references, and they are in the view's training.
- **No decision threshold is set.** The evaluator's precision, recall and F1 are not results of this stage.
- **Each training runs once.** GPU training is not bit-reproducible.
- **Three sets of scores are written on the evaluation set.** Only the kept system's is the result.

## Outputs
- **The Drive folder `P04_DomainShift`** (new) holds the nine trainings, the embeddings and features, the evaluation score files, `domain_shift.json`, the picture and `p04_n28_results.zip`.
- **It reads** `P04_BEATs` (both BEATs checkpoints). No other folder of the project is needed.

## Compute
- About 12.4 GB of downloads (6.2 + 5.2 + 1.0) and 50,400 clips.
- The frozen recipe: six trainings of 10 epochs on 33,600 clips, each followed by a pass over every clip.
- The view: three trainings of 300 epochs on 39,900 clips. This is most of the time.
- **About 4.5 hours on G4,** from the step times notebook 27 measured there: about 1 hour for Part 1 and 3 hours for Part 2. Everything is saved to Drive and resumes.

## Next (not part of this notebook)
What follows depends on the three answers. If the change of conditions costs much, the next stage works on that, with the development set for its choices and a second scoring of the evaluation set, named as such.

## Sources
- DCASE 2022 Task 2, task page and results page (dcase.community/challenge2022), read 2026-11-12: the data layout, the metrics, the rules, the baselines' development tables, the team ranking of 31 teams, and the two baselines' official scores. The team ranking and the baselines' tables were each read twice.
- The official evaluator: `github.com/Kota-Dohi/dcase2022_evaluator`, commit de8719b7d35280ff3a39487c9d1003413eef7818, read in full. Its README's example output has the autoencoder baseline's official score, 0.5294160921841315, which is the results page's 52.94.
- Zenodo records 6355122, 6462969 and 6586456.
- ECHO: arXiv 2508.14689, Table 2 (ECHO-Small: AS2M + MTG).
- Notebooks 21, 23, 25 and 27 of this project, for the two recipes and their settings.
