# P04 · Notebook 26 · The second view on MIMII's noise levels (valves and pumps at 6 dB). Rules written before running

Written 2026-11-12, after notebook 25's results (`N25_outputs_2026-11-12.md`) and before notebook 26 read any real data. Three reported-only items (the reading ZB, the look at the held-out clips, the saved calibration scores) were added while the notebook was tested on synthetic data; no real result existed then. This is the second notebook of the stage after the program ("a new stage on the weak points"). Notebook 25's rules named it under "Next", on the condition that the view was kept. The view was kept, and the user decided this notebook on 2026-11-12.

## Why
- **Weak point 2: valves and pumps at 6 dB on MIMII.** The frozen MIC system, applied unchanged at each level (notebook 23, "reading 1"), against arXiv 2309.07500's single-microphone numbers. Each cell is "Split-A AUC / published"; bold is below the published number.

  | Level | Pump | Fan | Valve | Slide rail | Reached |
  |---|---|---|---|---|---|
  | 6 dB | **0.9875** / 0.9891 | 0.9968 / 0.9682 | **0.9951** / 0.9978 | 0.9990 / 0.9974 | 2 of 4 |
  | 0 dB | 0.9679 / 0.9115 | 0.9675 / 0.9260 | 0.9902 / 0.9830 | 0.9929 / 0.9750 | 4 of 4 |
  | −6 dB | 0.9485 / 0.8977 | 0.8911 / 0.8778 | 0.9830 / 0.9794 | 0.9706 / 0.9441 | 4 of 4 |

  - The two missed cells: pumps at 6 dB by 0.0016, valves at 6 dB by 0.0027.
  - Where the gap sits, per machine at 6 dB: valves 1.0000, 0.9999, 0.9990 and **0.9816** (id_06); pumps 0.9969, **0.9744** (id_02), **0.9831** (id_04) and 0.9955.
  - Training the recipe at each level did not help (notebook 24, "reading 2": 9 of 12 cells). The program closed T1 as not reached in full.
- **What is new since then: notebook 25 kept a second view.** On DCASE 2020's development set, which is MIMII's microphone 0 at −6 dB in DCASE's own split, the frozen system fused with the view (form ZP) went from 0.9400 to 0.9794 on pumps and fans, and from 0.9953 to 0.9991 on valves and slide rails.
  - Per machine there: valve id_06 0.9925 → 1.0000; pump id_04 0.9916 → 1.0000; **pump id_02 0.9605 → 0.9408.**
- **The question left open in notebook 25's rules:** does the same fusion, in the program's own MIMII protocol, close the two cells at 6 dB?

## Questions
1. In MIMII's protocol at −6 dB, does the frozen MIC system fused with the view (form ZP) pass the program's keep rule on Split A?
2. If it does: at 6 dB, does it reach arXiv 2309.07500's numbers for pumps (0.9891) and valves (0.9978), and do the ten cells that reading 1 reached stay reached?

## What the view can serve
- **Split A only (a known machine).** The view is a classifier of machine identity, and its score is the cross-entropy of a clip for its own machine's class. A machine that is not one of its classes has no score.
- So nothing here touches splits B, B+1, B+3 or B+10, nor ARR or MIC+. The program's T2 and its B+10 numbers stay as they are.

## The view (notebook 25's recipe, unchanged; retrained on the program's split)
- **TASTgram-MFN trained with Noisy-ArcMix** (Choi and Choi, ICASSP 2024, arXiv 2310.06364; reference code at commit 674c759bfaa7723619a4b1d8d00eb9d155c50ba6, MIT licence). Every setting is notebook 25's: the three input channels, the network, ArcFace with scale 30 and margin 0.7, Noisy-ArcMix on every batch with λ ~ Beta(0.5, 0.5), AdamW at 1e-4 with cosine annealing to 1e-5, batch 64, 300 epochs, 5 % held out, the model with the lowest hold-out loss, seeds 0, 1 and 2. The code is notebook 25's, and so are its stated deviations from the reference (mixed precision with a gradient scaler, ArcFace in float32 with a floor, the speed settings, three seeds).
- **Why notebook 25's three models are not reused.** DCASE's development split puts 100 normal clips of each machine in its test set. The program's split puts a machine's first N normal clips by name there (N = its number of abnormal clips). A model trained on DCASE's lists has therefore seen clips that the program's split tests on.
- **Training lists: 28 classes, 21,583 clips, all microphone 0 at −6 dB.**
  - The 16 MIMII machines (IDs 00, 02, 04 and 06 of each type): their normal training clips in the program's split, 11,419 clips. Per type: pump 3,293; fan 2,600; valve 3,212; slide rail 2,314.
  - DCASE 2020's 12 additional machines (IDs 01, 03 and 05 of each type): all their additional-training clips, 10,164 clips, as in notebook 25. Per type: pump 2,417; fan 2,846; valve 2,531; slide rail 2,370.
  - 20,503 train and 1,080 are held out: one random draw of 5 % for the three seeds, saved in Drive.
- **Not used for training:** any test clip of the program's split at any level; any clip at 0 or 6 dB; DCASE's development or evaluation test clips.
- **One model per seed, on all 16 machines. No folds.** This differs from the sentence in notebook 25's rules ("in the program's two-fold protocol"), and the reason is the protocol itself, not a result:
  - The fold rule exists so that no learned part has heard a new machine before splits B and B+n. The view cannot serve those splits at all.
  - In Split A the machine is known, and its normal training clips may be learned from. That is DCASE's setting too.
  - So the lists are notebook 25's 28 classes with the program's split: three trainings instead of six.
- **Trained at −6 dB only, and applied unchanged at 0 and 6 dB,** as every system of reading 1. No clip at 0 or 6 dB is learned from.

## The frozen system (nothing in it changes)
- The frozen MIC system of notebooks 22b and 23: BEATs iter3+ and BEATs iter3, each R0 + XC with seeds 0–2 and two folds; per encoder, LDN1 on the three seeds' F (unit-normalised, concatenated, ÷√3) with ECHO concatenated (÷√2); the mean of the two encoders' scores.
- In Split A a machine is scored by its own fold's models, with its normal training clips at the level as references (notebook 23).
- **It is not retrained and not recomputed from audio.** Its embeddings of every MIMII clip are read from Drive: at −6 dB from `P04_StrongFineTune` (notebook 21), `P04_NewEncoders` (notebook 22b) and `P04_DenoiseScore` (notebook 20's ECHO and clip lists); at 6 and 0 dB from `P04_ExternalJudgement/levels` (notebook 23). No BEATs or ECHO model is loaded.
- Its level tables are recomputed from those files and must equal notebook 23's (gate G0(c)).

## The fused system: form ZP, as notebook 25 selected it (no new selection)
- **ZP = log(s_F)/σ_F + log(s_P)/σ_P.** Scores at or below 1e-6 are set to 1e-6.
  - s_F: the frozen system's score of the clip at that level.
  - s_P: the paper's score of the clip at that level: the cross-entropy for its own machine's class, averaged over the three seeds.
- No other form is computed: not A, G, C or Z.
- **σ is measured at each level, on that level's normal training clips.** Notebook 25's rule is that each system counts in units of its own spread on normal clips, with no test clip and no label. Each level is evaluated on its own (notebook 23), so the spread is measured where the scores are produced. σ is 1.4826 × the median absolute deviation of the log score, pooled over a machine type's four machines.
  - **σ_P (the view):** on the held-out 5 % of the type's MIMII training clips, at that level. No seed trained on these clips at −6 dB, and no model has seen them at 0 or 6 dB. The fixed draw holds out 576 MIMII clips: 157 (pump), 136 (fan), 171 (valve) and 112 (slide rail).
  - **σ_F (the frozen system):** on every normal training clip of the type's four machines at that level, each scored against its machine's other training clips (the clip itself left out), with its machine's own fold's models.
  - **A deviation from notebook 25, stated now.** There σ_F came from clips the frozen models had never trained on. In MIMII's protocol no such clips exist: every normal training clip of a machine was a training clip of its fold's models at −6 dB. The left-out score is the nearest thing that every level has. For comparison, notebook 25's values on unseen clips were 0.2035 (pump), 0.1925 (fan), 0.1696 (valve) and 0.2167 (slide rail); this notebook reports its own beside them.
  - **If a σ rests on fewer than 30 clips, or is zero,** for any type at a level that is read, ZP cannot be formed there and the run stops. There is no other form to fall back on.
- **What calibration at a level uses:** normal training clips of that level, which reading 1 already uses as references. It uses no test clip and no label.
  - The held-out clips are the same recordings at every level (the one draw made at −6 dB). Nothing is computed from a clip together with its twin at another level.

## Decision at −6 dB (the program's keep rule, on Split A)
- **Metric:** AUC per machine, rounded to four decimals; a type's value is the mean of its four machines, rounded to four decimals (notebook 11's evaluation, as in every level table of the program).
- **Scores:** development = the mean of pumps and fans; confirmation = the mean of valves and slide rails; both on Split A at −6 dB, rounded to four decimals.
  - The program's own score also averages in B+10. The view cannot serve B+10, so here both scores use Split A alone, for the current system and for the fused one.
- **Current system (notebook 23, Split A at −6 dB):** development 0.9198 (pump 0.9485, fan 0.8911); confirmation 0.9768 (valve 0.9830, slide rail 0.9706).
- **Keep rule:** the fused system is kept for known machines on MIMII if its development score is at least 0.9298 **and** its confirmation score is at least 0.9668.
  - There is no fallback. If it fails, nothing is kept and the levels below are not read.
- Valves and slide rails never choose anything.

## The per-level table (only if kept; read once)
- **Reading 3:** the kept system's Split-A value for each type at 6, 0 and −6 dB, against arXiv 2309.07500, Table 1 (the table above). The view's features and scores at 6 and 0 dB are computed only after the keep rule has passed.
- **How it will be read (fixed now):**
  - **Answered:** pumps and valves at 6 dB both reach the published numbers, and the ten cells that reading 1 reached stay reached.
  - **Answered in part:** one of the two is reached; or both are, and another cell is lost. It is then stated cell by cell.
  - **Not answered:** neither is reached.
- **T1 stays closed.** Notebook 24's rules closed it as not reached in full, for the program's recipe. Reading 3 is a later stage's table, for a system the program did not have. It is reported beside readings 1 and 2 and replaces neither.
- These 12 cells have been read twice before, and the weak cells were named with those readings in view. So reading 3 is "looked after", as notebook 25's evaluation scoring was.

## Gates
- **G0(a) · the files.** Every file the frozen system needs is in Drive: notebook 20's four clip lists with ECHO, the 12 embedding files at −6 dB, and the 104 level files at 6 and 0 dB.
- **G0(b) · the data.** Every zip has Zenodo's MD5 (the MD5s in notebook 23). At each level, a type's WAV names equal notebook 20's clip list, so this session's audio is in the order of the saved embeddings.
- **G0(c) · the frozen system comes back.** Recomputed from the saved embeddings, MIC's three level tables equal notebook 23's: all five splits, every machine and every average, within 0.0005 (the program's tolerance). The largest difference is reported.
- **G0(d) · the lists.** 28 classes, 21,583 clips, 20,503 train and 1,080 held out; 313 frames and a last kernel of 8 × 20.
- **G1 · every training:** the loss is finite throughout; the last epoch's training loss is below the first; the saved model's hold-out loss is below the first epoch's.
- **G-VIEW · the view works.** At −6 dB on Split A, for every type, the view alone reaches the DCASE 2020 baseline's development AUC (fan 0.6583, pump 0.7289, slide rail 0.8476, valve 0.6628: the same recordings in DCASE's split) by the paper's score or by LDN1. Checked after the first seed and again with three.
- A failed gate stops the run before any decision, and is reported as our failure.

## Reported, no decision
- **At −6 dB:** the frozen system, the view alone (the paper's score, LDN1, plain nearest-neighbour distance B1) and ZP, per type and per machine; ZP with each seed alone.
- **The spreads:** σ_F and σ_P with their clip counts, per type and level.
- **If kept, at 6 and 0 dB:** the same, and one more reading of ZP with the −6 dB spreads carried to 6 and 0 dB instead of measured there. That shows what calibration at the level changes. The result is the table with the level's own spreads.
- **One more fused reading, ZB, at every level that is read (reported only):** the same calibrated sum with the view's plain nearest-neighbour distance in place of the paper's score: log(s_F)/σ_F + log(s_B)/σ_B.
  - s_B is B1 on the view's matrix, against the machine's normal training clips at the level. σ_B is the spread of its log on the held-out clips, each against its machine's other training clips at the level.
  - Why it is reported: the paper's score measures a clip against classes learned at −6 dB. B1, like the frozen system's score, measures it against references of its own level. If the view's features carry over to quieter clips and its class score does not, ZB shows it.
  - It decides nothing and answers nothing. The kept system is the one notebook 25 kept, with the paper's score. If a spread for ZB cannot be formed, ZB is left out and that is reported.
- **How the view's class score carries over, without any abnormal clip:** at every level that is read, the share of the held-out MIMII clips (normal clips no seed trained on) that the view assigns to their own machine (largest logit with the margin on, as in the paper's score; mean over the seeds), and the median of their paper's score. It says whether a fall of the paper's score at another level comes from the normal clips themselves.
- **Every test clip's score** under the frozen system, the view and ZP, at every level that is read, so that each number can be recomputed from the results zip.
- **The clips the spreads rest on,** at every level that is read: the frozen system's left-out score of every normal training clip, and the held-out clips' scores under the view. The spreads can then be recomputed from the zip as well.
- **Looked after, at −6 dB** (the question notebook 25's record left open, asked of this view): for each machine, the reference clip that is nearest to the most abnormal test clips in the view's space, with its file name, its own nearest-neighbour distance and whether it was held out; and the three normal test clips with the highest paper's score.
- **Training:** the curves, the saved epoch, hold-out loss and accuracy, time per step, the GPU.

## Expected (written now; not decisions)
- **The view alone at −6 dB:** close to its numbers in notebook 25 (the paper's score: pump 0.9616, fan 0.9793, valve 0.9983, slide rail 0.9961). The recordings are the same; the split and the list of training clips differ.
- **The keep rule at −6 dB:** I expect it to pass, mostly through fans (0.8911 now; fan id_00 is at 0.6857).
- **Valves at 6 dB:** reached, if the view carries over to quieter clips. The gap is one machine, id_06, and at −6 dB the fusion took that machine to 1.0000 in DCASE's split.
- **Pumps at 6 dB:** uncertain, and they may fall. The lowest pump at 6 dB is id_02, the one machine the fusion made worse in notebook 25.
- **The other ten cells:** I expect them to hold. The thinnest margin is slide rails at 6 dB (0.9990 against 0.9974).
- **The part I trust least is the paper's score at 0 and 6 dB.** It measures a clip against classes learned at −6 dB, while the frozen system's score measures it against references of its own level.
- **What I do not know:** whether a model trained from scratch at −6 dB carries over to 0 and 6 dB. The fine-tuned BEATs did (notebook 24). If the view's own Split-A value at 6 dB is below its value at −6 dB, it did not carry over, and the fusion may not help there.

## Caveats written now
- **A third reading of the same 12 cells** (above).
- **Published MIMII numbers use another split.** The comparison is indicative, as in notebooks 23 and 24.
- **The view sees all four machines of a type; a fold's frozen models saw two.** Both are legitimate in Split A, where the machine is known. It is not the like-for-like test of one recipe that readings 1 and 2 were.
- **σ_F is measured on clips the frozen models trained on** (above).
- **The view trains on DCASE 2020's additional-training clips as 12 more classes,** all of them at −6 dB. The MIC track allows them (the program, Revision 2), and the frozen models used the first 600 per machine.
- **The fused system was selected on DCASE 2020's development set,** which holds the same recordings as MIMII's −6 dB level. The keep rule here is therefore a confirmation in another split, not an independent test. The 6 and 0 dB levels are where the system has not been seen.
- **One training per seed.** No training is repeated to pass a rule.

## Outputs
- A new Drive folder, `P04_SecondViewMIMII`: three models with their training logs, the view's features and scores of every MIMII clip at each level that is read, `second_view_mimii.json`, one picture, and `p04_n26_results.zip`.
- It reads the four folders named above and writes nothing there.

## Compute
- **Downloads:** MIMII's 12 zips, about 100 GB (the −6 dB zips first; the 6 and 0 dB zips download while the trainings run and are used only if the view is kept), and DCASE's additional training data, 2.5 GB. Only microphone 0 is kept: about 21 GB on the local disk.
- **Three trainings of 300 epochs:** 321 steps per epoch, 96,300 steps each.
- **Time.** Notebook 25 measured 0.019–0.020 s per step for this model on the RTX PRO 6000 (G4). That gives about 31 minutes per training, about 1.6 hours for the three, and with the downloads about 2–3 hours in all. A T4 was never measured for this model; it could take 8–20 hours. The notebook measures its own GPU in the first minute of training and prints the estimate.
- **Resume:** each training is saved to Drive every 10 epochs. A new session downloads the audio again and continues from the last save. Once the features of a level are in Drive, that level's audio is not needed again.
- **Order:** the frozen system's gates (minutes, before any download); the −6 dB audio; seed 0 and G-VIEW; seeds 1 and 2; the decision at −6 dB; then, only if kept, the 6 and 0 dB audio and reading 3.

## Next (not part of this notebook)
- **If weak point 2 is answered,** both weak points of this stage are. The write-up, deck, README and repository come only with the user's approval.
- **If it is not,** the result is recorded as it is, and the next step is decided with the user. One step this notebook would then point to, and does not take: the view trained at each level.

## Sources
- `PROGRAM_full_written_before_running.md` (T1, arXiv 2309.07500's Table 1, the splits, the fold rule, the three tracks).
- `N23_outputs_2026-11-11.md` and `run_2026-11-11_n23/` (reading 1, per machine and per split).
- `N24_outputs_2026-11-11.md` and `run_2026-11-11_n24/` (reading 2; the sizes of the two folds' training lists).
- `N25_outputs_2026-11-12.md` and `run_2026-11-12_n25/` (the kept view, form ZP, its spreads, the measured step time).
- `RULES_second_view_written_before_running.md` (the view's settings, form ZP, the calibration rule; "Next").
- Choi and Choi, "Noisy-ArcMix", ICASSP 2024, arXiv 2310.06364 (https://arxiv.org/abs/2310.06364); code https://github.com/soonhyeon/Noisy-ArcMix.
- MIMII, Zenodo 3384388; DCASE 2020 Task 2 additional training data, Zenodo 3727685; the task page for the baseline's development results (https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds).
