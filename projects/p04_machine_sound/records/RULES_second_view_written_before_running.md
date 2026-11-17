# P04 · Notebook 25 · A second view for the single-microphone system on DCASE 2020 (fans and pumps). Rules written before running

Written 2026-11-12, after notebook 24's results (`N24_outputs_2026-11-11.md`) and before notebook 25 existed. This is a new stage, after the program: every target of the program is already reached or closed with its reason. The user chose this stage on 2026-11-11 ("a new stage on the weak points").

## Why
- **Weak point 1 (this notebook): fans and pumps on DCASE 2020.** The frozen MIC system (notebook 23), AUC / pAUC:

  | Set | Pump | Fan | Valve | Slide rail | Mean AUC |
  |---|---|---|---|---|---|
  | Development | 0.9726 / 0.9356 | **0.9074** / 0.8338 | 0.9977 / 0.9897 | 0.9930 / 0.9712 | 0.9677 |
  | Evaluation | **0.9440** / 0.8557 | **0.9593** / 0.9092 | 0.9744 / 0.9379 | 0.9991 / 0.9966 | 0.9692 |

  - The best 2020 systems per type on the evaluation set (the program's table): fan 99.79, pump 97.76, slide rail 99.84, valve 97.82.
  - The weakest machines: development fan id_00 (0.7329) and id_04 (0.9161); evaluation pump id_03 (0.8831), valve id_05 (0.9233) and fan id_05 (0.9270).
- **Weak point 2 (the next notebook, only if this one keeps the view): valves and pumps at 6 dB on MIMII.** See "Next".
- **What the program never tried:** a model trained from scratch on spectrograms to tell the machines apart. Every system so far is a pre-trained encoder (BEATs, ECHO) with small adapters.
- **Why that may help (published numbers, checked at the source on 2026-11-11):**
  - Systems built on a fine-tuned BEATs are weak on fans and strong on pumps. AnoPatch (BEATs iter3, fully fine-tuned): development fan 86.46, pump 93.10 (mean of AUC and pAUC). Ours, in the same metric: fan 87.06, pump 95.41.
  - Classifiers of machine identity trained from scratch are strong on fans. Noisy-ArcMix with TASTgram-MFN: development fan 98.32, pump 95.44, slide rail 99.53, valve 99.95 (AUC).
  - Our own diagnostic (notebook 22b, looked after): our BEATs does not keep a band-limited change in its own frequency row.
  - So the two families may fail on different clips. That is what a fusion needs.

## Questions
1. Does a second view, trained from scratch with a published recipe on DCASE 2020's own training clips, fused with the frozen MIC system, raise pumps and fans on the DCASE 2020 development set by at least 0.01 AUC without losing valves and slide rails?
2. If it does: what does the fused system give on the evaluation set, scored once?

## The second view (published recipe; settings fixed now)
**TASTgram-MFN trained with Noisy-ArcMix** (Choi and Choi, ICASSP 2024, arXiv 2310.06364). The reference implementation is github.com/soonhyeon/Noisy-ArcMix at commit 674c759bfaa7723619a4b1d8d00eb9d155c50ba6 (MIT licence, Copyright (c) 2023 soonhyeon). I read it in full and did not run it. The notebook holds an adapted copy with the licence notice.

- **Input:** the whole clip (10 s, 160,000 samples, microphone 0 as DCASE provides it), as floats in [−1, 1]. Three channels of 128 × 313:
  - Sgram: log-mel spectrogram. FFT 2048, hop 512, 128 mel bands (Slaney scale and norm), power 2, zero padding, 10·log10(mel + 2.2e-16).
  - Tgram: Conv1d(1→128, kernel 1024, stride 512, padding 512), then 3 × [LayerNorm(313), LeakyReLU(0.2), Conv1d(128→128, kernel 3)].
  - TAgram: the log-mel times sigmoid(max over mel + mean over mel), per time frame.
- **Network:** MobileFaceNet as in the reference (bottlenecks [2, 128, 2, 2], [4, 128, 2, 2], [4, 128, 2, 2]; a 512-channel depthwise layer with kernel 8 × 20; a 128-d feature). About 1 million parameters.
- **Loss:** ArcFace with scale 30, margin 0.7, one centre per class. Noisy-ArcMix on every batch:
  - one λ ~ Beta(0.5, 0.5) and one permutation per batch;
  - the waves and the log-mels are mixed with it;
  - the margin is applied at the first clip's class; the loss is λ·CE(first class) + (1 − λ)·CE(second class) on the same logits.
- **Training:** AdamW, learning rate 1e-4, weight decay 0.01 (the default), cosine annealing over the epochs to 1e-5 (stepped per epoch), batch 64, 300 epochs.
- **Model selection, as the reference:** 5 % of the training clips are held out at random. After each epoch the hold-out loss is computed (evaluation mode, margin on, no mixing). The model with the lowest hold-out loss is the one saved and used.

### Data (DCASE 2020's own training lists; no test clip)
- **28 classes:** for each of the four types, the 4 development machines (all `train/` clips of the development zip) and the 3 evaluation machines (all clips of the additional training zip).
- **23,283 clips:** 13,119 development and 10,164 additional (counted from notebook 23's saved lists).
  - Per type (development + additional): pump 3,349 + 2,417; fan 3,675 + 2,846; valve 3,291 + 2,531; slide rail 2,804 + 2,370.
  - 22,118 train and 1,165 are held out, per seed.
- **Not used for training:** any `test/` clip of the development set, any clip of the evaluation test set, any label of the evaluation set, any MIMII clip outside these zips, ToyCar and ToyConveyor.
- These are the clips DCASE 2020's rules allowed for training (the program, Revision 2).

### Deviations from the reference (stated now)
1. **28 classes instead of 41.** ToyCar and ToyConveyor are not in this project.
2. **Three seeds (0, 1, 2)** instead of one run. Each seed has its own hold-out, its own initial weights and its own batch order.
3. **Mixed precision with a gradient scaler.** The reference uses autocast without a scaler. The ArcFace layer runs in float32, and 1 − cos² is clamped at 1e-9 before the square root.
4. **Score.** The paper's anomaly score is the cross-entropy of the clip for its own machine class. The program's scorer is LDN1 on embeddings, and the fusion uses that (below). The paper's score is reported.
5. **Speed settings that do not change the mathematics:** the audio is kept on the GPU, the log-mel is computed on the GPU with the same torchaudio transform, and the memory format and convolution algorithms are chosen for speed. Training is not bit-reproducible.
6. **Short clips in the synthetic test only:** the frame count (313) and the last kernel (8 × 20) follow the clip length. A real run stops unless they are 313 and 8 × 20.

### The view's embedding and score
- **Embedding of a clip:** the 128-d feature of the saved model, in evaluation mode.
- **The view's matrix:** the three seeds' features, each unit-normalised, concatenated, divided by √3 (the program's seed ensemble).
- **The view's score:** global LDN1 (K = 1, floor 1e-6) on that matrix. The references are all normal training clips of the same machine, as in notebook 23: for a development machine its `train/` clips, for an evaluation machine all its additional-training clips. The held-out 5 % are among them.

## The frozen system (nothing in it changes)
- Notebook 23's MIC system on DCASE's lists: BEATs iter3+ and BEATs iter3, each R0 + XC with seeds 0–2, LDN1 on F with ECHO concatenated; the mean of the two encoders' scores.
- It is **not retrained and not recomputed from audio.** Its embeddings of every DCASE clip are read from Drive › `P04_ExternalJudgement` (notebook 23). No BEATs or ECHO model is loaded in this notebook.
- Its numbers are recomputed from those embeddings and must equal notebook 23's (gate G0).

## Candidates (three fusion forms, fixed now)
With s_F the frozen system's score, s_V the view's score, M_e the frozen matrix of encoder e and V the view's matrix:
- **A · mean of scores:** ½·s_F + ½·s_V. The program's fusion, with one weight per family.
- **G · geometric mean:** √(s_F · s_V). Both are ratios; this form does not depend on either one's scale. Scores at or below 1e-6 are set to 1e-6.
- **C · concatenation:** for each encoder, LDN1 on [M_e, V]/√2; then the mean of the two encoders' scores. This is how the program added ECHO.

The view alone is **not** a candidate. A model trained from scratch on a fixed list of machines cannot serve a new machine without retraining, which the program's B and B+10 splits need. It is a complement for known machines.

## Decision (DCASE 2020 development set; official metric)
- **Metric:** AUC per machine, averaged over a type's four machines (the official evaluator's way). pAUC (p = 0.1) is reported.
- **Scores:** development = the mean of pumps and fans; confirmation = the mean of valves and slide rails. Both are rounded to four decimals.
- **Current system (notebook 23):** development 0.9400 (pump 0.972641, fan 0.907373); confirmation 0.9953 (valve 0.997729, slide rail 0.992963).
- **Selection:** the form with the highest development score. A tie goes to the earlier one in the order A, G, C.
- **Keep rule (the program's):** the selected form is kept if its development score is at least 0.9500 **and** its confirmation score is at least 0.9853.
  - There is no fallback. If the selected form fails, nothing is kept, and the other forms' numbers are only reported.
- Valves and slide rails never choose anything.

## The evaluation set (a second scoring; stated plainly)
- The program's rule: the evaluation set is scored once, by the frozen final system. Notebook 23 did that. Its result is the program's T3 (mean AUC 0.9692 against 0.9549, confirmed by the official evaluator in notebook 24), and it stands. **T3 is not judged again.**
- This stage comes after the program. The weak types were named with that first scoring in view, so whatever is measured on the evaluation set from now on is "looked after".
- To keep this second look as clean as it can be:
  - every choice is made on the development set (the view's settings are the published ones; the form is selected on development pumps and fans);
  - the evaluation set is scored **only if the keep rule passes**, **once**, by the one system the rule keeps;
  - the scores are written to files before any label is read, and a later run only reads them;
  - no other system is scored on it: not the view alone, not the other forms, not single seeds.
- If nothing is kept, the evaluation set is not scored at all.
- **Metric and check:** as notebook 23, then the official evaluator (the same pinned file as notebook 24, SHA-256 1a514d38…e106a) on the new score files. If the two disagree by more than 1e-9, the official evaluator's numbers are the result.

## Gates
- **G0(a) · notebook 23's files in Drive are the recorded ones.** `external_judgement.json` (MD5 2e5b735796eb4029d5af8cad1d47d673), `dcase2020_evaluation/results.json` (MD5 afc5c84df7271303189c5bd4bbff459e) and the 12 score files (the MD5s in notebook 24's rules).
- **G0(b) · the data.** Every zip has Zenodo's MD5. The clip counts per type, set and machine equal notebook 23's.
- **G0(c) · the frozen system comes back, which also proves that this session's clip order matches the saved embeddings.**
  - Development: AUC and pAUC of all 16 machines equal notebook 23's within 0.0005 (the program's tolerance for a recomputation on another machine). The largest difference is reported.
  - Evaluation: the frozen system's scores of all 3,893 evaluation test clips equal notebook 23's saved score files within a relative 1e-6. No label is read for this, and it is not a new scoring: it is the same system and the same numbers.
- **G1 · every training:** the loss is finite throughout; the training loss of the last epoch is below the first; the hold-out loss of the saved model is below the first epoch's.
- **G-VIEW · the view works:** its development AUC (LDN1, alone) is at least the DCASE 2020 baseline's for every type (fan 0.6583, pump 0.7289, slide rail 0.8476, valve 0.6628). Checked after the first seed and again with three.
- A failed gate stops the run before any decision, and is reported as our failure.

## Reported, no decision
- **The view alone on development,** per type and per machine: LDN1, plain nearest-neighbour cosine distance (B1), and the paper's score; for each seed and for the three together.
  - The paper's score is also reported pooled over a type's machines, because the reference's evaluation script pools them. The official metric averages machines.
- **All three forms** on development, per type and per machine (AUC and pAUC).
- **Seed spread:** the selected form with each seed alone.
- **A preview after the first seed** (the three forms with that seed alone). It is not a decision: unless G-VIEW fails, all three seeds are trained whatever it shows.
- **Training:** the curves, the saved epoch, hold-out loss and accuracy, time per step, the GPU.
- **If the evaluation set is scored:** per type and per machine, beside notebook 23's numbers, the five teams' and AnoPatch's.

## Yardsticks (not decision bars)
- **Development set, published, AUC / pAUC %:**

  | System | Fan | Pump | Slide rail | Valve |
  |---|---|---|---|---|
  | Noisy-ArcMix, TASTgram-MFN (arXiv 2310.06364, Table 1) | 98.32 / 95.34 | 95.44 / 85.99 | 99.53 / 97.50 | 99.95 / 99.74 |
  | STgram-MFN with ArcFace (the same paper's Table 1) | 94.04 / 88.97 | 91.94 / 81.75 | 99.55 / 97.61 | 99.64 / 98.44 |
  | FGASpecNet (Mathematics 2026, 14(3), 530; no code) | 98.05 / 94.73 | 96.03 / 88.24 | 99.49 / 97.32 | 99.93 / 99.61 |
  | Ours, frozen (notebook 23) | 90.74 / 83.38 | 97.26 / 93.56 | 99.30 / 97.12 | 99.77 / 98.97 |

- **Evaluation set, AUC %:** the best of the program's five teams per type: fan 99.79, pump 97.76, slide rail 99.84, valve 97.82. Ours, frozen: 95.93, 94.40, 99.91, 97.44.
- **AnoPatch (arXiv 2406.11364, Table 1; mean of AUC and pAUC, %):**

  | Set | Fan | Pump | Slide rail | Valve |
  |---|---|---|---|---|
  | Development, AnoPatch | 86.46 | 93.10 | 99.20 | 97.53 |
  | Development, ours frozen | 87.06 | 95.41 | 98.21 | 99.37 |
  | Evaluation, AnoPatch | 95.56 | 94.34 | 99.77 | 96.26 |
  | Evaluation, ours frozen | 93.42 | 89.99 | 99.78 | 95.62 |

## Expected (written now; not decisions)
- **The view alone:** close to the published development numbers on fans (above 0.95 AUC), a little below the frozen system on pumps, about equal on valves and slide rails. Its weakest fan machine should be far above our 0.7329.
- **The fusion:** development fans rise by several points and pumps do not fall, so the development score passes 0.9500. Valves and slide rails stay within 0.01.
- **Which form:** I do not know. That is why three are fixed and the choice is made on development.
- **The evaluation set:** fans should rise. Pumps are uncertain: the frozen system's weak machine there (id_03, 0.8831) may or may not be one the view separates.
- **What would make me wrong:** 28 classes instead of 41, or LDN1 on these embeddings, may cost part of the published strength.

## Caveats written now
- **Published development numbers are not like for like in three ways.** They use 41 classes. The paper does not say whether a type's AUC averages its machines or pools them, and the reference's script pools them. And a paper reports its chosen run.
  - The paper also reports the worst machine of each type (mAUC): fan 92.67, pump 91.17, slide rail 97.96, valve 99.89. Ours, frozen: 73.29, 94.54, 98.66, 99.25.
- **The form is selected on the development test set,** from three, and then judged on the same set. The evaluation set is the check of that choice.
- **The evaluation set has been seen once already** (above).
- **BEATs (2022) and ECHO (2025) are not on DCASE 2020's list of allowed models,** so any fused result stays a post-challenge comparison, as T3 was. The view itself uses only what the 2020 rules allowed.
- **The frozen system trained on the first 600 additional clips per evaluation machine; the view trains on all of them.** Both are inside DCASE's training lists.
- **One training per seed.** No training is repeated to pass a rule.
- **The view serves known machines only.** It changes nothing on the program's B and B+10 splits, and nothing on ARR or MIC+.

## Outputs
- A new Drive folder, `P04_SecondView`: three models with their training logs, their embeddings and scores of every DCASE clip, `second_view.json`, one picture, and `p04_n25_results.zip`.
- If the evaluation set is scored: `dcase2020_evaluation_second_view/` with 12 score files in the official format, `scores_written.json`, `results.json`, and the official evaluator's result file.
- It reads `P04_ExternalJudgement` (notebook 23) and writes nothing there.

## Compute
- **Downloads:** about 7.8 GB (development 4.4 GB, additional training 2.5 GB, evaluation test 1.0 GB; sizes from the Zenodo records). No MIMII zip, no BEATs, no ECHO.
- **Three trainings of 300 epochs:** 346 steps per epoch, 103,800 steps each. About 0.8 GMACs per clip (my count from the layer sizes).
- **Time (not measured; my estimate from the layer sizes):** about 1–3 hours on the RTX PRO 6000 (G4), and about 6–24 hours on a T4.
  - The range is wide because I cannot measure it: a network of small convolutions uses a GPU less fully than BEATs does, and by how much I do not know.
  - Notebook 24 taught this: I gave one estimate and the run landed on a GPU eight times slower.
  - So the notebook prints its GPU, measures its own step time in the first minutes, and prints the estimate for the whole run.
- **Resume:** each training is saved to Drive every 10 epochs. A new session continues from the last save. Finished trainings and their embeddings are never redone.
- **Order:** the gates (minutes), seed 0 and its preview, then seeds 1 and 2, the decision, and the evaluation set last.

## Next (not part of this notebook)
- **If the view is kept: notebook 26, weak point 2.** The same view, trained in the program's two-fold protocol on MIMII's microphone 0 at −6 dB, fused with the frozen MIC system in the form selected here (no new selection), and read once on the per-level table: valves and pumps at 6 dB (0.9951 against 0.9978 and 0.9875 against 0.9891 in notebook 23's reading).
  - It needs MIMII at three levels again (about 100 GB) and six trainings. That is why it is its own notebook.
  - Its rules are written before it is built.
- **If the view is not kept:** the result is recorded as it is, and the next step is decided with the user.
- The write-up, deck, README and repository come only with the user's approval.

## Sources
- `PROGRAM_full_written_before_running.md` (rules for every stage, the five teams' table, Revision 2 on DCASE 2020's data rules).
- `N23_outputs_2026-11-11.md` and `run_2026-11-11_n23/` (the frozen system's numbers, the clip counts, the MD5s).
- `N24_outputs_2026-11-11.md` (T1's two readings, the official evaluator's check).
- `N22b_outputs_2026-11-10.md` (the frequency-row diagnostic).
- Choi and Choi, "Noisy-ArcMix: Additive Noisy Angular Margin Loss Combined With Mixup for Anomalous Sound Detection", ICASSP 2024, arXiv 2310.06364 (https://arxiv.org/abs/2310.06364); code https://github.com/soonhyeon/Noisy-ArcMix (`config.yaml`, `model/net.py`, `losses.py`, `dataloader.py`, `utils.py`, `trainer.py`, `train.py`, `eval.py`, read in full).
- Jiang et al., "AnoPatch", Interspeech 2024, arXiv 2406.11364, Table 1 (https://arxiv.org/abs/2406.11364).
- Bi et al., "Anomalous Sound Detection by Fusing Spectral Enhancement and Frequency-Gated Attention" (FGASpecNet), Mathematics 2026, 14(3), 530 (https://www.mdpi.com/2227-7390/14/3/530).
- DCASE 2020 Task 2: development data, Zenodo 3678171; additional training data, Zenodo 3727685; evaluation data, Zenodo 3841772; ground truth, Zenodo 3951620; official evaluator, https://github.com/y-kawagu/dcase2020_task2_evaluator.

## Amendment 1 (2026-11-12, before any run on real data; written while the notebook was being tested on synthetic clips)
Nothing has run on DCASE or MIMII clips. The synthetic test showed a weakness in the three forms above, and the fix is fixed here.

- **What the test showed.** In the first synthetic run (a view trained for 6 epochs on 232 two-second clips, and a stand-in for the frozen system), form C came out equal to the view alone, to four decimals, for all four types.
  - The reason is general. A and C add the two systems in each one's own units, so the system with the larger numbers decides.
  - A view trained for 300 epochs on its own reference clips will give larger LDN1 ratios on unseen clips than the frozen system does. G removes a constant factor between the two, but not a difference in spread.
  - If one system decides, the fusion is that system with a small correction. The gain this notebook looks for comes from clips that only one system catches.
- **A fourth form, Z · calibrated geometric mean:** log(s_F)/σ_F + log(s_V)/σ_V. Scores at or below 1e-6 are set to 1e-6.
  - σ is the spread of a system's log score on **normal training clips that the system did not train on**: 1.4826 × the median absolute deviation, pooled over a machine type.
  - **σ_V (the view):** the held-out 5 % of the type's clips, each scored with LDN1 against all the other training clips of its machine (the other held-out clips included: these are the references a test clip meets).
  - **σ_F (the frozen system):** for the type's evaluation machines, the additional-training clips after the first 600 by name (notebook 23's trainings used only the first 600), scored against those first 600.
  - Each system then counts in units of its own spread on normal clips. No test clip and no label enters.
  - **If either σ rests on fewer than 30 clips for any type,** Z is left out and that is reported. The counts are known for the frozen system: pump 617, fan 1,046, valve 731, slide rail 736.
- **A fifth form, ZP · the same calibrated sum with the paper's own score for the view:** log(s_F)/σ_F + log(s_P)/σ_P.
  - s_P is the paper's anomaly score (the cross-entropy of the clip for its own machine class), averaged over the three seeds. Scores at or below 1e-6 are set to 1e-6.
  - σ_P is the spread of log(s_P) on the held-out 5 % of the type's clips, which no seed trained on, with the same robust formula and the same minimum of 30 clips.
  - Why: every other form rests on LDN1 for the view. The published results come from the paper's score, and I do not know which of the two is better on these embeddings. Without a scale-free form the paper's score could not be fused at all: it runs over several orders of magnitude.
- **G-VIEW reads the view's two scores.** It passes for a type if the view alone reaches the DCASE 2020 baseline with LDN1 **or** with the paper's score.
  - The gate is there to catch a broken training or a broken pass. If only LDN1 were weak on these embeddings, the training would still be sound, and ZP would be the form to judge.
- **One hold-out for the three seeds,** not one per seed (this replaces the words "per seed" and "Each seed has its own hold-out" above).
  - The held-out clips must be unseen by every seed's model for σ_V to mean what it says.
  - The draw is random, 5 %, made once. It is saved in Drive (`hold_out.json`), and a later session reads it from there. The seeds still differ in their initial weights and their batch order. Each seed still keeps its own model with the lowest hold-out loss.
- **The family is now A, G, C, Z, ZP,** and a tie goes to the earlier one in that order. The selection, the keep rule and the use of the evaluation set are unchanged. Two gates read differently: G-VIEW (above) and G0(a) (next).
- **G0(a) reads differently for one file.** The 12 score files and `results.json` are checked by MD5, as written (notebook 24's run already passed that check).
  - `external_judgement.json` is not required to be byte-identical in Drive: running notebook 23's last cell again would rewrite it with a new date.
  - The numbers G0(b) and G0(c) need from it (the clip counts and the 16 machines' development AUC and pAUC) are carried in the notebook itself, read from notebook 23's results zip (MD5 8301df3f5c591db368a94265fd352cef).
  - Whether Drive's copy is the recorded file is reported, without stopping.
- **Reported:** σ_F, σ_V, σ_P and their clip counts per type; and every development test clip's score under every system, so that each development number can be recomputed from the results zip.
- **Caveat:** five forms are now compared on the development test set instead of three. They share their two ingredients, so they move together; the evaluation set stays the check of the choice.
