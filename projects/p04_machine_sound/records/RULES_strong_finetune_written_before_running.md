# P04 · Notebook 21 · Strong fine-tuning on three tracks. Rules written before running

Written 2026-11-10, after notebook 20's results and before notebook 21 existed. This is stage 3 of the program, as revised in `PROGRAM_full_written_before_running.md` (Revisions 1 and 2).

## Questions (all at −6 dB, all four machine types, for each track)
1. Does fine-tuned BEATs beat frozen BEATs (LDN1) on the same input?
2. Which of these ingredients helps, each added one at a time?
   - **EP:** scoring on the fine-tuned backbone's pooled output instead of the head.
   - **XC:** extra machines from DCASE 2020.
   - **PL:** pseudo-labels.
   - **TS:** twin noise swap.
   - **PD:** pseudo-clean distillation.
   - **SE:** a seed ensemble.
   - **EF:** fusion with frozen ECHO.

## Why
- **Fine-tuning was the largest gain so far for a known machine.** Notebook 16 raised pumps from 0.88 to 0.95 at −6 dB.
- **Strong DCASE systems of 2025–2026 fine-tune BEATs or EAT.**
  - They use LoRA and an angular-margin classifier over machines and attributes, and train on as many machines as the rules allow.
  - SJTU-AITHU 2025 fine-tuned on all machine types of DCASE 2020–2025. THUEE 2025 fine-tuned on all six ASD datasets.
  - Where attributes are missing, they use k-means pseudo-labels: MERL 2026; arXiv 2509.12845.
- **DCASE 2026's winner (MERL, first of 51 teams).**
  - It learned noise-robust representations by distillation: the student hears the noisy sound, and the target is the frozen model's representation of the clean sound.
  - It then fine-tuned discriminatively and averaged three seeds.
  - Its largest gain was on valves (valveEmu 66.6 → 93.3 with NA-BEATs).
  - It scored on the fine-tuned backbone's features, not on the classification head. Notebook 18 (looked after) also saw the pooled 768 beat the 256 head on new pumps (B+10 0.9429 against 0.8874).
  - MIMII's twins give a pseudo-clean target (the 6 dB twin) for every training clip.
- **Inputs:** notebook 20 chose N2N-CH0 (one microphone) and MWF (array). Revision 2 adds the strict microphone-0 track.

## Tracks (inputs at −6 dB)
| Track | Input of a MIMII clip | Twin-based ingredients | Frozen reference (BEATs, LDN1; notebook 20) |
|---|---|---|---|
| **ARR** | MWF₋₆(x₋₆), notebook 19's filter of the machine | TS and PD allowed | development 0.9442, confirmation 0.8835 |
| **MIC+** | N2N-CH0: notebook 20's fold-k denoiser of the type on microphone 0 | TS and PD allowed | development 0.8135, confirmation 0.8238 |
| **MIC** | CH0: microphone 0 | none (DCASE 2020 setting) | development 0.7633, confirmation 0.7767 |

The frozen references were computed from notebook 20's JSON before writing these rules: S of BEATs with LDN1, as defined below.

## Data
- **MIMII.** The 6 dB and −6 dB zips of all four types, with the MD5s of notebook 20. The 6 dB twin is read for training clips only.
- **DCASE 2020 additional training dataset** (Zenodo 3727685; CC BY-NC-SA 4.0, so we report results and never redistribute the data):

  | File | MD5 |
  |---|---|
  | `eval_data_train_fan.zip` | 567798854130c8019df8b664c095ed1e |
  | `eval_data_train_pump.zip` | 4e33ae9c0db5cc88437675f5e317caee |
  | `eval_data_train_slider.zip` | f84f186f4295fc23754abcdd1b0580a7 |
  | `eval_data_train_valve.zip` | 057cdd560caffb8305449541afd28c6d |

  - Normal clips of the machines DCASE 2020 evaluates, whose IDs are not in the public MIMII release.
  - We use the first 600 by name per machine.
  - They are never scored here. Their test clips are not downloaded.
- **From Drive:**
  - `P04_TwinFrontEnds` (notebook 19): MWF weights.
  - `P04_DenoiseScore` (notebook 20): embeddings, the N2N-CH0 denoisers and `denoise_score.json`.
  - `P04_BEATs`: the checkpoint.
  - `P04_AudioCache` is not read (Amendment 1).

## Folds and splits (as notebooks 16 and 20)
- **Folds:** fold 1 is IDs 00 and 02 of every type; fold 2 is IDs 04 and 06.
- **Training:** model k trains on fold k's MIMII machines, Split-A training clips only, plus the extra machines when XC is in the recipe.
- **Scoring:** Split A of a machine uses its own fold's model. Splits B and B+n use the other fold's model, so that model never heard the machine.
- **MIC+:** model k always reads the output of denoiser k, so no learned part has heard a new machine.
- **ARR:** the MWF filters were estimated from each machine's own training-clip twins. This is calibration, as in notebook 19.
- **Test clips:** never trained on, and only at −6 dB.

## Base recipe R0
Notebook 16's recipe, with the changes marked.
- **Model:**
  - LoRA rank 16, α 32, dropout 0.1, on the query and value projections of all 12 layers.
  - Mean pooling of the last layer, then a head 768 → 256 (no bias) with batch normalisation.
  - Sub-center ArcFace: 3 sub-centres, scale 30, margin 0.2.
- **Classes:** one per machine in the fold, so 8 (2 IDs × 4 types). *Changed:* notebook 16 had 12, with valves and slide rails in both folds and never scored.
- **Training:**
  - 4-second random crops; mixup in half the batches, λ ~ Beta(0.5, 0.5).
  - AdamW: LoRA 2e-4 (no weight decay); head and ArcFace 1e-3 (weight decay 1e-4).
  - Batch 32, 10 epochs, 1 epoch of warm-up, then cosine decay.
  - float16 autocast with gradient scaling; gradient norm clipped at 5; seed 0.
  - No early stopping and no validation on test clips.
- **Embeddings:**
  - Model in evaluation mode, the whole 10-second clip.
  - Encoder in float16. *Changed:* notebook 16 used float32. Notebook 20's G0 showed float16 moves AUC by at most 0.0003.
  - F, the 256 head outputs, and P, the 768 pooled outputs, are both saved for every training. Scoring starts on F; EP decides whether P replaces it.
- **Scorer:** LDN1 (notebook 20's D3). B1 is reported.

## Ingredients (tested in this order; each is kept only if it passes)
0. **EP · the pooled embedding (all tracks; scoring only).**
   - The candidate scores R0 on P (768) instead of F (256).
   - If it is kept, every later decision on that track uses P, and SE and EF combine P.
1. **XC · extra machines (all tracks).**
   - The 12 DCASE 2020 machines (3 per type), first 600 clips each, are added as extra classes in both folds.
   - This gives 20 classes, or 80 with PL.
   - Their input: raw microphone 0 on MIC. On ARR and MIC+, N2N-CH0 with the fold's denoiser of their type, because they have no array recording.
   - XC also adds training steps, since each epoch holds more clips.
2. **PL · pseudo-labels (all tracks).**
   - Each class machine is split into 4 sub-classes by k-means (k = 4, 10 starts, seed 0).
   - The k-means runs on the unit-normalised frozen BEATs embeddings of the machine's training clips, on the track's input.
   - The MIMII embeddings come from notebook 20's files. The XC machines are embedded here.
3. **TS · twin noise swap (ARR, MIC+; MIMII clips only).**
   - With probability 0.5, a training sample is replaced by y₆[i] + c·g·roll(D[j], r).
     - D = y₋₆ − y₆ is noise only, because the twins share the machine segment (notebook 19's G1, case I).
     - j is another training clip of the same machine, and r is a uniform circular shift.
     - c = √((κ₋₆ − κ₆)/(κ₋₆ + κ₆)) = 0.93877, with κ₋₆ = 3.98107 and κ₆ = 0.25119.
     - g = √(P_D[i]/P_D[j]), with P_D the mean power of D over the clip.
   - Physics: the result has clip i's machine sound, and noise from another segment at the power of clip i's own −6 dB noise.
   - **Domains:**
     - ARR: in the MWF domain, y = MWF₋₆(x) of the machine.
     - MIC+: in the microphone-0 domain, then the fold's denoiser on the whole clip, then the crop.
4. **PD · pseudo-clean distillation (ARR, MIC+; MIMII clips only).**
   - **Added loss:** the mean over tokens of 1 − cos between two sets of last-layer tokens:
     - the student's tokens;
     - frozen BEATs's tokens (no LoRA, evaluation mode) on the 6 dB twin of the same window.
   - **The 6 dB twin:** ARR MWF₋₆(x₆); MIC+ raw microphone 0 at 6 dB.
   - **Mixup:** the teacher input is mixed with the same λ and partner. Pairs that involve an XC clip are left out of this loss.
   - **Weight:** λ_PD = mean ArcFace loss / mean distillation loss, over 8 batches before training (no gradient), then fixed for the training.
5. **SE · seed ensemble (all tracks).**
   - Seeds 0, 1 and 2 of the current recipe.
   - Each model's embedding (F, or P after EP) is unit-normalised; the three are concatenated and divided by √3. This averages the cosine distances. LDN1 is applied to the result.
6. **EF · fusion with frozen ECHO (all tracks; scoring only).**
   - The current system's embedding and notebook 20's ECHO embedding of the same input, each unit-normalised, concatenated and divided by √2, then LDN1.
   - For MIC+, the ECHO embedding uses the same denoiser fold as the model.

## Gates
- **G0 · reproduction** (any failure stops the run):
  - (a) The clip lists equal notebook 20's.
  - (b) Frozen BEATs LDN1 and B1, recomputed from notebook 20's embeddings, equal notebook 20's JSON for every split and machine of CH0, MWF and N2N-CH0.
    - The tolerance is 0.0005 per machine. Floating-point ties can move the fourth decimal.
    - The three frozen references above must also come back within 0.0005.
  - (c) Frozen BEATs on our streams, first 32 clips of each type, against notebook 20's saved embeddings: cosine ≥ 0.999 for CH0, MWF, N2N-CH0 fold 1 and N2N-CH0 fold 2.
- **GT · swap physics.**
  - Measured on 200 random training pairs per type and domain.
  - The median of P(swap) / P(the clip's −6 dB input) must be within [0.95, 1.05].
  - If it fails for a domain, TS is not tested on that track. This is reported.
- **G1 · learning.** Each training's last-epoch mean loss must be below its first.
  - If a candidate's training fails, its decision is "not decidable", and the current recipe stays.
- **Drive space.** The run needs at least 2 GB free in Drive, for about 1.5 GB of embeddings and adapters.

## Decisions (per track)
- **The score:**
  - S is the mean over the 8 development machines (pumps, fans) of (A + B+10)/2, at −6 dB, with LDN1 on F (or on P after EP).
  - Confirmation is the same over valves and slide rails.
- **Keep rule:** keep a candidate if both hold:
  - its development score ≥ the current recipe's + 0.01;
  - its confirmation score ≥ the current recipe's − 0.01.
- **Order:** R0 → EP → XC → PL → TS → PD → SE → EF. Each candidate is the current recipe plus one ingredient. MIC skips TS and PD.
- **FF · the final system against frozen BEATs LDN1 on the same input** (references above), with the same keep rule.
  - If it passes, the track's system from here on is the fine-tuned final.
  - Otherwise the track keeps frozen BEATs with LDN1.

## Reported, no decision
- **Tables:** every recipe, track, type, split and machine, for LDN1 and B1 and for F and P.
- **Seed spread:** S of seeds 0, 1 and 2 of the final recipe.
- **T1 per track** against arXiv 2309.07500 at −6 dB (Split A). Like for like only on MIC.
- **T2:** ARR B+10 against the frozen 6 dB targets (pump 0.9785, fan 0.9539).
- **Context:** notebook 16's fine-tuned CH0 and BF (B1).
- **Other numbers:**
  - frozen BEATs + ECHO fusion (LDN1);
  - the training logs, λ_PD and the GT ratios;
  - an XC level check: the median RMS of XC clips over that of MIMII −6 dB microphone-0 clips, per type;
  - run times.

## Expected (written now; not decisions)
- Fine-tuning raises Split A on every track, as in notebook 16, and B+10 less. The gain is largest on MIC and smallest on ARR, whose input is already clean.
- XC and PL help, because more classes make a harder proxy task (the DCASE consensus).
- TS helps fans most, because fan noise resembles fan sound.
- PD helps B+10 most, because it pulls the −6 dB representation towards the quiet one, and frozen BEATs at 6 dB already reaches the T2 targets.

## Caveats written now
- **Seeds:** one seed per decision. The SE step measures the spread for the final recipe only.
- **Many comparisons:** up to 7 decisions per track on 3 tracks can produce a chance pass. The 0.01 bar and confirmation reduce that but do not remove it.
- **Untuned settings:** the TS probability (0.5), the PD balancing rule, k = 4 for PL, 600 XC clips and 10 epochs.
- **DCASE 2020 compliance of MIC:**
  - MIC uses only −6 dB microphone-0 data and DCASE 2020's additional training data.
  - But our MIMII Split-A training clips may include clips that DCASE 2020 used as development test clips. Notebook 23 therefore retrains MIC's final recipe on DCASE's own training lists before scoring the evaluation set.
  - BEATs is not on DCASE 2020's list of allowed models. If EF is kept on MIC, ECHO (2025) is a second such model, and T3 will say so.
- **Licences:** MIMII is CC BY-SA 4.0; the DCASE 2020 data are CC BY-NC-SA 4.0.
- **Compute:** about 38 trainings. The notebook times one step before any download and prints its estimate. The expectation is about 4–5 h on an RTX PRO 6000 (Colab G4).

## Next (not part of this notebook)
- **Notebook 22:** newer encoders (EAT and others) on the winning inputs.
- **The 2026 scoring ideas, on the final models:**
  - MERL's frequency-wise memory bank;
  - deviation-based temporal pooling, which preserves short events and is aimed at valves.

## Sources
- DCASE 2020 Task 2 rules and data (https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds); Zenodo 3727685.
- DCASE 2025 Task 2 technical reports:
  - SJTU-AITHU: https://dcase.community/documents/challenge2025/technical_reports/DCASE2025_Zheng_64_t2.pdf
  - THUEE: https://dcase.community/documents/challenge2025/technical_reports/DCASE2025_Jiang_74_t2.pdf
- arXiv 2509.12845: attribute-aware representation from domain-adaptive pre-training, with pseudo-attribute labels.
- MERL, "The MERL Systems for DCASE 2026 Challenge Task 2", TR2026-100 (https://www.merl.com/publications/docs/TR2026-100.pdf); DCASE 2026 Task 2 description, arXiv 2606.01578.
- Wilkinghoff et al., IEEE TASLP 2026 (LDN); arXiv 2309.07500 (published MIMII numbers).

## Amendment 1 (2026-11-10, after the first attempt stopped at gate G0(c); no model had been trained)
- **What happened.**
  - On fans, frozen BEATs on our rebuilt streams did not reproduce notebook 20's embeddings.
  - Smallest cosine: MWF 0.976, N2N-CH0 fold 1 0.987 and fold 2 0.99994, against the bar 0.999. Microphone 0 gave 1.0.
  - Pumps passed. Their streams were read from notebook 20's cache.
- **The likely cause.**
  - The processed signals (MWF and the denoiser's output) were stored as 16-bit integers, which clip every sample beyond ±1.
  - Notebook 20 computed those embeddings from the unclipped float signals.
  - Microphone 0 cannot clip, because it comes from 16-bit files.
- **The change.** It touches storage only; every rule, setting and bar stays.
  - MWF and N2N streams are stored as float16, which does not clip. Microphone-0 streams stay 16-bit, which is exact.
  - Notebook 20's audio cache is no longer read, because its MWF streams are 16-bit.
  - The notebook reports, per type and stream, the share of clips whose processed signal reaches |x| ≥ 1. If that share is zero where G0(c) failed, the cause was something else.
  - The first attempt's XC frozen embeddings (`xc_frozen_{type}.npz`, pumps and fans) are not used. They are recomputed as `xc_frozen2_{type}.npz`.
- **Consequence for earlier notebooks (looked after).** Notebook 19's N2N (MVDR base) and notebook 20's N2N-MWF were trained and scored on 16-bit streams.
  - If those streams clipped, those two results carry the defect.
  - Every other front end of notebooks 19 and 20 (CH0, DS, MVDR, MWF, N2N-CH0) was embedded from float signals and is not affected.

Anything computed outside these rules is labelled "looked after".
