# P04 · Notebook 22 · Newer encoders and the DCASE 2026 scoring ideas, on three tracks. Rules written before running

Written 2026-11-10, after notebook 21's results (`N21_outputs_2026-11-10.md`) and before notebook 22 existed. This is the "newer encoders" stage of the program (Revision 1: frozen first, then LoRA on the best), with the scoring ideas that notebook 21's rules listed as next.

## Questions (all at −6 dB, all four machine types, for each track)
1. **D2 again.** On float streams, does N2N after MWF beat MWF? Notebook 20's D2 compared them unfairly (16-bit input against float).
2. **Scoring.** Does scoring per frequency band beat scoring the whole clip?
   - **BEAM:** per-band memory banks.
   - **AdaBEAM:** BEAM on two temporal views, mean and max.
   - **RDP:** relative deviation pooling in place of the band mean.
3. **Encoders.** Do newer encoders beat BEATs iter3+ (AS2M), frozen? Does the best one, fine-tuned with each track's winning recipe, replace or add to the track's system?

## The systems at the start (notebook 21)
| Track | Input | System | Development | Confirmation |
|---|---|---|---|---|
| **ARR** | MWF | fine-tuned BEATs, R0 + PD, scored on P (768), LDN1 | 0.9672 | 0.9558 |
| **MIC+** | N2N-CH0 | frozen BEATs, LDN1 | 0.8135 | 0.8238 |
| **MIC** | CH0 | fine-tuned BEATs, R0 + XC, seeds 0–2 (SE), fused with frozen ECHO (EF), scored on F (256), LDN1 | 0.8377 | 0.9006 |

- They are read from notebook 21's files in Drive (`P04_StrongFineTune`): adapters `ft_*.pt`, embeddings `emb_*.npz`, `strong_finetune.json`.
- Nothing of notebook 21 is retrained, except under D2′ below.

## Why
- **Scoring per band.**
  - BEAM (Saengthong and Shinozaki, arXiv 2603.13749) keeps the encoder's frequency rows apart.
    - For each frequency row of the patch grid, it pools over time and keeps a memory bank of the reference clips.
    - It finds the nearest neighbour per band and averages the band scores.
  - On DCASE 2020 (MIMII and ToyADMOS), with BEATs iter3:
    - frozen: global 82.1 → BEAM 86.9 → AdaBEAM 87.6 (development sections);
    - fine-tuned: 91.6 → 93.0 → 95.6 (development) and 93.0 → 94.0 → 96.2 (evaluation).
- **The DCASE 2026 winner (MERL, TR2026-100, first of 51 teams).**
  - It used "frequency-wise memory bank construction" (BEAM) and "deviation-based pooling" (RDP, Wilkinghoff, Yadav and Tan, arXiv 2603.04605) with deviation power γ = 4.
  - It added RDP to NA-BEATs and gained 0.65 points on its development set (63.92 → 64.57).
  - Its encoders were BEATs iter3, EAT-base (epoch 30, pre-trained) and EAT-large (epoch 20, pre-trained). Its five-encoder ensemble beat each encoder alone.
- **Encoders.**
  - Frozen, BEATs was the strongest on DCASE 2020 in arXiv 2603.04605: development / evaluation 86.8 / 87.4, against EAT 79.1 / 82.2 and Dasheng 80.9 / 81.6.
  - Fine-tuned, MERL's noise-aware EAT matched noise-aware BEATs (63.13 against 63.18 on its development set). A 2026 Task 2 system fine-tuned EAT with LoRA and scored with BEAM (Zhou et al., technical report 58).
  - So EAT is unlikely to win frozen, but it may help fine-tuned, alone or fused.
- **D2 again.** Notebook 21's diagnostic showed the 16-bit storage did not clip. The remaining difference is rounding, which notebook 20's N2N-MWF had and MWF did not.

## Encoders tested (fixed now)
| Name | Source | Patch grid of a 10-second clip | Width |
|---|---|---|---|
| BEATs iter3+ (AS2M), **the reference** | as notebooks 1–21 (SHA-256 d43cbfad…cd34) | 62 × 8 | 768 |
| BEATs iter3 | the authors' link `BEATs_iter3` (OneDrive), else Microsoft's Azure copy in the same container as our checkpoint; SHA-256 recorded | 62 × 8 | 768 |
| EAT-base | Hugging Face `worstchan/EAT-base_epoch30_pretrain` (MIT); commit recorded | 64 × 8 | 768 |
| EAT-large | Hugging Face `worstchan/EAT-large_epoch20_pretrain` (MIT); commit recorded | 64 × 8 | 1,024 |

- **Not tested, and why:**
  - Dasheng: weaker than BEATs frozen on DCASE 2020 in arXiv 2603.04605 (above).
  - FISHER: weaker than BEATs on DCASE 2020 in the ECHO paper (0.705 against 0.743). Our loader failed in notebook 15.
  - CED: no published ASD result that would make it a candidate.
  - ECHO: already part of MIC's system.
- **BEATs (both):** our forward pass of notebook 21 (filter banks in float32, encoder in float16), last layer.
- **EAT, as on its model card:**
  - The waveform minus its mean.
  - Kaldi filter banks: 128 mel bins, 25 ms Hanning window, 10 ms shift, no dither, HTK-compatible, no energy.
  - Zero-padded to 1,024 frames, then normalised as (x + 4.268) / (2 × 4.569).
  - `extract_features`, last layer, without the CLS token.
  - Training crops are padded to a multiple of 16 frames if the model accepts other lengths; otherwise to 1,024 frames.
- **Frozen:** evaluation mode, float16 autocast, whole clip.
- **Smoke test, before any download.** A new encoder is dropped from the run, and the failure is reported as our failure, if any of these fail:
  - it loads;
  - its tokens have the stated grid;
  - they are finite;
  - for fine-tuning: LoRA attaches to the stated number of layers, the output is unchanged with LoRA at zero, and a gradient reaches LoRA.

## Band views of a clip (for every encoder)
- The last-layer patch tokens are arranged as time × frequency × width.
  - The frequency index of each token is found by gate GB below, not assumed.
  - Only the time patches that start inside the clip are pooled. For EAT that excludes the padding at the end.
- **Per frequency row j (8 rows), the band vector is pooled over time in one of three ways:**
  - **mean** (BEAM's Tmean);
  - **max**, element-wise (AdaBEAM's Tmax);
  - **RDP**, as defined in arXiv 2603.04605:
    - d_t = ‖x_t − mean(X)‖₂;
    - d̂_t = d_t / max d;
    - w_t = (1 + d̂_t)^γ / Σ (1 + d̂)^γ;
    - RDP = Σ w_t x_t;
    - γ = 4, MERL's value. The paper's encoder-agnostic γ = 10 is not computed, to save local disk.
- **The global view** is the mean over all tokens (our P), or the head output F for a system scored on F.

## Scorers
- **Global LDN1** (notebooks 20–21): min over references y of d(x, y) / d(y, y₁), with cosine distance.
- **BEAM:**
  - Per band, LDN1 on the band vectors (K = 1, cosine distance, as BEAM's main experiments).
  - The clip score is the mean over the bands.
- **AdaBEAM:** the mean of the BEAM score on the mean view and the BEAM score on the max view.
- **Multi-model systems** (the same algebra as notebook 21):
  - **Seeds:** per band, the seeds' unit band vectors are concatenated and divided by √(number of seeds). That averages the cosine similarities.
  - **ECHO (EF on MIC):**
    - Under a band scorer, ECHO's own seven band vectors (its utterance output: the CLS of each band, 7 × 384) form a second component. ECHO has no time axis, so it is the same in every view.
    - The system score is the mean of the BEATs component's score and the ECHO component's score.
  - **Fusion of two systems:** the mean of their scores. Both are LDN ratios, so they share a scale.
- **Splits, folds and the score S:** as notebook 21.
  - S is the mean over the 8 development machines (pumps, fans) of (A + B+10)/2 at −6 dB.
  - Confirmation is the same over valves and slide rails.
- **Keep rule:** a candidate's development score must be ≥ the current system's + 0.01, and its confirmation score ≥ the current system's − 0.01.

## Stages and decisions (in this order, per track; each candidate is the current system with one change)
### Stage 1 · D2′ · N2N after MWF on float streams (ARR)
- **The denoiser:** notebook 20's N2N training, unchanged (one per fold and type, input MWF₋₆(x₋₆), target MWF₋₆(x₆), fold training clips), but on float16 MWF streams.
- **The candidate:** frozen BEATs LDN1 on N2N-MWF. Model k reads denoiser k, as MIC+.
- **The reference:** frozen BEATs LDN1 on MWF (0.9442 / 0.8835).
- **If D2′ passes:**
  - ARR's recipe (R0 + PD, seed 0, scored on P) is retrained on N2N-MWF. The student reads denoiser k's output; PD's teacher reads MWF₋₆(x₆), as in notebook 21.
  - It then faces ARR's current system with the keep rule. If it passes, ARR's input and system change for the rest of the notebook.
- If D2′ fails, ARR stays on MWF.

### Stage 2 · Scoring, on each track's current system
| Step | Candidate |
|---|---|
| SC1 | BEAM (mean view) |
| SC2 | AdaBEAM (mean and max views) |
| SC3 | RDP (γ = 4) in place of the mean view, in whichever band scorer is current; BEAM with RDP if neither was kept |

- **The tokens:**
  - ARR and MIC: tokens of notebook 21's fine-tuned models (LoRA on).
  - MIC+: frozen BEATs.
  - Model k reads its own fold's input, as before.
- **The frozen back-end:** after this stage, MIC+'s scorer is "the frozen back-end" for stage 3.

### Stage 3 · Encoder screen, frozen (reported for all tracks; one decision)
- Every encoder that passed its smoke test is scored, frozen, on each track's input, under the frozen back-end.
  - MIC+ uses notebook 20's denoisers with the fold rule.
  - ARR uses MWF, or N2N-MWF if stage 1 changed it.
- **N0 (MIC+ only, whose system is frozen):** the new encoder with the highest MIC+ development score, against MIC+'s current system, with the keep rule.
- **Selection for fine-tuning:**
  - The new encoder with the highest development score averaged over the three tracks, among those whose LoRA checks passed.
  - Ties go in the table's order.
  - This is a selection, not a decision.
  - If Colab disconnects, a resumed session keeps the first session's selection, so that finished trainings stay valid.

### Stage 4 · The selected encoder, fine-tuned with each track's winning recipe
- **Recipes:**
  - **ARR:** R0 + PD, seed 0, scored on P.
  - **MIC:** R0 + XC, seeds 0–2, + EF with ECHO, scored on F.
  - **MIC+:** R0, seed 0, scored on P (BEATs's best fine-tuned MIC+ recipe in notebook 21).
- **Settings:** every setting as in notebook 21 (LoRA rank 16 and α 32 on the query and value projections of every layer, head of width 256, sub-centre ArcFace, 10 epochs, the λ_PD rule).
  - EAT's attention has one fused query-key-value projection. Its LoRA adds the low-rank update to the query and value parts only.
- **Scoring:** with the track's current scorer. The band views of the new models are built as in stage 2.
- **Decisions:**
  - **N1:** the new encoder's system against the track's current system. If it passes, it becomes the current system.
  - **N2:** the fusion of the current system and the new encoder's system (the mean of their scores) against the current system, after N1. If N1 passed, the fusion is with the replaced system.

### The final system of each track
The current system after stage 4. It is compared with notebook 21's final system in the report.

## Gates
- **G0 · reproduction** (any failure stops the run):
  - (a) The clip lists equal notebook 20's.
  - (b) Notebook 21's three systems, recomputed from its files, give its development and confirmation scores within 0.0005: 0.9672 / 0.9558, 0.8135 / 0.8238, 0.8377 / 0.9006. Notebook 20's frozen references do the same (as in notebook 21).
  - (c) Frozen BEATs on our streams against notebook 20's embeddings, first 32 clips per type: cosine ≥ 0.999 (as in notebook 21).
  - (d) Notebook 21's final adapters, reloaded, reproduce its saved embeddings (P for ARR, F for MIC), first 32 clips per type: cosine ≥ 0.999.
- **GB · band layout,** for each encoder:
  - **The default layout is read from the source code.** BEATs flattens its patch grid as time × frequency, with frequency fastest. EAT's code does the same (`PatchEmbed_new`: flatten, then transpose, on a 64 × 8 grid). The Hugging Face port loads the same weights.
  - **The check.** A filter-bank input with extra energy in the top 16 mel bins over the middle third of the clip is compared with the same input without it. So is one with extra energy in the bottom 16 bins.
    - Under each candidate layout (frequency fastest, or time fastest), the change is averaged per frequency row.
    - A layout is confirmed when the top and bottom inputs peak at the two end rows (one each), and each peak is at least 1.5 × the mean of the other rows.
  - **What the notebook uses:**
    - If only the other layout is confirmed, it uses that one and reports it.
    - Otherwise it uses the source-code layout, and reports the check.
- **G1 · learning:** a training's last-epoch mean loss must be below its first. If it fails, the candidate is "not decidable", and the current system stays.
- **D2′ · the denoiser learned:** each N2N's final validation loss must be below doing nothing, as in notebook 20 (G3). If not, D2′ is not decidable.

## Reported, no decision
- **Tables:** every system, scorer and view, per track, type, split and machine.
- **Scoring variant:** the global scorer on P for systems scored on F. This separates "bands" from "P against F".
- **Encoder screen:** all four encoders under the global scorer, BEAM and AdaBEAM.
- **Seed spread** of the new encoder on MIC.
- **Against the yardsticks:**
  - T1: Split A against arXiv 2309.07500. Like for like only on MIC.
  - T2: ARR B+10 against the 6 dB targets.
  - Notebook 21's final systems, side by side.
- **Run details:** checkpoint hashes, Hugging Face commits, the GB indices, training logs, λ_PD, run times.

## Expected (written now; not decisions)
- BEAM raises S on every track, most on the frozen track (MIC+), as in the BEAM paper. AdaBEAM adds most on the fine-tuned tracks.
- RDP adds little (MERL: +0.65 points).
- Frozen, no new encoder beats BEATs iter3+ (AS2M) by 0.01. BEATs iter3 is close to it, and EAT is below.
- Fine-tuned, the selected encoder is close to BEATs. Fusion (N2) is the likeliest pass on MIC, where seeds and ECHO already helped.
- D2′ fails on development again. Its confirmation gain stays.

## Caveats written now
- **Many comparisons:**
  - up to 3 scoring decisions per track;
  - N0, N1 and N2;
  - D2′ and its follow-up.
  - The 0.01 bar and confirmation reduce chance passes but do not remove them.
  - MIC's seed range in notebook 21 (0.0112) is larger than the bar.
- **Selection bias:**
  - N0 picks the best of up to three encoders on development and is then judged on development.
  - Confirmation is the only guard.
- **Untuned settings:** γ, K = 1, equal fusion weights, the band count (the encoders' own 8 rows), and every training setting copied from BEATs to EAT.
  - MERL fine-tuned with LoRA rank 64 on query, key and value, for 25 epochs. We keep notebook 21's settings so that the encoders are compared under one recipe.
- **EAT features:**
  - arXiv 2603.04605 thresholded EAT-large's features before mean pooling and reported a large gain from it.
  - We use the model card's features without that step, so frozen EAT may be underrated.
- **DCASE 2020 compliance of MIC:** BEATs (2022), ECHO (2025) and EAT (2024) are all outside DCASE 2020's list of allowed models. T3 (notebook 23) is a post-challenge comparison and will say so.
- **Remote code:** EAT loads with `trust_remote_code`. Its commit is recorded, and its code runs in the user's Colab session.
- **Compute:** 10 trainings (12 if D2′ passes) and about 36 passes of an encoder over all clips. The notebook times one step of each encoder before any download and prints its estimate. The expectation is 3–5 h on Colab G4.

## Next (not part of this notebook)
- **Notebook 23:** external judgement with each track's final system frozen. That means the MIMII per-level table (T1), the DCASE 2020 evaluation set scored once on MIC (T3), and confirmation on valves and slide rails.

## Sources
- MERL, "The MERL Systems for DCASE 2026 Challenge Task 2", TR2026-100 (https://www.merl.com/publications/docs/TR2026-100.pdf).
- P. Saengthong, T. Shinozaki, "Sub-Band Spectral Matching with Localized Score Aggregation for Robust Anomalous Sound Detection", arXiv 2603.13749 (BEAM, AdaBEAM).
- K. Wilkinghoff, S. Yadav, Z.-H. Tan, "Temporal Pooling Strategies for Training-Free Anomalous Sound Detection with Self-Supervised Audio Embeddings", arXiv 2603.04605 (RDP, GeM; encoder comparison).
- DCASE 2026 Task 2 description, arXiv 2606.01578; technical report 58 (Zhou et al., EAT with LoRA and band memory banks).
- EAT: arXiv 2401.03497; https://github.com/cwx-worst-one/EAT; Hugging Face `worstchan/EAT-base_epoch30_pretrain`, `worstchan/EAT-large_epoch20_pretrain`.
- BEATs: arXiv 2212.09058; checkpoints from https://github.com/microsoft/unilm/tree/master/beats.

Anything computed outside these rules is labelled "looked after".

## Amendment 1 · Notebook 22b (written 2026-11-10, after notebook 22's results and before notebook 22b was built)

### Why
- Notebook 22 ran stages 1 and 2. Stages 3 and 4 did not run, because none of the three new encoders loaded (`N22_outputs_2026-11-10.md`):
  - **EAT-base and EAT-large:** transformers 5 needs an attribute that `post_init()` sets, and EAT's remote code does not call it. This was our loader's failure.
  - **BEATs iter3:** neither the authors' link nor the Azure copy downloaded.
- Notebook 22b runs stages 3 and 4 as written above, unchanged, with the loaders fixed.

### Where notebook 22b starts (notebook 22's final systems)
| Track | Input | System | Development | Confirmation |
|---|---|---|---|---|
| **ARR** | MWF | notebook 21's fine-tuned BEATs (R0 + PD), BEAM | 0.9794 | 0.9652 |
| **MIC+** | N2N-CH0 | frozen BEATs, BEAM | 0.8302 | 0.8491 |
| **MIC** | CH0 | notebook 21's system, unchanged (global LDN1 on F + ECHO) | 0.8377 | 0.9006 |

- Stages 1 and 2 are not run again, and their decisions stand. ARR stays on MWF (D2′ failed).
- The frozen back-end for the screen is BEAM (stage 2's result on MIC+).

### Loader fixes (the smoke tests are unchanged and still run before any MIMII download)
- **EAT (both):**
  - The repository is downloaded first, and its commit is recorded.
  - `from_pretrained` is tried first.
  - If it fails, the model is built from the repository's own configuration and code, and the weights file of the same commit is loaded into it.
    - Every trainable parameter of the model must come from the file. If one does not, the encoder is dropped.
    - Fixed tensors (buffers, and parameters without gradient such as a sin-cos position table) may come from the model's own construction. They are listed.
    - Keys of the file that the model does not have are listed and not used, as `from_pretrained` does.
  - The reason `from_pretrained` failed and the way the model was loaded are recorded.
  - Checked before building, on a small remote-code model that fails the same way under transformers 5.22.1: the fallback loads it and reproduces its saved output exactly.
- **BEATs iter3:**
  - The authors' link and the Azure copy are tried first, as before.
  - **A third source:** the copy in PretrainedSED's release v0.0.1 (`BEATs_ssl.pt`, F. Schmid et al.). Its README gives the authors' iter3 link as the source.
  - The file is loaded with `weights_only=True`, so only tensors and plain values are read and no code in it runs. A file that does not load that way is not used.
  - The file holds weights only. The architecture is the configuration of our iter3+ checkpoint, with the classifier off; the authors' BEATs checkpoints share one base architecture. So its dropout and layer-drop are ours, as in notebook 21's fine-tuning.
  - Every trainable weight of the model must be in the file. A common prefix of the file's keys is removed.
  - **Recorded:**
    - its SHA-256;
    - its relative distance from our iter3+ weights (a copy of our checkpoint would give 0).
  - **Provenance caveat:** this copy cannot be checked against the authors' file, which no longer downloads. Settings that have no weights, such as deep norm, come from our checkpoint.
  - Once one BEATs iter3 file is in Drive, a resumed session uses it and downloads nothing.
- No file that notebook 22b downloads is loaded with `weights_only=False`.

### Gates
- G0(a)–(d) as above.
- **New, G0(e):** notebook 22's final systems, recomputed in this session, give its numbers within 0.0005: 0.9794 / 0.9652, 0.8302 / 0.8491 and 0.8377 / 0.9006.
  - Their band views are recomputed from notebook 21's adapters and notebook 20's denoisers.
  - A failure stops the run.
- D2′'s gate does not apply, because D2′ is not run.

### Outputs
- The same Drive folder (`P04_NewEncoders`), with new names: `new_encoders_22b.json`, `new_encoders_22b_steps.png`, `new_encoders_22b_final.png` and `p04_n22b_results.zip`.
- The selection is kept in `selection_22b.json`.
- Also reported: the final systems against notebook 22's (the start) and notebook 21's.

### What does not change
Everything else: the encoders and their preprocessing, the smoke tests, GB, the screen, N0, the selection, stage 4's recipes and settings, the keep rule, G1, the reported items and the caveats.

### Compute
- **The screen:** four passes per encoder that loads (MWF, CH0, and N2N-CH0 once per fold).
- **G0(e):** four passes (notebook 21's ARR model on each fold, frozen BEATs on N2N-CH0 on each fold). The last two are reused by the screen.
- **Stage 4:** 10 trainings of the selected encoder. Cell 11 prints the estimate for each candidate.

### Expected (written now; not decisions)
- As above: frozen, no new encoder passes N0; fine-tuned, N1 fails, and N2 is the likeliest pass, on MIC.
- BEATs iter3 is the likeliest selection.

### Caveat
- **More comparisons:** notebook 22's ten decisions (D2′ and nine scoring steps) are followed here by N0 and up to six N1 and N2 decisions.
