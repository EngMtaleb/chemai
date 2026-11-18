# P04 · Notebook 20 · Single-microphone denoiser, N2N after MWF, and smarter scoring. Rules written before running

Written 2026-11-09, after notebook 19 and before notebook 20 ran. This is stage 2 of the program, as revised in `PROGRAM_full_written_before_running.md` (Revision 1).

## Questions (all at −6 dB, all four machine types)
1. **Single-microphone track** (microphone 0, the DCASE 2020 setting). Does a Noise2Noise denoiser trained on microphone-0 twins (N2N-CH0) beat raw microphone 0 (CH0)?
2. **Array track.** Does an N2N post-filter after MWF (N2N-MWF) beat MWF, the program's front end since notebook 19?
3. **Scoring.** Does any of these beat B1 on the chosen inputs?
   - k-NN with k = 4.
   - Local density normalisation with K = 1 (Wilkinghoff et al., TASLP 2026).
   - Segment-level nearest-neighbour scoring (1-second segments; aimed at short events such as valve clicks).

## Why
- **Single microphone.** The DCASE 2020 MIMII part is MIMII microphone 0 at −6 dB (notebook 19's side check). On one microphone our frozen pipeline is below the published MIMII numbers for every type.
  - A single-microphone denoiser trained from twins needs no clean sound and no second microphone.
  - Trained on other machines of the same type, it can be applied to new machines, including DCASE's evaluation machines.
- **N2N after MWF.** In notebook 19, N2N after MVDR tied MWF on development (0.9154 against 0.9174) and beat it on confirmation (0.9098 against 0.8466). That was seen after the results, so the natural next candidate, N2N after MWF, is tested here under a rule written now.
- **Scoring.** B1 (the nearest clip, averaged over 10 s) can wash out short events.
  - Local density normalisation was proposed for exactly the mismatch between reference sets, which is what our new-machine splits contain.
  - The authors recommend "only using a single sample to define the local density, i.e., K = 1".

## Data
- **MIMII zips at 6 dB and −6 dB for all four types** (8 zips; MD5s as in notebook 19's rules). The 0 dB zips are not needed.
- **From notebook 19's Drive folder `P04_TwinFrontEnds`:**
  - `stats_{type}.npz`: the MWF weights per machine (the −6 dB filter).
  - `emb_{type}.npz`: for gate G0.
- **Notebook 15's `P04_Encoders`:** the ECHO CH0 references for pumps and fans.
- **Splits:** A, B, B+1, B+3 and B+10, as in notebook 11.
- **Roles:** development is pumps and fans; confirmation is valves and slide rails.

## Inputs (front ends), all at −6 dB
1. **CH0:** microphone 0.
2. **N2N-CH0:** the notebook 19 post-filter network, trained on microphone 0.
   - Input: microphone 0 at −6 dB.
   - Target: microphone 0 of the same clip at 6 dB (independent noise; notebook 19's G1 found case I for every type).
3. **MWF:** notebook 19's −6 dB filter of each machine, read from its stats file.
4. **N2N-MWF:** the same network, trained on MWF outputs.
   - Input: MWF₋₆(x₋₆).
   - Target: MWF₋₆(x₆), the same filter applied to the 6 dB twin, so the machine part of input and target is identical.
5. **Reported only:** N2N after MVDR, from notebook 19's results.

## N2N training (as notebook 19, with two changes forced by the data)
- **Unchanged from notebook 19:**
  - Folds: fold 1 trains on IDs {00, 02}, fold 2 on {04, 06}.
  - Split A uses the machine's own fold's denoiser; B and B+n use the other fold's.
  - Network, loss, crops, optimiser, batch 32, gradient clip 5, and validation on every 20th training clip.
  - Seed 0 plus the fold number.
- **Changes:**
  - Inputs are −6 dB only, because the 0 dB zips are not downloaded.
  - **60 epochs** instead of 30, so the number of updates matches notebook 19's 30 epochs over two input levels.
- **G3:** each of the 16 trainings (2 inputs × 2 folds × 4 types) must end with validation loss below the identity loss. If a fold fails, that input is invalid for that type.

## Encoders and embeddings
- **BEATs** (iter3+ AS2M, float16 encoder, float32 filter banks, as notebook 19):
  - Clip embedding: the mean over all output tokens.
  - Segment embeddings: tokens averaged over the 8 frequency patches, then the time steps split into 10 near-equal consecutive groups (`np.array_split`), each averaged. This gives 10 × 768 per clip.
- **ECHO-small:** float32, clip embedding only (notebook 15's loader).

## Scorers (cosine distance 1 − cos; scorers that are a monotone rescaling of each other give the same AUC)
- **B1:** the distance to the nearest reference clip.
- **kNN4:** the mean distance to the 4 nearest reference clips.
- **LDN1:** min over references y of d(x, y) / d(y, y₁).
  - y₁ is y's nearest other reference clip.
  - The denominator is floored at 1e-6.
  - Wilkinghoff et al., ratio form with K = 1.
- **SEG-max** (BEATs only): each of the clip's 10 segments gets its distance to the nearest segment of any reference clip. The clip score is the largest of the 10.
- **SEG-mean** (BEATs only): the same, averaging the 10.

## Gates
- **G0 · reproduction.** Any failure stops the run.
  - (a) The clip lists equal notebook 19's.
  - (b) BEATs CH0 and MWF clip embeddings recomputed here give B1 AUCs within 0.02 per machine and 0.005 per type average of notebook 19's files, for every split.
  - (c) ECHO CH0 and MWF have cosine ≥ 0.999 with notebook 19's files on every clip. For pumps' and fans' CH0 the comparison is with notebook 15's files.
- **G3:** as above.

## Decisions
- **The score:** S_e(input, scorer) = the mean over the 8 development machines of (A + B+10)/2. S(input) with B1 is the mean of S_BEATs and S_ECHO. Confirmation uses the same score over valves and slide rails.
- **D1 · single-microphone input.** N2N-CH0 replaces CH0 if both hold. Otherwise CH0 stays.
  - S(N2N-CH0) ≥ S(CH0) + 0.01.
  - Confirmation(N2N-CH0) ≥ confirmation(CH0) − 0.01.
  - Both need G3 to pass for all four types.
- **D2 · array input.** N2N-MWF replaces MWF under the same two conditions against MWF. Otherwise MWF stays.
- **D3 · scorer, one per encoder.**
  - Each scorer's development score is the mean of its S_e on the two chosen inputs (D1 and D2).
  - Candidates: kNN4 and LDN1 for both encoders; SEG-max and SEG-mean for BEATs.
  - The best candidate replaces B1 if its development score is ≥ B1's + 0.01 and its confirmation score (same mean over the two inputs) is ≥ B1's − 0.01.
  - Ties go to the first in the order kNN4, LDN1, SEG-max, SEG-mean.

## Reported, no decision
- Every table: type × input × encoder × scorer × split, with per-machine values.
- **Single-microphone track against arXiv 2309.07500 at −6 dB** (Split A; pump 0.8977, fan 0.8778, valve 0.9794, slide rail 0.9441). Same input, one microphone, so this comparison is like for like. The split may still differ.
- N2N after MVDR from notebook 19 (B1), for reference.
- BEATs + ECHO fusion (rescaled sum, B1) for the four inputs.
- N2N training logs, download speeds, the audio-cache status and run times.

## Audio cache (for later stages)
- **What:** for each type, four int16 streams in Drive › `P04_AudioCache`: CH0 at −6 dB, CH0 at 6 dB, MWF₋₆(x₋₆) and MWF₋₆(x₆).
- **When:**
  - all four are written when Drive's free space is at least their size plus 5 GB;
  - otherwise only the two −6 dB streams;
  - otherwise none.
- **Use:** later notebooks read the cache before downloading.

## Expected (physics, written now; not decisions)
- **N2N-CH0** should help every type at −6 dB, but less than the array does. One microphone can separate machine from noise only by spectro-temporal pattern, not by direction.
- **SEG scoring** should help valves most, because their faults are short events.
- **LDN1** should help the B splits more than A, because it evens out density differences between machines.

## Caveats
- **Twins:** N2N-CH0 learns from the twins of training clips. In a plant that corresponds to a second recording of the same machine in other noise, or to training on other machines of the same type.
- **Many comparisons:** up to five scorers on two encoders can produce a chance winner. The 0.01 bar and the confirmation step reduce that but do not remove it.
- **Downloads:** Zenodo gave about 2.4 MB/s per file in notebook 19. This notebook downloads with several connections per file and falls back to one if that fails.
