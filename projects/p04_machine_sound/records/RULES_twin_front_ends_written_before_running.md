# P04 · Notebook 19 · Array front ends built from the twins. Rules written before running

Written 2026-11-09, before any clip of valves or slide rails was read by us. No twin statistics, MVDR, Wiener or Noise2Noise output existed yet. This is stage 1 of `PROGRAM_full_written_before_running.md`.

## Question
At −6 dB, does a front end built from twin statistics beat delay-and-sum (DS, the BF of notebooks 11 and 13)? The test covers a known machine (Split A) and a new machine with 10 clips (B+10), with frozen BEATs and frozen ECHO, on all four MIMII machine types.

## Why
- **What went wrong in notebook 12.** MPDR lost to DS on pumps in every split at −6 dB.
  - Its covariance contained the machine itself, and its steering used delays only, with the machine about 50 cm from the array.
  - Together these cancel part of the wanted signal. That was the caveat named in notebook 12's rules.
- **What the twins offer.** MIMII holds every clip at 6, 0 and −6 dB.
  - The MIMII paper's recipe adds a noise segment, power-tuned to each SNR, to the same machine segment.
  - So at level l the covariance per frequency is D_l = R_s + κ_l R, with κ_l = 10^(−γ_l/10) known from the recipe.
  - Three levels give the machine's spatial covariance R_s and the noise's R separately. No noise-only recording is needed.
- **The recent literature.**
  - DCASE 2025 gave supplementary noise clips; DCASE 2026 gives a near and a far microphone. Several 2026 systems use the second signal as a noise reference (for example spectral subtraction with a per-machine noise-transfer function).
  - NA-SSL (arXiv 2608.00447) notes that denoising training is not available for ASD because clean machine sounds are missing.
  - Twins remove that obstacle: a Noise2Noise target (Lehtinen et al. 2018; for audio, arXiv 2104.03838) needs only a second recording of the same sound with independent noise.

## Data
- **MIMII.** Zenodo 3384388, public 1.0, CC BY-SA 4.0. All 8 channels, 16 kHz, 10 s per clip. The MD5s were read from the record page on 2026-11-09:

  | Type | 6 dB | 0 dB | −6 dB |
  |---|---|---|---|
  | pump | a09ba6060c10fc09cd4c8770213b0b9f | 488748295c3f60b25de07b58fe75b049 | d20b783a0ff9c93d58f452f98c37b112 |
  | fan | 0890f7d3c2fd8448634e69ff1d66dd47 | 6354d1cc2165c52168f9ef1bcd9c7c52 | f02ae808a58d84b6815b7ec38ff30879 |
  | valve | fe5fb7c337cd701b1d31dc641e621892 | 178478eb0d11c79080a35562bfdeee71 | fdfaf185fea61b21e11952a070a4ada7 |
  | slider | 838c2b3441858359c4704ef13a1b27ff | 4d674c21474f0646ecd75546db6c0c4e | 49913eda7d37f182cbf8ed5c984140e0 |

- **Machine IDs:** whatever the zips contain, expected id_00, id_02, id_04, id_06. The twins of a clip are the files with the same name in the three zips of its type.
- **Splits:** as in notebook 11.
  - Test: the first N normal clips by name plus all abnormal clips (N = the number of abnormal clips).
  - Training: the other normal clips.
  - B: the reference is the training clips of the type's other machines. B+n adds the machine's first n training clips by name (n = 1, 3, 10).
- **Roles:** pumps and fans are for development; valves and slide rails are for confirmation only.
- **DCASE 2020 Task 2 development data, pump only.** `dev_data_pump.zip`, Zenodo 3678171, MD5 90e7091ef722b7238a7f1009365779cd, CC BY-NC-SA 4.0. Used for the side check only.

## Statistics (per machine, per frequency bin, from all of its training clips at each level)
- **STFT:** 1024-sample frames, hop 512, square-root periodic Hann, centred with zero padding (perfect reconstruction).
- **Noise power per level:** κ = 0.25119 (6 dB), 1 (0 dB), 3.98107 (−6 dB), relative to 0 dB.
- **Covariances:**
  - D_l = (sum over frames of x xᴴ) / (number of frames).
  - R = (D_−6 − D_6) / (κ_−6 − κ_6), made Hermitian, eigenvalues floored at 1e-6 × the largest.
  - R_s = D_6 − κ_6 R, made Hermitian, negative eigenvalues set to 0. Built from the quietest level, where errors in R count least.
- **Model check (reported):** D_0 is predicted as R_s + R. The median relative Frobenius error over bins is reported per machine.
- **Twin noise correlation (reported):** ρ = Re tr(C − R_s) / tr R, where C is the cross-covariance of the 6 dB and −6 dB twins.
  - Expected ≈ 0 if the noise segments are independent, ≈ 1 if they are the same segment.
- **Calibration status.** These statistics are physical calibration of the array, like notebook 11's installation delays. For pumps and fans the delays are read from Drive. They use the machine's training clips and are used for every split.

## Front ends
1. **CH0:** microphone 0.
2. **DS:** installation delays (GCC-PHAT on the 6 dB training clips, exactly notebook 11's code), then delay-and-sum. For pumps and fans the delays are read from Drive.
3. **MVDR:** w = R⁻¹R_s e₀ / tr(R⁻¹R_s) (Souden, Benesty & Affes 2010, β = 0).
   - R carries diagonal loading of 1e-3 × tr(R)/8.
   - For a rank-one R_s this is distortionless: the output is the machine at microphone 0 plus residual noise.
   - The weights are the same at every level.
   - At bins where tr(R⁻¹R_s) ≤ 1e-9 × its largest value, w = e₀.
4. **MWF:** w_l = (R_s + κ_l R + loading)⁻¹ R_s e₀, the multichannel Wiener filter, separate for each level.
5. **N2N:** a learned single-channel post-filter on a linear base front end.
   - **Base:** DS or MVDR, whichever has the higher twin-meter SNR at −6 dB for the machine type (median over its machines).
   - **Training:** per machine type and fold, by Noise2Noise. Details are given below.
6. **Diagnostics** (pumps and fans, −6 dB, BEATs and the meter only):
   - **MVDRd:** MVDR's noise covariance R, with notebook 12's delay steering instead of R_s. It separates notebook 12's two causes of failure.
   - **MVDR10:** MVDR built from the first 10 training clips only, to measure the dependence on calibration size.

## Twin structure (gate G1, per type)
- **Measured:** on the first 50 training clips of each machine, microphone 0:
  - c_tw = corr(x₋₆, x₆);
  - c_d = corr(x₋₆ − x₆, x₀ − x₆).
- **Expected:**
  - independent noise segments ("I"): c_tw ≈ 0.40, c_d ≈ 0.11;
  - the same segment, rescaled ("S"): c_tw ≈ 0.80, c_d ≈ 1.00.
- **Classification on the type's medians:**
  - S if c_d ≥ 0.9;
  - I if c_d ≤ 0.3 and c_tw ≥ 0.2;
  - otherwise unclear.
- **If unclear:** the meter and N2N are skipped for that type. MVDR and MWF still run, because the covariance model does not depend on the twin structure.
- **Exact separation in case S:** the machine sound is s = (a₋₆x₆ − a₆x₋₆)/(a₋₆ − a₆), with a = 10^(−γ/20). The noise is x_l − s.

## Twin SNR meter (gate G1b, per type)
- **Clips and front ends:** the first 100 training clips of each machine, for CH0, DS, MVDR, MWF, MVDRd and MVDR10, at −6 and 0 dB.
- **Signal power:** the same filter is applied to x_l and to its 6 dB twin.
  - Case I: signal power per bin = Re Σ Y_l Z₆*, total power = Σ|Y_l|², noise = total − signal.
  - Case S: the exact s and noise are filtered.
- **Reported:** broadband SNR (bins 1–512) and the SNR per bin.
- **G1b:** CH0's median broadband SNR must fall within −6 ± 1.5 dB at −6 dB and within 0 ± 1.5 dB at 0 dB.
  - If it does not, the meter is invalid for that type and the N2N base is DS.

## N2N (fixed now)
- **Folds:** fold 1 trains on IDs {00, 02} of the type, fold 2 on {04, 06}.
  - Split A of a machine uses its own fold's denoiser.
  - B and B+n use the other fold's denoiser for every clip involved, so a new machine's clips never trained the denoiser that scores it.
- **Training clips:** the fold's Split-A training clips, except every 20th by name, which is held out for validation.
- **Pairs:**
  - Inputs: base(x₋₆) and base(x₀).
  - Target in case I: base(x₆), which carries independent noise, so the L2 optimum is the clean estimate.
  - Target in case S: base(s), the exact clean sound.
- **Network:**
  - STFT of 512 samples, hop 256, square-root periodic Hann.
  - Input: log₁₀(|Y|² + 1e-10), normalised per bin with the mean and standard deviation of 200 training crops.
  - Linear 257 → 256, ReLU, a 2-layer bidirectional GRU (256), then Linear 512 → 257 and a sigmoid. The output is a real mask M.
- **Loss:** per sample, Σ|M·Y − T|² / Σ|Y|². The weight depends on the input only, so the Noise2Noise optimum is unchanged.
- **Optimiser:** Adam, learning rate 1e-3 with cosine decay to 1e-5, 30 epochs (each a new random order of all training pairs), batch 32, 4 s random crops (the same offset for input and target), gradient clip 5. The seed is 0 plus the fold number.
- **G3:** the fold's last-epoch validation loss (−6 dB input, first 4 s) must be below the identity loss (M = 1). Otherwise that type's N2N is invalid.
- **Inference:** each fold's denoiser is applied to every clip of the type at −6 dB.

## Encoders, levels and scorer
- **BEATs:**
  - Checkpoint iter3+ AS2M, SHA-256 d43cbfad…cd34; the last layer, averaged over time.
  - Filter banks in float32. The encoder runs under float16 autocast on the GPU, for every front end in this notebook, so that all of them are compared at the same precision. G0 checks the effect against the float32 files.
- **ECHO-small:**
  - `yucongzh/echo-small-0824`, notebook 15's loader, float32, −6 dB only.
  - For pumps and fans, CH0 and DS come from notebook 15's files.
- **Levels:**
  - −6 dB: every front end with BEATs (pumps and fans also MVDRd and MVDR10). ECHO with CH0, DS, MVDR, MWF and N2N.
  - 0 dB: CH0, DS, MVDR and MWF with BEATs.
  - 6 dB: CH0 and DS with BEATs. For valves and slide rails they are computed here; for pumps and fans they come from Drive.
- **Scorer:** B1, the cosine nearest neighbour.
- **Splits:** A, B, B+1, B+3 and B+10.

## Gates
- **G0 · reproduction (pumps and fans).** Any failure stops the run.
  - (a) The clip lists of the zips equal notebook 11's and 13's.
  - (b) Our GCC-PHAT delays match the Drive installation delays within 0.07 samples, on every channel of every machine.
  - (c) BEATs float16 CH0 and DS at −6 and 0 dB, against the float32 Drive files, for every split: per machine |ΔAUC| ≤ 0.02, and per type average |ΔAUC| ≤ 0.005.
    - Float16 can reorder near-ties between clips. A pipeline error moves results far more: at −6 dB, going from CH0 to DS alone moved pumps by 0.056.
  - (d) The Drive files give the recorded Split-A averages exactly:
    - BEATs, pump: CH0 0.8234, BF 0.8795;
    - BEATs, fan: CH0 0.7255, BF 0.8200;
    - ECHO BF: pump 0.8317, fan 0.8564.
  - (e) Our ECHO DS embeddings of the first 16 clips of each pump and fan at −6 dB have cosine ≥ 0.999 with notebook 15's.
- **G1, G1b and G3:** as above.

## Decisions (−6 dB)
- **The score:** per front end f and encoder e, S_e(f) = the mean over the 8 development machines of (A + B+10)/2. S(f) is the mean of S_BEATs(f) and S_ECHO(f).
- **F1 · a twin front end beats DS on development machines.**
  - f* = argmax of S over the candidates. MVDR and MWF are candidates. N2N is a candidate if G1 is I or S and G3 passes, for both folds of both pumps and fans.
  - A tie goes to the first in the order MVDR, MWF, N2N.
  - Pass if S(f*) ≥ S(DS) + 0.01.
- **F2 · confirmation on valves and slide rails.** The same score over their 8 machines. Pass if S(f*) ≥ S(DS) − 0.01.
- **The program's front end from here on:** f* if F1 and F2 both pass, otherwise DS.

## Reported, no decision
- Every table: per type, level, encoder, front end and split, with per-machine values.
- **R, the share of the noise loss won back** (Split A, BEATs): (A_f(−6) − A_CH0(−6)) / (A_CH0(6) − A_CH0(−6)), per type.
- **The cause of notebook 12's failure,** pumps (BEATs, A and B+10):
  - MPDR (notebook 12: 0.8713, 0.8437);
  - MVDRd;
  - MVDR;
  - DS (0.8795, 0.8985);
  - for fans, MPDR from notebook 13 (0.8228, 0.6780).
- Calibration size: MVDR10 against MVDR.
- The twin meter (broadband per machine, and per frequency bin averaged over each type's machines), the model check, and ρ.
- BEATs + ECHO fusion per front end at −6 dB (rescaled sum, as in notebook 15).
- **Against 2309.07500 at −6 dB (Split A, indicative):** pump 0.8977, fan 0.8778, valve 0.9794, slide rail 0.9441.
- **DCASE 2020 side check:**
  - Clips: the first 20 normal training clips of DCASE pump id_00 and its first 10 anomaly test clips.
  - For each clip and level, the highest correlation of its first 2 s with the first 2 s of every MIMII pump id_00 clip, microphone 0.
  - A level is identified if the median maximum is ≥ 0.99 for both groups. Otherwise the numbers are reported, along with a ±0.25 s lag search on each level's best candidate.
  - This fixes how notebook 23 reads DCASE 2020.
- Run times.

## Expected (physics, written now; not decisions)
- MVDR should beat DS on the twin meter at −6 dB, most where the noise is coherent. Notebooks 11 and 13 saw one coherent source in the noise at about +4.5 samples on microphone 4.
- MWF may help A but hurt B. It weights each machine's spectrum differently, as notebook 12 suspected for MPDR.
- In case S, the exact clean machine sound becomes available for every training clip. That would change what notebook 21 can train on.

## Caveats
- **Twins are a laboratory stand-in** for an installation recording with the machine off and on (see the program).
  - The calibration uses each machine's training clips for every split. That is the same premise as the installation delays, and MVDR10 shows how much it matters.
- **Precision:** BEATs runs in float16 here, so its numbers can differ from earlier notebooks at the third decimal. G0(c) bounds this.
- **The N2N denoiser is trained on normal sounds only.** It may soften anomalies along with the noise, and the AUC will show it.
- **The published MIMII numbers use another split,** so the comparison is indicative.
