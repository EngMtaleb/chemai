# P04 · Pump sound in factory noise: the physics of the 8-microphone array, plus a 2026 training-free scorer — rules written before running

Written 2026-11-08. No multichannel MIMII audio, and no MIMII clip at 0 or −6 dB, had been read by us when this was written.

## Why
- **The recordings:** MIMII was recorded with a circular 8-microphone array (TAMAGO-03), 50 cm from the machine. The factory noise was recorded in real factories with the same array, then mixed in at 6, 0 and −6 dB.
- **The gap:** the MIMII baseline, the DCASE challenges and our notebooks so far all use one channel.
- **What recent work does about noise (2025–2026):**
  - The DCASE 2025 Task 2 systems that handled noise best used learned front ends trained on extra clean-machine and noise-only data: target-signal enhancement (Nagoya University) and noise-aware training.
  - The DCASE 2026 Task 2 winner (Fujimura et al., MERL, Nagoya and Mitsubishi Electric; arXiv 2608.00447) trains a "noise-aware" SSL model that also listens to a second, far microphone.
- **Physics offers a training-free way to use more microphones:** align the pump's wave front across the 8 microphones and add them. The pump adds up coherently; noise from other directions does not.
- **The scorer:** for training-free scoring we also add the 2026 state of the art, Wilkinghoff, Yadav & Tan (Aalborg; arXiv 2603.04605).
  - Relative-deviation-weighted generalized-mean pooling (RDP+GeM) of BEATs frames.
  - A nearest-neighbour score with local-density normalisation. On DCASE 2025 this beat that year's challenge winner with no training.

## What had been seen before writing
- Notebooks 1 and 3: channel 0 at 6 dB with B1, the same scorer as here. Average AUC: Split A 0.9904, Split B 0.7696, B+1 0.9687, B+10 0.9872.
- Nothing at 0 or −6 dB. No multichannel audio. We do not know the array geometry, so the delays below are estimated from the data.

## Data
- MIMII, pump, Zenodo 3384388. All 8 channels of all 4,205 clips at each noise level.
  - `-6_dB_pump.zip`, MD5 d20b783a0ff9c93d58f452f98c37b112.
  - `0_dB_pump.zip`, MD5 488748295c3f60b25de07b58fe75b049.
  - `6_dB_pump.zip`, MD5 a09ba6060c10fc09cd4c8770213b0b9f.
- Licence: CC BY-SA 4.0 (Hitachi).

## Front ends (what goes into BEATs)
- **CH0:** channel 0 only, as in notebook 1.
- **AVG:** the plain average of the 8 channels, with no alignment. This is the control: more microphones, no physics.
- **BF (delay-and-sum toward the pump):** y(t) = (1/8) Σ_m x_m(t + τ_m), applied as phase shifts on the FFT of the whole clip.
  - **The delays τ_m are measured once per pump, at installation:** by GCC-PHAT on that pump's Split-A training clips at 6 dB, the quietest level. The same delays are then used at every noise level, because the pump does not move when the factory gets louder.
  - **GCC-PHAT:**
    1. Sum the cross-power spectra conj(X_0)·X_m over all frames of all those clips (Hann window, 1,024 samples, hop 512).
    2. Divide by the magnitude (floor 10⁻¹²).
    3. Inverse FFT with 16× zero padding.
    4. τ_m = the peak within ±16 samples.
  - **No test clip is used.**
- **BFs (reported, no decision):** the same beamformer, but with delays estimated from the training clips at the same noise level as the test. This shows what happens without an installation measurement.
- **Why the delays come from 6 dB (changed before any real clip was read):**
  - The first version of these rules estimated the delays at the same noise level.
  - A test of the notebook on synthetic 8-channel data showed the failure: when the noise contains a coherent source, GCC-PHAT at −6 dB locks onto the noise instead of the machine. The delays collapsed towards the noise source's own delays.
  - Steering at the noise would defeat the purpose, so the installation measurement became primary and the same-level estimate became the reported variant.

## Embedding
- BEATs iter3+ AS2M: the same checkpoint and code as notebook 1, frozen, float32.
- Last-layer frames, then two poolings:
  - **mean over time:** primary, as in notebook 1;
  - **RDP+GeM:** frame weights w_t ∝ (1 + d_t / max d)^γ, where d_t = ‖x_t − mean‖₂; then (Σ w_t max(0, x_t)^p)^(1/p) with Σ w_t = 1. Values γ = 16 and p = 3, the paper's BEATs setting.

## Scorers
- **B1 (primary):** cosine distance to the nearest normal training embedding (k = 1), on the mean-pooled embedding. This is exactly notebook 1's scorer.
- **W26 (reported):** RDP+GeM embedding, Euclidean nearest neighbour, with the paper's normalisation:
  - score = min_Y [log ‖x − y‖ − α* · log d₁(y)], where d₁(y) is y's distance to its nearest other training embedding;
  - α* minimises the variance of the leave-one-out scores of the training embeddings, over a grid from −1 to 2 in steps of 0.02.

## Splits and metric
- Split A, Split B and B+n (n = 1, 3, 10) exactly as in `RULES_written_before_running.md` of this folder: the same test clips for every front end and noise level.
- Metric: AUC per pump and on average over the 4 pumps.

## Gates (checked first)
- **G0 · pipeline:** CH0 at 6 dB with B1 must reproduce notebook 1's Split-A average of 0.9904 within 0.005. Otherwise the notebook stops.
- **G1 · noise must hurt:** if CH0 at −6 dB with B1 is not at least 0.02 below CH0 at 6 dB (Split-A average), S1 is "not decidable: the noise does not hurt this pipeline".

## Decisions (−6 dB, scorer B1)
- **S1 · physics wins back what the noise took:**
  - R = (BF − CH0 at −6 dB) / (CH0 at 6 dB − CH0 at −6 dB), on Split-A average AUC;
  - pass if R ≥ 0.5, i.e. the beamformer recovers at least half of the loss.
- **S2 · it is the alignment, not just more microphones:** BF − AVG ≥ 0.01 (Split-A average AUC).
- **S3 · a new pump commissioned with 10 clips:** at B+10, BF − CH0 ≥ 0.02 (average AUC).
- In S1–S3, "BF" is the beamformer with the installation delays (6 dB).

## Reported, no decision
- Every front end at 6, 0 and −6 dB: Split A, Split B, B+1, B+3 and B+10, per pump and on average.
- The W26 scorer for every front end.
- **The delays:**
  - in samples and in µs, per pump and noise level;
  - the agreement of two halves of the training clips (odd and even, sorted by name);
  - the agreement across noise levels. Same pump and same position: physics says the delays should not change with the noise. A drift at low SNR would mean the estimate is locking onto the noise.

## Caveats written now
- **Digital mixing:** the noise was added to the machine recordings digitally. Machine and noise each keep one fixed spatial pattern; a real factory changes more.
- **Unknown geometry:** we do not know the array geometry. The delays come from the data, so a wrong estimate would steer the beam to a noise source.
- **The installation measurement is a privilege of this dataset:** in MIMII, the 6 dB and −6 dB files are very likely the same machine recordings with more noise mixed in.
  - So the 6 dB training clips are a quieter copy of the training data, from which only 7 delays per pump are taken. No test clip is used.
  - In a plant, the equivalent is measuring once at a quiet time, or from the known positions of the pump and the array.
- **Small array:** with 8 microphones close together, the beamforming gain lies mostly in the higher frequencies, where noise decorrelates between microphones.
- **Signal level:** AVG and BF change the level of the signal; BEATs' fixed input normalisation sees a slightly different level.
- **One dataset.** Split A at 6 dB is near the ceiling.

Anything computed outside these rules is labelled "looked after".
