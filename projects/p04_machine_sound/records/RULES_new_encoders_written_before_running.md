# P04 · Do the newest machine-sound models beat BEATs in factory noise? Rules written before running

Written 2026-11-08, after notebook 13 and before any FISHER or ECHO embedding of any MIMII clip existed.

## Why
- Every sound result so far uses BEATs (2022) frozen, with the nearest-neighbour scorer B1.
- Two newer models were built for machine signals, and both are open (MIT):
  - **FISHER** (Tsinghua and partners; arXiv 2507.16696; IEEE TII 2026);
  - **ECHO** (arXiv 2508.14689; ICASSP 2026).
- Both split the spectrum into frequency bands and model each band. Factory noise is not spread evenly over frequency, so a band model might hold up better in it. **That is the question here:** does a newer model beat BEATs on pumps and fans in heavy factory noise, with and without the microphone array?

## What had been seen before writing
- Every result of notebooks 1–14.
- **Published comparisons, all on DCASE data, single channel, no array:**
  - **ECHO paper, Table 2** (frozen models, kNN):
    - DCASE 2020, the set built from MIMII and ToyADMOS: BEATs 0.743, FISHER-small 0.705, ECHO-small 0.722.
    - Mean over DCASE 2020–2025: BEATs 0.619, FISHER-small 0.610, ECHO-small 0.621.
  - **Wilkinghoff, Yadav & Tan 2026** (arXiv 2603.04605):
    - Among frozen models, BEATs gave the best training-free results.
    - EAT-large was worse on every DCASE set, even with the extra preprocessing its authors' pipeline needed.
- **So the literature expects no large gain.** This test asks whether that still holds in factory noise at −6 dB, with an 8-microphone array, on our splits.

## Not tested here
- **EAT:** the 2026 study above found it weaker than BEATs as a frozen model. It belongs with fine-tuning, which is notebook 16.

## Data
- MIMII (Zenodo 3384388, CC BY-SA 4.0), all 8 channels, at **6 dB and −6 dB only**:

  | File | MD5 |
  |---|---|
  | `6_dB_pump.zip` | a09ba6060c10fc09cd4c8770213b0b9f |
  | `-6_dB_pump.zip` | d20b783a0ff9c93d58f452f98c37b112 |
  | `6_dB_fan.zip` | 0890f7d3c2fd8448634e69ff1d66dd47 |
  | `-6_dB_fan.zip` | f02ae808a58d84b6815b7ec38ff30879 |

- 0 dB is left out to save download time. The decisions use −6 dB; 6 dB shows the quiet case.

## Front ends
The same as notebooks 11 and 13:
- **CH0:** microphone 0 alone.
- **BF:** delay-and-sum with the installation delays measured at 6 dB. They are read from notebook 11's and notebook 13's files in Drive, not measured again.

## Models (every choice fixed now)
- **BEATs (reference):**
  - The mean embeddings that notebooks 11 and 13 saved in Drive (`P04_ArrayNoise`, `P04_ArrayNoise_Fan`).
  - Not recomputed.
- **FISHER-small:**
  - Weights: Hugging Face `jiangab/FISHER-small-0723`. Its config gives 12 layers, width 384, 6 heads and band width 50 bins.
  - Input, as in the authors' README:
    - the clip minus its mean;
    - magnitude STFT with a 25 ms window and 10 ms hop (400 and 160 samples at 16 kHz), not centred;
    - natural log of (magnitude + 1e-10);
    - normalised as (x + 3.017344307886898) / (2 × 2.1531635155379805);
    - at most 1,024 frames.
  - Embedding: `extract_features`, i.e. the [CLS] output of every 50-bin band, concatenated. At 16 kHz that is 4 bands × 384 = 1,536 numbers.
  - Loading:
    - first with `transformers.AutoModel` (`trust_remote_code`), as on the model card;
    - if that fails, the authors' GitHub code (commit 60dd0c8, file hashes checked) with the same weights file.
- **ECHO-small:**
  - Weights and code: Hugging Face `yucongzh/echo-small-0824`, loaded exactly as on its model card: width 384, 12 layers, 6 heads, band width 32, shift 16.
  - The load must leave no model weight missing.
  - Input: the model's own `extract_features_from_audio` at 16 kHz. That is a magnitude STFT of 25 ms / 10 ms, a log, and its own normalisation.
  - Embedding: the utterance-level output, i.e. the [CLS] of every 32-bin band, concatenated. At 16 kHz that is 7 × 384 = 2,688 numbers.
- **All models:** float32, evaluation mode, no gradient, whole 10-second clip, no tuning of any kind.

## Scorer and splits
The same as notebooks 11 and 13:
- **B1:** cosine distance to the nearest normal reference embedding (k = 1).
- **Splits:**
  - A: a machine the model knows;
  - B: a new machine, with other machines of the same type as reference;
  - B+n: the same, plus n of its own normal clips (n = 1, 3, 10).

## Fusion (fixed now)
**"BEATs + X"** adds the two models' scores after putting each on a common scale:
- **Reference scores:** every reference clip's distance to its nearest other reference clip (leave one out).
- **Common scale:** each model's test scores minus the mean of its reference scores, divided by their standard deviation.
- **Fused score:** the sum of the two models' rescaled scores.

## Smoke test before any download
- Each new model is loaded and run on a synthetic 10-second signal. The output must have the expected length and be finite.
- A model that fails is dropped from the run, and the failure is recorded.

## Gates
- **G0 · the BEATs reference reproduces:** at −6 dB, Split A, the embeddings from Drive must give exactly:
  - pumps: CH0 0.8234, BF 0.8795;
  - fans: CH0 0.7255, BF 0.8200.
  - Otherwise the notebook stops.
- **G1 · a new model loaded properly:** its pump CH0 Split-A average at 6 dB must be at least 0.80. A model below that is excluded from the selection and reported as such. If both are excluded, E1–E4 are "not decidable".

## Selection on pumps, decisions on fans
- **Selection:**
  - Score: S = (Split-A average + B+10 average) / 2 on pumps, −6 dB, BF.
  - The new model with the higher S is selected.
  - If the two are within 0.001, the higher Split-A average wins; if still tied, FISHER.
  - **Pumps decide only the selection.** Their numbers are reported, not judged.
- **Decisions:** fans, −6 dB, never used for any choice. The BEATs numbers are notebook 13's.
  - **E1 · a newer model helps a known fan in heavy noise:** selected model, BF, Split-A average ≥ 0.8200 + 0.02 = 0.8400.
  - **E2 · a newer model helps a new fan with 10 clips:** selected model, BF, B+10 average ≥ 0.5898 + 0.02 = 0.6098.
  - **E3 · the array still adds on top of the newer model:** selected model, BF − CH0, Split-A average ≥ +0.01.
  - **E4 · the two models together help:** BEATs + selected, BF, Split-A average ≥ 0.8400.

## Reported, no decision
- Every model, front end, level, split and machine.
- Both fusions (BEATs + FISHER, BEATs + ECHO).
- The model not selected.
- Run times and the exact weight files used (SHA-256).

## Caveats written now
- **Settings:** each model is used the way its authors document it. Other settings, such as layer choice, pooling or scorer, might suit a model better. The claim is about these settings.
- **Machine types:** two (pump and fan). Selection uses pumps, so the pump numbers favour the selected model a little.
- **Pre-training data:** neither paper lists MIMII or DCASE among its pre-training data. We cannot rule out that AudioSet holds similar sounds, for any of the three models.
- **Untested code:** FISHER's and ECHO's code could not be run in our workspace before this run; only the rest of the pipeline was tested, with stand-in models. The smoke test catches a broken load before any download.

Anything computed outside these rules is labelled "looked after".
