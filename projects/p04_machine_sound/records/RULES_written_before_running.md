# P04 · MIMII pumps with a pre-trained audio model (BEATs) — rules written before running

Written 2026-11-06, before any BEATs embedding was computed.

## Why
- Our best MIMII family so far (M2, clip statistics + Mahalanobis) reaches an average AUC of 0.966 on Split A, but 0.577 on a pump it never heard (Split B). See `results/mimii/summary.json`.
- Recent anomalous-sound-detection systems (DCASE 2024–2026) score distances on embeddings from pre-trained audio models, most often BEATs, with k-nearest neighbours or Mahalanobis.

## Data
- **MIMII, 6 dB, pump:** channel 0 of all 4,205 clips in `6_dB_pump.zip` (MD5 a09ba6060c10fc09cd4c8770213b0b9f). The same clips, splits and test sets as `results/mimii/`, matched by file name.
- **MIMII DUE, development pump data:** `dev_data_pump.zip` (Zenodo 4740355, MD5 8c7090f0974336404070315a1e1445f4).
  - Embeddings are computed in the same Colab run, then **sealed**.
  - No score of any kind is computed on DUE until a separate rules file for the confirmatory test has been written.
  - **Amended 2026-11-06, before any DUE score existed:** that file is `RULES_finetune_written_before_running.md`. DUE development data is opened for development; the one-time confirmation moves to the DCASE 2021 evaluation set for pump (sections 03–05).

## Embedding (fixed now)
- **Model:** BEATs iter3+ pre-trained on AudioSet-2M, self-supervised and not fine-tuned. Official checkpoint `BEATs_iter3_plus_AS2M.pt` from the authors' page (github.com/microsoft/unilm/tree/master/beats, MIT licence).
  - Frozen: no training and no fine-tuning, so the result does not depend on any choice made on our test clips.
  - The checkpoint's SHA-256 is recorded in the output file.
  - **Amended 2026-11-07, before any embedding existed:** the authors' OneDrive link now refuses automatic downloads (Colab got an authentication error). The notebooks then try Microsoft's own Azure copy of the same file, the link that Salesforce's LAVIS library uses. Each output records which source was used, with its SHA-256.
  - In the first Colab run both failed (the Azure link also refuses access now). The file is then downloaded by hand from the authors' OneDrive link and placed in Google Drive; outputs record "placed in Drive by hand".
- **Input:** the full 10-second clip, channel 0, 16 kHz, with BEATs' own preprocessing (128 mel bands, 25 ms frames, 10 ms hop, its fixed normalisation).
- **Clip embedding (primary):** the mean over time of the last layer's output, 768 numbers.
- **Also stored, for looked-after analysis only:** the mean and the standard deviation over time of every one of the 12 layers.

## Families
- **B1 · BEATs + kNN:** cosine distance to the nearest normal training embedding (k = 1).
- **B2 · BEATs + Mahalanobis:** the same `Maha` class as M2 (standardise → PCA keeping 95 % of the variance → T² + SPE, each divided by its 99th training percentile).

## Splits and metrics
- **Split A and Split B:** exactly as in `results/mimii/RULES_written_before_running.md`, with the same test clips.
  - Split B trains on all Split-A training clips of the other three pumps, for both families.
- **Metrics:** AUC per pump and on average; true-positive rate at 5 % false positives.

## Decisions
- **BH1 · as good on a pump it knows:** if B1's Split-A average AUC ≥ 0.946 (M2's 0.966 minus 0.02, the margin used in MH2) → "a pre-trained model is at least as good as simple statistics on a pump it knows".
- **BH2 · better on a new pump:** if B1's Split-B average AUC ≥ 0.677 (M2's 0.577 plus 0.10, the margin used in MH3) → "pre-trained embeddings transfer to a new pump better than simple statistics". Otherwise → "not better on a new pump".
- **BH3 · commissioning a new pump with a few clips:**
  - Split B, with the first n normal training clips of the held-out pump (sorted by file name) added to the kNN memory, for n = 1, 3, 10 and 30. The test clips do not change.
  - Decision at n = 10 (100 seconds of sound): if B1's average AUC ≥ B1's Split-A average − 0.05 (the margin used in MH1) → "under two minutes of a new pump's normal sound is enough". Otherwise → "a new pump needs more than ten clips".
  - BH3 is decided only if B1's Split-A average is at least 0.80. Below that, the verdict is "not decidable: the model is weak even on a pump it knows".
    - Added before any real run: a test of the scoring code on random numbers showed that a weak model would otherwise pass BH3 trivially.
  - The whole curve over n is reported.

## Development, no decision on MIMII (the idea to confirm later on DUE)
- **"Fleet shape, pump zero":** B2 fitted on the other three pumps, then re-centred on the mean of the n clips of the new pump (n = 1, 3, 10, 30). The scaling, the PCA directions and the percentile weights stay those of the fleet.
  - This is the same idea as the physics residuals, whose "zero" is learned at commissioning.
  - **Prior art, found after this section was drafted and before any run:** shifting source embeddings by the target-domain mean ("shiftall", in a DCASE 2026 Task 2 technical report by Moradi et al.) and domain-wise normalisation of kNN scores (GenRep, Saengthong & Shinozaki, 2024). So this is not a new idea. It stays as a development baseline, not as our contribution.
- Other variants may be tried on MIMII as development (layer choice, k, pooling, fusion with M2). Every one of them is labelled "development".
- **The confirmatory test:** one method, frozen and written down in a new rules file before DUE is opened, then run once on DUE and compared with the DCASE 2021 baseline for pump. Only that test supports a claim.
  - Amended 2026-11-06 (see above): the one-time test runs on the DCASE 2021 evaluation sections 03–05 instead, as written in `RULES_finetune_written_before_running.md`.

## Caveats written now
- Split A at 6 dB is the easiest MIMII setting. BH1 sits near the ceiling.
- BEATs was pre-trained on AudioSet, which is general audio from online videos. We do not know how many pump sounds it contains.
- The margins 0.02, 0.10 and 0.05 are the ones fixed in the earlier MIMII rules, not values from a standard.

## Run
- **Embeddings:** notebook `P04_MIMII_BEATs_Embeddings_Colab.ipynb`, run by the user on Colab with a GPU. The MD5 of each output file is recorded.
- **Scores:** `code/p04_mimii_beats.py` in this workspace.

Anything computed outside these rules is labelled "looked after".
