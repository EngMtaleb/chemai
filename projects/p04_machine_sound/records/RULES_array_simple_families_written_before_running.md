# Rules · The 8-microphone array with the four simple families (notebook 34)

Written on 2026-11-17, before notebook 34 ran on any real file.

## Why
Notebook 11 showed that aiming the eight MIMII microphones at the pump helps the pre-trained model in
factory noise (at −6 dB: one microphone 0.8234, plain average 0.8495, aimed array 0.8795; average AUC of
four pumps). The deck shows that number on the microphone page. Families 1 to 4 were only ever run on one
microphone at 6 dB. So the deck cannot say whether the array helps them too. This notebook measures it.

Only MIMII has eight channels. Every DCASE recording is one channel, so the array cannot be tried on the
three public tests.

## What had been seen before writing
- Notebook 11: the three front ends with the frozen pre-trained model (the table above and its record).
- `results/mimii`: families 2, 3, 4 and loudness on one microphone at 6 dB.
- Nothing of families 1 to 4 at 0 or −6 dB, and nothing of them on the average or the aimed array.

## Data
MIMII, pump, Zenodo 3384388, the three zips of notebook 11 (same MD5s): 6 dB, 0 dB and −6 dB. All 4,205
clips at each level, all eight channels. Licence CC BY-SA 4.0 (Hitachi).

## Front ends (the code of notebook 11, unchanged)
- **CH0:** microphone 0 only.
- **AVG:** the plain average of the eight, no alignment (more microphones, no physics).
- **BF:** delay-and-sum toward the pump. The delays are measured once, on the normal training clips at
  6 dB (GCC-PHAT), and used unchanged at 0 and −6 dB.

## Families (the classes and settings of the pump runs and of notebooks 31 to 33, unchanged)
Log-mel: 64 bands, n_fft 1,024, hop 512 (`code/p04_mimii_features.py`).
| Family | Features | Detector |
|---|---|---|
| F1 · a limit on one number | the clip's loudness (mean log-mel) | distance from the training mean, in training standard deviations |
| F2 · distance from the normal cloud | 128 clip statistics | `Maha(0.95)`: PCA 95 %, T² + SPE |
| F3 · isolation | the same 128 numbers | Isolation Forest, 200 trees, seed 0 |
| F4 · autoencoder | 5 log-mel frames (320 inputs) | MLP 64-64-8-64-64, 50 epochs, seed 0, reconstruction error |

`F1_raw` (loudness as it is, louder = more anomalous) is reported beside F1, as on the pump page.

Split A of the pump runs: for each pump, test = the first N normal clips by name plus all N abnormal
clips; train = the other normal clips of that pump. One unit = one level, one front end, one pump:
3 × 3 × 4 = 36 units.

## Gates
- **G0a:** the delays measured here at 6 dB equal notebook 11's (largest difference at most 0.0625 sample,
  one step of the search grid).
- **G0b:** CH0 at 6 dB reproduces the pump runs per pump: F2, F3 and `F1_raw` within 0.002 AUC. F4 within
  0.02 (the network depends on the library version; the difference is reported).

## The number
Average AUC over the four pumps, for each level, front end and family.

## Questions, asked of each family at −6 dB (bars as in notebook 11)
- **Does the noise hurt?** CH0 at 6 dB minus CH0 at −6 dB at least 0.02. If not, the next two are not decidable.
- **Q1 · Does the aimed array help?** BF minus CH0 at least 0.02.
- **Q2 · Is it the aiming, not only more microphones?** BF minus AVG at least 0.01.

No expected number is written. The answers are printed as they come out.

## What is not done
- No setting is tuned. No family is changed after a result is seen.
- Family 5 is not run here. Its pre-trained model alone is in notebook 11 and is printed beside the
  result for reading. The fine-tuned system (expert, trainee, fusion) was never trained on these three
  front ends; that would be a separate, longer run.
- MIMII's labels are in the folder names, so this is not a blind evaluation; it is the same open split
  as the pump runs.
