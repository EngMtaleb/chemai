# P04 · Notebook 29 · rules: a scoring that knows the two domains (DCASE 2022 Task 2)

**Written 2026-11-13, before the notebook exists.** Notebook 28's evaluation result (70.63 %) stands whatever this notebook finds; a new result is reported beside it, not in its place.

## Why
- Notebook 28 measured the kept system as it is under a change of conditions.
- On its **development** score files (the evaluation set was not used for this), the frozen recipe separates normal from anomalous clips inside each domain, but the two domains' scores sit at different levels:
  - fan section 00: source normals against source anomalies 0.49, against target anomalies 0.17; the source normals score above the target normals in 99 % of pairs;
  - ToyTrain section 00: target normals against source anomalies 0.18; the target normals score above the source normals in 85 % of pairs.
- The official AUC of a domain sets that domain's normal clips against the anomalies of both domains, so a level difference costs directly.
- The scoring used so far (LDN1 against all 1,000 training clips of a section, 990 of the source and 10 of the target) does not treat the two domains apart. This notebook tests three scorings that could remove the level difference. **No model is trained.**

## Questions
1. On the development set, does one of three fixed scorings beat the system as it is by at least 0.01 of the official score?
2. If one does: what is its official score on the evaluation set, scored once?

## What is fixed
- Every embedding is notebook 28's, read from `P04_DomainShift`: the six trainings of the frozen recipe, ECHO, the view's three seeds, the view's hold-out.
- The recordings are downloaded again (the same 21 zips, MD5) only to rebuild the list of clips with their names, labels and domains, by notebook 28's own loader and gates G0(a) and G0(b).
- Metrics, the official evaluator and its pins, the table of 31 teams and its words: notebook 28's rules, unchanged.

## What is used that notebook 28 did not use
- **The domain of a training clip** (source or target), from its file name. The challenge gives it.
- The domain of a test clip is not used. No attribute is used.

## The three scorings
For a section, S is its source-domain training clips and T its target-domain training clips, as unit vectors; x is a test clip; d is the cosine distance.
- **NN (no normalisation):** the distance to the nearest training clip of either domain.
- **DN (each domain in its own units):** min( d(x, S) / m_S , d(x, T) / m_T ).
  - d(x, S) is the distance to the nearest clip of S; m_S is the median, over the clips of S, of the distance to their nearest other clip of S (floor 1e-6). The same for T.
  - With fewer than two clips in T, only the source term is used.
- **KM (the usual one in the literature for this task):** min( distance to the nearest of 16 k-means centres of S , distance to the nearest clip of T ).
  - k-means on the unit vectors (scikit-learn, 10 starts, random state 0); the centres are scaled back to unit length. With fewer than 16 clips in S, the clips themselves.
- **As it is (LDN1)** stays the reference.

## The candidates
For each scoring b of the three:
- **F_b:** the frozen recipe with scoring b: the mean of its two encoder parts' scores (each part: its three seeds' features with ECHO, as before).
- **K_b:** F_b fused with the view under the same scoring b on the view's features (three seeds, unit-normalised, concatenated), by form ZP.
  - σ_F: as before, from the additional-training clips after the first 600 of each evaluation section by name, scored against those 600. The 600 are source-domain clips, so only the source part of a scoring acts there.
  - σ_P: the view's held-out clips, each scored against its own section's training clips that were not held out.
  - Each spread needs at least 30 clips.
That is six candidates, in this order: K_DN, K_KM, K_NN, F_DN, F_KM, F_NN.

## Order of work, and gates
1. **G0(a), G0(b):** notebook 28's gates on the zips and the files.
2. **G-FILES:** every saved file of notebook 28 is there and has this session's number of clips.
3. **G-SAME:** the system as it is, recomputed here from the saved embeddings, gives notebook 28's saved scores for the kept system, the frozen recipe and the view: every development test clip and every evaluation test clip, within 1e-6 relative. This proves the rows are the same clips. No label is involved.
4. **The development set:** the official score of the six candidates and of the system as it is.
5. **The choice (rule):** the candidate with the highest development official score, the first in the order above on a tie.
   - It is adopted only if its development official score, rounded to four decimals, is at least 0.01 above the system as it is.
   - Otherwise nothing is adopted, the evaluation set is not touched, and notebook 28's result is the final one.
6. **The evaluation set, once, only for an adopted candidate:** its scores are written (and those of its frozen part and, for a K candidate, its view part) before the ground truth is fetched; then the metrics and the official evaluator, as in notebook 28.

## How the result is read
- The adopted candidate's official score, its word and its place by notebook 28's table, beside 70.63 %.
- Its source and target lines beside notebook 28's (0.8294 and 0.7000).
- **The evaluation set has now been scored twice in this program** (notebook 28, and here). The choice here rests on the development set only; the reason for trying these scorings came from the development set only. I had seen notebook 28's evaluation numbers per type before writing these rules; they did not enter the design.

## Expected, before running
- I expect DN or KM to raise the development score, mostly on fan, ToyTrain and ToyCar. I have no number.
- NN may do worse than LDN1.

## Caveats
- One development set chooses among six candidates; its 21 sections are other sections than the evaluation set's.
- The view's features under a nearest-neighbour scoring were weaker than the frozen recipe in notebook 28's development set.
- No decision threshold is set.

## Outputs
- In Drive: `P04_DomainShift/n29_backends/` with `domain_backends.json`, the development scores per clip, the evaluation folder, one picture, and `p04_n29_results.zip`.

## Compute
- No GPU is needed. About 20 to 30 minutes, most of it the download and check of the 12.4 GB (notebook 28 took 963 s for that).
