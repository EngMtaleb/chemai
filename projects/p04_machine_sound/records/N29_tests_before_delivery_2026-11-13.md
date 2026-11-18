# P04 · Notebook 29 · tests before delivery (2026-11-13)

Nothing here is evidence about real data. Every test ran on the synthetic folders that notebook 28's tests left, on a CPU.

- **Notebook:** `P04_N29_Domain_Backends_DCASE2022_Colab.ipynb`, MD5 e95f1c74d92dfd406adfb72732a3453d (10 cells, 8 of them code).
- **Builder:** `notebooks/build_n29_domain_backends.py`, with the new cells in `notebooks/n29_src/`. It stops unless notebook 28 is the file that ran and the rules' numbers are notebook 28's saved ones.
- **Rules:** `RULES_domain_backends_2022_written_before_running.md`, MD5 43f1ee3741c915d98448cbb2c507c866, written before the notebook existed.
- **Checker:** `notebooks/check_n29_results.py`. **Tests and logs:** `results/mimii_beats/tests_n29/` (`run_all29.sh`, `logs/summary.txt`).

## What is reused from notebook 28, which ran on real data
- Its setup (two stops removed: for a GPU and for the BEATs folder), its rules cell, and its data cell with gates G0(a) and G0(b), to the byte.
- Its frozen system from saved files, its scoring functions, its evaluator functions (fetching by commit, the two hashes, the official script).
- **New here:** gates G-FILES and G-SAME, the three scorings, the spreads under each, the choice, the evaluation step for the chosen candidate, the picture.

## What was tested: 12 scenarios, 12 passed
- **A full run on the structured set** (864 clips, three types). Gate G-SAME reproduced notebook 28's scores on 432 test clips exactly. The rule adopted F_KM (0.6316 against 0.6131). The official evaluator agreed. The checker recomputed the folder and the unpacked zip: 13 checks, 0 failed each.
- **Run again:** the evaluation set was not scored again; the score files are untouched; the results are identical.
- **A later run that would choose another candidate** stops: "the evaluation set was already scored for F_KM". The score files are untouched.
- **A fused candidate adopted** (forced by the harness, to take that path): three score sets were written; the chosen files equal log(F)/σ_F + log(V)/σ_P of its parts' files; the official evaluator agreed on all three.
- **Nothing adopted** (forced): no evaluation score, no ground truth fetched, the results zip still written.
- **A full run on the plumbing set** (396 clips): the rule adopted F_DN; checker 13 checks, 0 failed.
- **The stops**, each with no evaluation score and no ground truth:
  - the hold-out file missing → G-FILES; notebook 28's result file missing → stop naming the folder;
  - one seed's view features replaced by another's → G-SAME; one frozen training's embeddings replaced → G-SAME; one type's ECHO embeddings replaced → G-SAME;
  - notebook 28's folder of another data set → G-FILES.
- **Self-checks inside the notebook:** NN and DN equal a slow, direct computation of the rules' formulas on random vectors.
- **Lint:** ruff's F and E9 rules are clean for the builder and the checker.

## What the tests changed
- A candidate of the frozen recipe alone wrote its scores twice (as "chosen" and as "frozen part"). It now writes one set.
- A folder from another data set stopped on an assertion inside reused code. It now stops with a G-FILES message.

## What was not tested
- **The real embeddings.** The saved files in Drive are read for the first time by this notebook. Gate G-SAME must reproduce notebook 28's 8,400 saved test scores before anything else is computed.
- **The run time.** k-means runs about 200 times on at most 990 clips each; I expect a few minutes, unmeasured.
- **Whether any of the three scorings helps.** That is the experiment. On the synthetic sets the results mean nothing: the encoders are two-layer stand-ins.
