# P04 · Notebook 26 · tests before delivery (2026-11-12)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU, in my workspace. This notebook has not run on a GPU.

- **Notebook:** `P04_N26_Second_View_MIMII_Levels_Colab.ipynb`, MD5 351803cad613caf6352f4caff62f1b42 (12 cells, 1,533 lines of code; code SHA-256 061433bc9a9bb32a4e3124da5ab20ba4df293db1a05156e1ee154d8707949da8).
- **Builder:** `notebooks/build_n26_second_view_mimii.py`, MD5 45fbdd220e10d8bc1eb747cbd70060c4. Run again, it writes the same file. It stops unless the numbers it quotes are in the rules file and in the saved runs of notebooks 23, 24 and 25.
- **Rules:** `RULES_second_view_mimii_written_before_running.md`, MD5 b3deacab0f7a80fc201e2b87e12f740a.
- **Checker for a results folder:** `notebooks/check_n26_results.py`, MD5 776a35ec25511bab5813d58edbbcef89.
- **Test tools and logs:** `results/mimii_beats/tests_n26/` (`run_all26.sh` runs the whole plumbing set; `logs/summary.txt` is its last result).
- **Which file the tests ran on:** this exact file, with one exception. The structured run (item 6) ran on the build before one sentence of the header cell was corrected. The code cells of the two builds are identical (the same SHA-256).

## What was tested
1. **What comes from notebooks that ran on real data.** The builder copies these parts byte for byte and stops if they are not found whole:
   - From notebook 25 (which ran 117 minutes on the RTX PRO 6000): the model, the log-mel front end, Noisy-ArcMix, the training loop with its saves and resume, the hold-out, the spread, the left-out LDN1. Two random-stream labels changed from 25 to 26, and one comment.
   - From notebook 23: the downloads, the WAV reader, the program's split, the scorers, the evaluation, the table of Zenodo files.
2. **The constants as a real run sees them** (`constants26.py`, the constants cell executed with the test mode off).
   - The 12 MIMII zips and the 4 additional-training zips: the names and MD5s that notebook 23's real run downloaded and verified.
   - The view's 18 settings and the reference commit: equal to notebook 25's real run.
   - The published table, notebook 23's level tables (300 values), and the two bars 0.9298 and 0.9668 from 0.9198 and 0.9768.
   - The rules' sizes: 28 classes; 21,583 clips; 20,503 train; 313 frames; a last kernel of 8 × 20; the minimum of 30 clips.
3. **The plumbing set: 14 scenarios, 14 passed** (`run_all26.sh`). The fixtures: 416 two-second clips per level at three levels, in 8-channel WAVs whose channels 1–7 hold constants; 168 additional clips; stand-ins for the four Drive folders. Lists of 392 clips in 28 classes, 313 train and 79 held out; 3 seeds × 6 epochs.
   - **A full run from nothing** (141 s). G0(a): 120 files. G0(c): 300 values, largest difference 0.0. The rule itself did not keep the system (0.5556 → 0.5590 against 0.5656 needed), so 6 and 0 dB were not read and no reading 3 was made. The checker recomputed 192 test-clip rows and 275 calibration rows: largest difference 0.0.
   - **The keep branch, forced** (168 s). The background downloads, the 6 and 0 dB audio, reading 3, the picture. The checker recomputed 576 test-clip rows and 825 calibration rows at three levels: largest difference 0.0.
   - **A second session with no audio reachable and an empty local disk** (12 s). Nothing was downloaded. The results (5,409 values), the models and the per-clip files are identical to the first session's.
   - **Stop and resume.** A run was killed right after the save at epoch 4 of the second training, then started again. The models (the kept one and the last epoch), the results (2,955 values) and the per-clip files are identical to the uninterrupted run's.
   - **The stops.** In each, the run stopped with the message named and wrote no result file.
     - One of the 120 files missing → G0(a).
     - A 6 dB embedding file in another clip order → G0(c) (largest difference 0.22). The same at −6 dB → G0(c) (0.083).
     - A zip with another MD5 → "is not the published file".
     - A zip whose clip names are not notebook 20's → G0(b).
     - Fewer clips than the minimum behind a spread → "ZP cannot be formed".
     - Training lists in Drive that are not this session's → stop. A hold-out in Drive that is not the saved models' → stop.
   - **No room for the background downloads** → the zips came in the foreground, and the results are identical to the keep run's.
   - **Two feature files that claim other weights** → both were computed again, and the results are identical.
4. **Checks by hand in the notebook's own namespace** (`after26.py`; in the full run, the keep run and the structured run).
   - No test clip is in the training lists, the hold-out or any spread. The held-out clips are outside every seed's training clips.
   - The frozen system's left-out scores equal a slow, direct computation.
   - The spreads, ZP, the reading with the −6 dB spreads carried, and ZB equal their formulas written by hand.
   - The features at 6 and 0 dB differ from the −6 dB features (in the two runs that read three levels).
   - The saved calibration files hold the clips and values the spreads rest on.
   - A fresh pass of the saved model over the audio gives the saved features, paper's scores and own-machine flags.
5. **Real-length clips** (10 s, the notebook's own path, one seed, two epochs at batch 8): 160,000 samples, 313 frames, a last kernel of 8 × 20, 28 classes, 1,158,514 parameters (874,524 + 280,406 + 3,584), as in notebook 25.
6. **A structured synthetic set** (seven harmonic machines per type in coloured noise, three kinds of anomaly, 896 clips per level at three levels; a weak stand-in for the frozen system; 2 seeds × 40 epochs with the test's learning rate 5e-4; 30 minutes).
   - The view learned: training loss 14.96 → 3.60 and 14.99 → 3.86; hold-out loss 25.95 → 6.02 and 25.08 → 6.18 at the saved epochs (26 and 27).
   - The rule itself kept the system: the stand-in's 0.5840 and 0.6387 became 0.6972 and 0.6954, against 0.5940 and 0.6287 needed. Reading 3 was then made.
   - The checker recomputed 768 test-clip rows and 2,091 calibration rows: largest difference 0.0.
   - The two models are identical to those of the same run on an earlier build: the training code did not change between builds.
   - **What this set showed, and what it changed.** At its "6 dB", the view by the paper's score reached 0.7520 on the development types, and by B1 against that level's references 0.9668; ZP gave 0.8965 and ZB 0.9903. The paper's score measures a clip against classes learned at −6 dB; B1 measures it against references of its own level. ZB was added to the rules as a reported-only reading because of this. It says nothing about real data: the fixture makes its levels by scaling the machine, and MIMII scales the noise.
7. **The lines only a real run takes** (`after26_real_mode.py`: cells 8 to 10 run again with the test mode off, in the structured run's namespace).
   - Gate G-VIEW stops the run when the view alone is below the baseline.
   - The rules' bars are used, and the note prints when the recomputed scores differ from the rules' quote.
   - The same decision, scores and reading 3 come out, and the result file says it is not a test run.
   - The checker's real-run branch executes: on synthetic results it lists the nine mismatches it should (the rules' sizes, bars and settings) and nothing else.
8. **Lint:** ruff's F and E9 rules, clean for the notebook's code, the builder, the checker and the test tools.

## What was not tested
- **Any GPU, for the parts that are new here.** The training code ran on a GPU in notebook 25. Three pieces are new and ran only on a CPU: the training audio gathered from two stores, the features computed per level (with the own-machine flag), and the lists.
- **Real data, Drive, the downloads from Zenodo.** Whether the frozen system's 120 files are still in Drive is not known; gate G0(a) says so in the first minutes, before any download.
- **The run time.** The estimate (2–3 hours on G4) uses notebook 25's measured step time and notebook 23's measured downloads on that machine. It was not measured for this notebook. A T4 was never measured for this model.
- **How the view behaves at 6 and 0 dB on real data.** That is the experiment.

## What the synthetic tests changed (stated in the rules' first paragraph; no real result existed then)
- Three reported-only items: the reading ZB, the look at the held-out clips at each level, and the calibration clips' scores saved in the results zip.
- With the calibration scores in the zip, the checker recomputes every spread from the clips it rests on, and ZP from those spreads. Before, it took the spreads as recorded.
