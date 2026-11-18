# P04 · Notebook 28 · tests before delivery (2026-11-12)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU, in my workspace. This notebook has not run on a GPU, and no recording of DCASE 2022 Task 2 has been read.

- **Notebook:** `P04_N28_Domain_Shift_DCASE2022_Colab.ipynb`, MD5 1414291f7c887fcd1173d83913f528d5 (19 cells, 17 of them code; 2,664 lines of code; code SHA-256 0e9bbae23051772fee3e1d3299fe09366a4b1d6c12a19186e92796bd34288c8d).
- **Builder:** `notebooks/build_n28_domain_shift.py`, MD5 a72aa49c09602cac3e0b1a37686c2cbd, with the new cells as templates in `notebooks/n28_src/` (12 files, MD5 of them in name order 30b85bca59c0c00d0e85e72c29567058). Run again, it writes the same file. It stops unless the numbers it quotes are in the rules file and in notebook 27's saved run.
- **Rules:** `RULES_domain_shift_2022_written_before_running.md`, MD5 fc5874607591e58eab5f21caa3172ae3. Written before the notebook existed. One gate was changed while the notebook was being tested, before any run and before any recording was read (below).
- **Checker for a results folder:** `notebooks/check_n28_results.py`, MD5 068f8f000843d5e9c0e1fc17a1cc886a.
- **Test tools and logs:** `results/mimii_beats/tests_n28/` (`run_all28.sh` runs the plumbing set, `extra28.sh` the two longer runs; `logs/summary.txt` is the plumbing set's last result).
- **Which file the tests ran on:** this exact file, for every test below.

## The one change to the rules, and why
- **Gate G-DEV was first written on the development set's official score, at 0.50.** Its purpose is to catch a broken pipeline before the evaluation set is touched.
- **That bar was wrong for this data.** Under a change of conditions a working system's AUC (target) can fall below 0.50, so an official score near 0.50 does not show a broken pipeline. The autoencoder baseline's published AUC (target) is 34.81 on ToyCar and 23.35 on ToyTrain.
- **The tests showed it.** In the structured run below, the view learned (hold-out accuracy 0.91 to 0.95) and its development score was 0.489. The first gate would have stopped a working system there, after every training had finished.
- **The gate now reads the source domain:** each of the three systems needs a source-domain line (the harmonic mean of the 21 sections' AUC (source)) of at least 0.55 on the development set.
  - Random scores give 0.495 with a spread of 0.011 there (2,000 draws with the documented clip counts; the largest was 0.536).
  - Both baselines have more than 0.63.
- The change is marked in the rules where the gate stands. Nothing else in the rules changed.

## What was tested
1. **What comes from notebooks that ran on real data.** 1,896 of the notebook's 2,648 code lines are also lines of notebook 27, which ran on the RTX PRO 6000. The builder copies them and stops if a piece is not found whole.
   - **Six whole cells are notebook 27's to the byte:** the BEATs code, the checkpoint, the model pieces, LoRA, BEATs iter3, ECHO with gate G0(c). (They are notebook 23's cells.)
   - **Part 1's functions** (the training of one model, its pass over every clip, the DCASE lists) are notebook 27's lines. The class names, one comment and two printed phrases now say DCASE 2022.
   - **The whole cell of the view** is notebook 27's. Three things change: the two random-stream labels (27 → 28) and the class names.
   - **The readers, the scorers, the spread and the frozen system computed from saved files** are notebook 27's lines.
   - **The loader** is notebook 27's with seven checked substitutions for the 2022 file names: a section where a machine ID was; the label and the domain read from the name by one pattern.
   - **New here:** the data checks of gate G0(b), the official evaluator's metrics, the development tables, the evaluation step with the ground truth fetched from the evaluator's repository, the readings, the picture.
2. **The numbers the rules quote** (the builder stops on any mismatch):
   - the 21 MD5s, each once in the rules and once in the notebook, and the three Zenodo record numbers;
   - the evaluator's commit, the script's SHA-256 and the ground truth's combined SHA-256, recomputed from the clone;
   - the table of 31 teams, row by row; the two baselines; question 1's four lines; the 2.88 points between the seventh and the eighth team;
   - the baselines' development lines (52.61 and 55.94, with their source, target and pAUC lines), recomputed from the published per-type tables;
   - the autoencoder baseline's 64.23 and 45.13, against the evaluator's own example output;
   - the counts: 42,000, 8,400, 33,600, 39,900, 50,400 clips; 42 classes; 63 values;
   - the step time notebook 27 measured.
3. **The constants as a real run sees them** (`constants28.py`: the rules cell executed with the test mode off; 13 checks, 0 failed).
   - The 21 zips: each URL is its Zenodo record's, and each name and MD5 is in the rules' table.
   - The frozen recipe's 25 settings equal those saved by notebook 23's nine real DCASE trainings and notebook 27's six. The view's 18 settings and the reference commit equal notebook 25's and notebook 27's real runs.
   - The evaluator's pins are the cloned repository's, and the clone is at the pinned commit.
   - The ground truth has 200 rows per section, 100 anomalous and 100 of the target domain. Only counts were taken.
   - Too little room in Drive, or on the local disk, stops the run.
4. **The official evaluator itself, on the real ground truth with random scores.** No system's score was involved.
   - The pinned script ran unmodified under this workspace's pandas 3.0.5 and wrote its result file for 7 types × 3 sections. The notebook's parser reads that layout.
   - Its official score for random scores was 0.49996. It took 28 s for one set of score files.
5. **The plumbing set: 26 scenarios, 26 passed** (`run_all28.sh`).
   - **The fixtures:** three machine types (ToyCar, fan, valve), 396 clips with the challenge's folder and file names. Each section has 8 source-domain and 2 target-domain training clips and 12 test clips (3 of each kind in each domain). The fan's clips are a little longer than the 2 s the tests keep. The target domain runs 7 % faster and is noisier.
   - **The ground truth** is an archive laid out as GitHub archives the evaluator's repository: the real script and label and domain files for these clips, named as the repository names them (`section_03_0000.wav`, without "test").
   - **Stand-ins:** a two-layer BEATs for both encoders and a small stand-in for ECHO. The evaluator is the real pinned script, run unmodified.
   - **A full run from nothing** (195 s).
     - The official evaluator gave the same 39 values for each of the three score sets (largest difference 1.1e-16).
     - The checker recomputed the output folder, and then the unpacked results zip alone: 66 checks, 0 failed. That is 416 development values from 108 clips × 8 systems, 156 evaluation values from 324 scores, and both spreads.
   - **Run all again** (13 s). No epoch was trained, the evaluation set was not scored again, the score files and the lock were untouched, and the results are identical (3,835 values).
   - **A new local disk, Drive as it was** (13 s). The audio was downloaded again and nothing else was redone. The results are identical.
   - **A training of the frozen recipe killed after an epoch.** The next run resumed it at that epoch, did not repeat the finished training, and completed.
   - **A training of the view killed after a save.** After the resume, the models, the results and the per-clip files are identical to the uninterrupted run's.
   - **The session dies after the evaluation scores are written, before the ground truth is fetched.** Nothing of the evaluator's repository was on the disk at that point. The next run wrote no score again, fetched the ground truth, and finished with results identical to the uninterrupted run's.
   - **The stops.** In each, the run stopped with the message named, no evaluation score was written and no ground truth was fetched.
     - A clip shorter than the kept length, a two-channel clip, a clip at 8 kHz → G0(b), naming the file.
     - Other sections → G0(b). An evaluation test clip whose name carries a label and a domain → G0(b). An additional-training clip that is not normal → stop.
     - A training section with too few source-domain clips before its first target-domain clip → G0(b). A training section with no target-domain clip → G0(b). Evaluation clips not numbered from 0000 → G0(b).
     - A zip with another MD5 → "is not the published file". A store on the local disk built from other zips → G0(a).
     - No `P04_BEATs` folder, or no checkpoint in it → stop, naming what to put where.
     - A system below the gate's bar → G-DEV. Too few clips behind a spread → "a spread of form ZP cannot be formed".
     - A saved training of the frozen recipe, or a saved view, made with other settings → stop.
   - **A wrong hash for the evaluator's script, and for the ground truth** (two scenarios). The score files were already written and locked. The run stopped; nothing was unpacked and no result was written. With the right pins, the next run finished on the same score files, identical to the full run.
   - **Metadata entries in a zip** (`__MACOSX`, `._` files). They are skipped and counted, and the clip count is unchanged.
6. **Checks by hand in the notebook's own namespace** (`after28.py`, 17 checks; in the full run, the real-length run and the structured run).
   - The store holds the first samples of each clip, row by name (63 clips read again from the zips with Python's own reader; 21 were longer than the store).
   - The frozen recipe's list is every development training clip and the first clips of each evaluation section by name. The view's list is every training clip once. No test clip is in either. The hold-out is the rules' draw, the same in Drive.
   - The frozen recipe's scores equal a slow, direct computation from the saved embeddings (largest relative difference 1.3e-13).
   - The view's and the kept system's evaluation score files equal a direct computation from the saved feature files, with both spreads recomputed (difference 0.0).
   - The store holds no label and no domain for an evaluation clip. Every other clip carries the domain its name says.
   - One section's AUC (source) and AUC (target), by counting pairs against the fixture's own ground truth. A type's score and Ω as harmonic means, by hand.
   - The rules' words at their edges: 70.98 is "above the first team" and 70.97 is not; 66.83 is "with the leading teams" at place 7 and 66.82 is "in the upper half" at place 8; 54.02 and 54.01; a score below every team is place 32. The fusion's three words at ±0.01.
   - The final numbers are the official evaluator's when it ran and differs, and the notebook's otherwise.
   - The file times: the scores were written before the ground truth was kept and before the results. The evaluator that ran is the real script, to the byte.
7. **Real-length clips, all seven types** (10 s; 462 clips; 733 s for the run that resumed after Part 1).
   - 160,000 samples kept; 313 frames; a last kernel of 8 × 20; 42 classes; 1,162,112 parameters. Gate G0(d) passed with the rules' own sizes.
   - From cell 14 on, the test mode was off: gates G0(d) and G1 stop the run as they do on Colab, and the result file says it is not a test run.
   - **Gate G1 did stop it once.** With two epochs, seed 0's hold-out loss did not fall below its first epoch's, and the run stopped with "G1 failed for seed 0" (`logs/real_length_g1_stop.txt`). With eight epochs the three seeds passed and the run finished.
   - The by-hand checks passed. The checker: 95 checks, 0 failed.
8. **A structured synthetic set** (864 clips in three types; 4 epochs for the frozen recipe, 30 for the view at the test's learning rate 1e-3; 977 s).
   - The view learned: hold-out loss 24.91 → 0.70, 23.25 → 0.50 and 24.72 → 0.63 at the saved epochs (18, 21 and 27).
   - Development set: the view's source line is 0.7638 and its target line 0.3427; the stand-in frozen recipe's are 0.5869 and 0.7204. The official scores are 0.4890 (view), 0.6477 (frozen) and 0.6131 (kept).
   - Evaluation set: 0.5182 for the kept system, read by the notebook as "below the baselines", place 25. The fusion's word came out "costs" on both sets (−0.0636 and −0.0346).
   - These numbers say nothing about the real systems. The encoders are two-layer stand-ins and the "change of conditions" is a 7 % speed-up with more noise.
   - The official evaluator gave the same 39 values per score set. The checker recomputed 216 development clips × 8 systems and 648 evaluation scores: 66 checks, 0 failed.
9. **The lines only a real run takes.**
   - `real_mode28.py` (5 checks): with the stand-in model files, gate G0(c) stops a real run. It passes when the four hashes are the recorded ones, and stops again when one BEATs code file differs. Gate G1 stops a real run when a training of the frozen recipe did not learn.
   - The checker's real-run branch executes. On the synthetic real-length results it lists the eleven mismatches it should (not a real run, the gates, the zips, the pins, the documented counts twice, the two sets of settings, the minimum of 30 clips, the gate's bar twice) and nothing else.
10. **Lint:** ruff's F and E9 rules.
    - Clean for the builder, the checker and the test tools that run on their own.
    - In the notebook's code: ten uses of eight undefined names, all inside notebook 23's branches for its MIMII tracks and band views. With the DCASE lists those branches are never entered. Notebook 27 has the same ten.

## What the tests and the review changed
- **Gate G-DEV** (above).
- **A printed phrase.** Notebook 23's training function printed "from DCASE 2020's additional training data". It now says 2022.
- **The picture.** The right panel's scale was fixed at 0.3 and cut values below it; both scales now follow the values, and the note "the scale does not start at zero" appears only when it is true.
- **The checker accepts an evaluation clip name with or without "test" in it,** and a training name with or without attributes after its number. The notebook itself joins clips to the ground truth by the four-digit number.

## What was not tested
- **Any GPU.** The cells from notebook 27 ran on a GPU there. What is new here ran only on a CPU.
- **The real files.** Their folder layout, names, counts and lengths come from the task page and the evaluator's repository, not from the files. Cell 9 checks every file before any training (about the first 10 minutes) and stops with the reason if one is not as assumed. Three assumptions matter most:
  - the zips unpack to `<type>/train` and `<type>/test`;
  - a training name is `section_00_source_train_normal_0000_<attributes>.wav`, and names put the source domain before the target domain;
  - an evaluation test name ends in a four-digit number that is the ground truth's.
- **The downloads** (Zenodo, GitHub, Hugging Face) and Drive. The evaluator's archive is fetched from GitHub by commit; its script and ground truth are checked by hash before use.
- **The run time.** The estimate (about 4.5 hours on G4) scales the times notebook 27 measured there to 33,600 and 39,900 training clips and 50,400 clips in all. Drive use was not measured; I expect about 2 GB.
- **How the two recipes behave under a change of conditions.** That is the experiment.
