# P04 · Notebook 27 · tests before delivery (2026-11-12)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU, in my workspace. This notebook has not run on a GPU, and no real ToyCar or ToyConveyor clip has been read.

- **Notebook:** `P04_N27_Toy_Types_Colab.ipynb`, MD5 ac2eb394cf784c9b7b595a9669a33d3a (19 cells, 17 of them code; 2,576 lines of code; code SHA-256 a1f8b1cc912db12f87e1ec69fea58866e28e967b7aa11318d49cf35d937e1894).
- **Builder:** `notebooks/build_n27_toy_types.py`, MD5 ec048fe7a0f91ebea80db203470b3d5f, with the new cells as templates in `notebooks/n27_src/` (12 files, MD5 of them in name order 4913a609604b502911fcdc53d1ac5324). Run again, it writes the same file. It stops unless the numbers it quotes are in the rules file and in the saved runs of notebooks 25 and 26.
- **Rules:** `RULES_toy_types_written_before_running.md`, MD5 e6235841387877c3df9226db4138b8dc. It was written before the notebook existed and has not changed since.
- **Checker for a results folder:** `notebooks/check_n27_results.py`, MD5 5d78abb7e5b2c07c8ff44bb45943c29a.
- **Test tools and logs:** `results/mimii_beats/tests_n27/` (`run_all27.sh` runs the whole plumbing set; `logs/summary.txt` is its last result).
- **Which file the tests ran on:** this exact file, for every test below.

## What was tested
1. **What comes from notebooks that ran on real data.** About 1,810 of the notebook's 2,560 lines. The builder copies them byte for byte and stops if a piece is not found whole.
   - **From notebook 23** (it ran on the RTX PRO 6000):
     - six whole cells, with only the number in their header changed: the BEATs code, the checkpoint, the model pieces, LoRA, BEATs iter3, ECHO. Notebook 24 ran the same six on a T4;
     - the training functions, the DCASE lists, the training of one model with its pass over every clip.
   - **From notebook 25** (117 minutes on the RTX PRO 6000):
     - the whole cell of the view. Two random-stream labels change from 25 to 27, and one comment;
     - the readers, the scorers, the spread, the frozen system computed from saved files (it now reads this notebook's folder), the training of one seed with its gate G1;
     - the loader, with eight checked substitutions: these types' machine IDs, the cut to 10 s, a check of every file, and what is recorded.
   - The builder also checks that notebook 25's readers and loader are notebook 23's, so "notebook 23's loader" is true of this notebook.
2. **The numbers the rules quote** (the builder stops on any mismatch):
   - the seven MD5s, each once in the rules and once in the notebook;
   - the yardstick table, row by row; T3′ = 0.9458 from the rank-1 team's six numbers; the 0.8523 the two toy types must average;
   - the kept system's four MIMII types, from notebook 25's saved results;
   - the published development numbers, the gate's bars, and the step time notebook 26 measured.
3. **The constants as a real run sees them** (`constants27.py`: the rules cell executed with the test mode off; 11 checks, 0 failed).
   - The seven files: each URL is its Zenodo record's, and each name and MD5 is in the rules' table.
   - The frozen recipe's 25 settings equal those saved by notebook 23's nine real DCASE trainings. The view's 18 settings and the reference commit equal notebook 25's real run.
   - The official evaluator is the file notebook 25's real run used. The hashes of BEATs iter3 and of ECHO's weights are those notebook 23's real run found.
   - Too little room in Drive, or on the local disk, stops the run. (My workspace has 6 GB free, so the second stop fired by itself the first time.)
4. **The plumbing set: 22 scenarios, 22 passed** (`run_all27.sh`).
   - **The fixtures:** 275 clips with DCASE's folder and file names. ToyCar's 149 clips are 2.2 s long and ToyConveyor's 126 are 2 s; the notebook keeps 2 s. A label file with the six machine-type sections. Lists of 120 clips for the frozen recipe and 144 for the view (115 train, 29 held out), in 13 classes. Six trainings of 3 epochs and three of 6 epochs.
   - **Stand-ins:** a two-layer BEATs for both encoders and a small stand-in for ECHO. The official evaluator is the real pinned file, run unmodified.
   - **A full run from nothing** (161 s).
     - The official evaluator gave the same 16 values for each of the three score sets (difference 0.0).
     - The checker recomputed the output folder, and then the unpacked results zip alone: 59 checks, 0 failed. That is 144 development values from 77 clips × 8 systems, 48 evaluation values from 162 scores, and both spreads.
   - **Run all again** (14 s). No epoch was trained, the evaluation set was not scored again, the score files and the lock were untouched, and the results are identical (2,352 values).
   - **A new local disk, Drive as it was** (14 s). The audio was downloaded again and nothing else was redone. The results are identical.
   - **A training of the frozen recipe killed after an epoch.** The next run resumed it at that epoch, did not repeat the finished training, and completed.
   - **A training of the view killed after a save.** After the resume, the models (the saved one and the last epoch), the results and the per-clip files are identical to the uninterrupted run's.
   - **The session dies after the evaluation scores are written, before the labels are read.** No label file had been fetched at that point. The next run wrote no score again, read the labels, and finished with results identical to the uninterrupted run's.
   - **The stops.** In each, the run stopped with the message named, no evaluation score was written and no label file was fetched.
     - A clip shorter than the kept length, a two-channel clip, a clip at 8 kHz → G0(b), naming the file.
     - Other machine IDs → G0(b). An evaluation test clip whose name carries a label → G0(b). An additional-training clip that is not normal → stop.
     - A zip with another MD5 → "is not the published file". A store on the local disk built from other zips → G0(a).
     - No `P04_BEATs` folder, or no checkpoint in it → stop, naming what to put where.
     - A system below the gate's bar on the development set → G-DEV. Too few clips behind a spread → "a spread of form ZP cannot be formed".
     - A saved training of the frozen recipe, or a saved view, made with other settings → stop.
   - **A wrong evaluator file.** The check is reported as not run, with the reason. The notebook's numbers stand and equal the full run's.
   - **Metadata entries in a zip** (`__MACOSX`, `._` files). They are skipped and counted, and the clip count is unchanged. None is expected in the real zips.
5. **Checks by hand in the notebook's own namespace** (`after27.py`, 14 checks; in the full run, the real-length run and the structured run).
   - The store holds the first samples of each clip, row by name. 42 clips were read again from the zips with Python's own reader; 23 of them were longer than the store.
   - The frozen recipe's list is every development training clip and the first clips of each evaluation machine by name. The view's list is every training clip once. No test clip is in either. The hold-out is the rules' draw, the same in Drive.
   - The frozen recipe's scores equal a slow, direct computation from the saved embeddings (largest relative difference 2.6e-13).
   - The view's and the kept system's evaluation score files equal a direct computation from the saved feature files, with both spreads recomputed (difference 0.0).
   - One machine's AUC, by counting pairs against the fixture's own label file.
   - The rules' words at their edges. In question 1, for example, 88.28 on ToyConveyor is "above the baseline" and 88.29 is "with the teams". T3′ flips between a toy average of 0.8521 and 0.8523. In the ranks among six, a tie shares the better rank.
   - The final numbers are the official evaluator's when it ran and differs, and the notebook's otherwise.
   - The file times: the scores were written before the labels were kept and before the results.
6. **Real-length clips** (ToyConveyor 10 s, ToyCar 11 s; 195 clips; 536 s).
   - 160,000 samples kept of ToyCar's 176,000; 313 frames; a last kernel of 8 × 20; 13 classes; 1,154,659 parameters. Gate G0(d) passed.
   - From cell 14 on, the test mode was off: gates G0(d) and G1 would have stopped the run as they do on Colab, and the result file says it is not a test run.
   - The by-hand checks passed. The checker: 59 checks, 0 failed.
7. **A structured synthetic set** (684 clips; every machine a tone with its own harmonics, anomalies of varying strength; 4 epochs for the frozen recipe, 30 for the view at the test's learning rate 1e-3; 803 s).
   - The view learned: hold-out loss 25.40 → 0.016, 28.06 → 0.008 and 23.99 → 0.020 at the saved epochs (22, 21 and 15).
   - On each type's mean, every system scores anomalies above normal clips. Development AUC: frozen recipe 0.6844 and 0.7344, view 0.5836 and 0.5490, kept system 0.6953 and 0.6844.
   - Evaluation AUC: kept system 0.7500 and 0.7083. Both of question 2's words came out by themselves: "helps" on ToyCar (+0.0122) and "costs" on ToyConveyor (−0.0625).
   - The checker recomputed 252 development clips × 8 systems and 504 evaluation scores: largest difference 0.0.
   - The same numbers came out on an earlier build of the notebook, to the last digit: the training code did not change between builds.
8. **The lines only a real run takes.**
   - `real_mode27.py` (5 checks): with the stand-in model files, gate G0(c) stops a real run. It passes when the four hashes are the recorded ones, and stops again when one BEATs code file differs. Gate G1 stops a real run when a training of the frozen recipe did not learn.
   - The checker's real-run branch executes. On the synthetic real-length results it lists the nine mismatches it should (not a real run, the gates, the zips, the two sets of settings, the minimum of 30 clips, the gate's bars twice, the evaluator's address) and nothing else.
9. **Lint:** ruff's F and E9 rules.
   - Clean for the builder, the checker and the test tools.
   - In the notebook's code: ten uses of eight undefined names, all inside notebook 23's branches for its MIMII tracks and band views. With the DCASE lists those branches are never entered.

## What the tests and the review changed
- **The development set is read only after the nine trainings.** A first build printed a development table after the view's first seed. The rules put the development set after the trainings, so that preview was removed.
- **A saved view made with other settings stops the run.** Notebook 25's code checks this only when it resumes a training; a finished one was accepted as it was.
- **The loader skips an archive's metadata entries** and counts them. It is a precaution: the same loader read the four MIMII types' zips of the same Zenodo records without meeting any.
- **The picture's layout** was fixed after the first rendering (a wide empty band under the title, and a legend that reached into the next panel).
- **One failure in the first full set came from the tests, not the notebook.** BEATs drops layers at random. With the two-layer stand-in, both layers are dropped in about 1 run in 400, and notebook 23's own check "a gradient reaches LoRA" then fails. It happened once. The test tool now fixes that random stream; the real model has 12 layers.

## What was not tested
- **Any GPU.** The cells from notebooks 23 and 25 ran on a GPU there. What is new here ran only on a CPU: the loader's cut, the hand-over from Part 1 to Part 2 (the encoders leave the GPU before the view is trained), the three score sets, the readings.
- **The real files of the two types.** Their clip counts, lengths, names and format come from DCASE's documentation, the reference code and the zip sizes, not from the files. Cell 9 checks every file in about the first 10 minutes, before any training, and stops with the reason if one is not as assumed.
- **The downloads** (Zenodo, GitHub, Hugging Face) and Drive.
- **The run time.** The estimate (1.5–2 hours on G4) uses the step times notebooks 23 and 26 measured there. ECHO's pass over about 22,000 clips was not timed.
- **How the two recipes behave on these types.** That is the experiment.
