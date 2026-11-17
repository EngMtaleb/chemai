# P04 · Notebook 30 · tests before delivery (2026-11-14, fourth delivery)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU.

## Why four deliveries
- **The first delivery stopped on Colab at gate G0(b), in the data cell, before any training.** The rules assumed every clip to be 10 s or 12 s, from the development record's page. The first training clip of AutoTrash is 6 s.
- The seven development types had loaded as documented: 990 source and 10 target training clips and 200 test clips each; 12 s for ToyCar and ToyTrain, 10 s for the other five.
- **The change (marked in the rules):** every system still reads 10 s of a clip. A shorter clip is repeated from its start until it fills 10 s; a clip must be at least 1 s. This is the same for a type's training and test clips, and uses no label.
- **My own first fix was incomplete, and the tests caught it.** The loader accepted a short clip, but a second check in gate G0(b) still required 10 s; the plumbing set stopped on the fixture's short type. That check now uses the rules' minimum too.
- **The second delivery stopped in the same cell, on BandSealer:** its files are 24-bit PCM, and the reader (notebook 23's) accepted 16-bit files only. Again nothing had been trained.
  - The eight types before it had loaded: AutoTrash with 990 source and 10 target training clips and 200 test clips, all 6 s, repeated to 10 s as the changed rule says.
  - **The change (marked in the rules):** the reader accepts PCM of 16, 24 or 32 bits and 32-bit float. The store stays 16-bit, so a clip of more bits is rounded to 16. The sample format of every file read is now counted per type and saved.
- **The third delivery ran every training on Colab and stopped at gate G-DEV,** before the evaluation set was touched: the frozen recipe alone has a development source-domain line of 0.5433, and the gate asked 0.55 of each part.
  - That is weakness on this data, not a broken pipeline: the same recipe's target-domain line is 0.6322, its three parts show the same pattern, and the system as it is has 0.6604.
  - **The change (marked in the rules, with its date and the numbers known when it was made):** the gate reads the system as it is; its two parts' lines are reported and do not stop the run. Mohammed chose this over stopping without an evaluation number.
  - The trainings in Drive are kept: no setting changed.

- **Notebook:** `P04_N30_First_Shot_DCASE2025_Colab.ipynb`, MD5 a594d52903ced0b266cb27af546f3b91 (19 cells, 17 of them code; 2,798 lines of code).
- **Builder:** `notebooks/build_n30_first_shot.py`, MD5 40e27203ccfe539e79c4810ff8224c3d, with the new cells in `notebooks/n30_src/`. It stops unless notebook 28 is the file that ran, and unless the numbers it quotes are in the rules, in the evaluator's repository as cloned, and in notebook 28's saved run.
- **Rules:** `RULES_first_shot_2025_written_before_running.md`, MD5 b4831df4bd8b3ed8a7e85247ebfe6452, written before the notebook existed; two assumptions about the data and gate G-DEV were changed after runs stopped (above).
- **Checker:** `notebooks/check_n30_results.py`, MD5 699bee385d3740313cbae5a1a77251e9. **Tests and logs:** `results/mimii_beats/tests_n30/`.
- The plumbing set and the gate's own test below ran on this exact file. The reader's unit test ran on the third delivery's file, the real-length run on the second's and the structured run on the first's; they differ from this one in the gate, the reader and the clip-length handling.

## What was tested
1. **What comes from notebook 28, which ran on real data.** 2,085 of the 2,798 code lines are also lines of notebook 28.
   - Six cells to the byte: the BEATs code, the checkpoint, the model pieces, LoRA, BEATs iter3, ECHO with gate G0(c).
   - The readers, the training and pass functions, the frozen system from saved files, the scorers, and the whole cell of the view.
   - **Changed by checked substitutions:** the loader (a development type has one zip, an evaluation type two; only the `train` and `test` folders are read); the frozen recipe's list (the cap of 600 for every type; the attribute classes); the class names; the view's two random-stream labels (28 → 30).
   - **New here:** the second set of trainings in its own folder, gate G0(b) for the 2025 names, the choice, the evaluation step for five score sets with the 2025 evaluator, the picture.
2. **The numbers the rules quote** (the builder stops on a mismatch): the 23 MD5s and three record numbers; the evaluator's commit and the two combined hashes, recomputed from the clone; the table of 35 teams row by row; the baselines against the evaluator's own example output; the words' four lines; the counts (15,000, 3,000, 9,000, 14,250; 15 classes; 24 values).
   - The team table was read twice from the results page; the two readings agree to four decimals.
3. **The plumbing set: 21 scenarios, 21 passed** (`run_all30.sh`).
   - **The fixtures:** three development types and three evaluation types, 18 training and 12 test clips each, with the challenge's folder and file names, two supplemental clips and an attribute table per type (which must not be read), attribute strings for four types and "noAttribute" for two. One evaluation type's test names start at 0001, as three real types' do. One type's clips are 0.6 of the kept length and one type's are longer. One type's files are 24-bit, with low bits that are not zero.
   - **The ground truth** is an archive laid out as the evaluator's repository: the real script and its two tool files, and label, domain and name-table files for these clips.
   - **A full run from nothing** (128 s). The official 2025 evaluator ran unmodified and gave the notebook's numbers for the five score sets. The checker recomputed the folder and the unpacked zip: 44 checks, 0 failed each.
   - **Run again; a new local disk.** Nothing trained or scored again; results identical.
   - **A training of the frozen recipe killed after an epoch,** once among the first six and once among the six with attribute classes. Each resumed at its epoch in its own folder; the run completed.
   - **A training of the view killed after a save.** After the resume, everything is identical to the uninterrupted run.
   - **The session dies after the scores are written, before the ground truth is fetched.** The next run wrote no score again and finished, identical to the uninterrupted run.
   - **A later run whose development choice differs from the scored one** stops; the score files are untouched.
   - **The other system as the result** (forced by the harness): the same five sets of numbers, the two readings exchanged.
   - **The stops,** each with no evaluation score and no ground truth: a short clip; a second section in a type; an evaluation test name with a label; too few source-domain clips before the first target-domain clip; a gap in the evaluation numbering; a zip with another MD5; G-DEV; too few clips behind a spread; a saved training with other settings.
   - **A wrong hash for the evaluator's code, and for the ground truth:** the scores stay locked, nothing is unpacked, the run stops; with the right pins the next run finishes on the same score files.
   - **A part below the gate's bar does not stop the run** (bar 0.40 on a copy of the full run: the system as it is 0.4521, the frozen recipe 0.2857): the run finished and the checker passed. A system as it is below the bar still stops it.
   - **Metadata entries in a zip** are skipped and counted; no supplemental clip is in a store.
   - **Every stored clip against the zips,** read again with Python's own reader (180 clips): the 30 short clips are repeated from their start, the 30 long ones cut to their first samples, and the 30 clips of 24 bits equal the values rounded to 16 bits by Python's own integers. A clip of half a second stops the run.
   - **The reader alone, against values built by hand** (`unit_read30.py`, 7 checks): 16-bit PCM comes back exactly; 24-bit PCM, plain and extensible, is read exactly and then rounded (ties to even, clipped); 32-bit PCM; 32-bit float; two channels; a chunk before the data; an 8-bit file is refused.
4. **Real-length clips, all fifteen types** (270 clips; 930 s). One type's clips are 6 s (96,000 samples, as AutoTrash's real ones) and were repeated to 10 s; one type's are longer than 10 s. 160,000 samples kept, 313 frames, a last kernel of 8 × 20, 15 classes, 23 attribute classes. From cell 14 on the test mode was off, and gates G0(d) and G1 passed as on Colab. Checker: 44 checks, 0 failed. Its real-run branch lists the nine mismatches a synthetic folder must give.
5. **A structured synthetic set** (312 clips; the view learned: hold-out accuracy 1.00 for seed 2).
   - The rule's own choice fell on the system with attribute classes there (0.5358 against 0.5112), so that path ran without the harness. Checker: 44 checks, 0 failed.
   - A first try stopped at "a spread of form ZP cannot be formed": one type had fewer held-out clips than the test's minimum. The gate was right; the test's minimum was lowered.
   - These numbers say nothing about the real systems.
6. **Lint:** ruff's F and E9 rules are clean for the builder, the checker and the test tools that run on their own. The notebook has the same ten uses of eight names inside notebook 23's unused branches as notebooks 27 and 28.

## What the tests changed
- The printed shift table named both systems "as it is". It now names the parts.
- The test harness, not the notebook: a kill meant for a training with attribute classes fired in the first six.

## What was not tested
- **Any GPU,** Drive. The downloads and the loader did run on Colab for the first nine types, up to each stop.
- **The six evaluation types not yet reached** (CoffeeGrinder, HomeCamera, Polisher, ScrewFeeder, ToyPet, ToyRCCar). The reader now takes any length from 1 s and four sample formats. A file at another sample rate, or with two channels, would stop the run again in the same cell.
- **The real files.** Their layout and names come from the Zenodo pages, the task page and the evaluator's repository. Cell 9 checks every file before any training and stops with the reason if one is not as assumed. What matters most:
  - the zips unpack to `<type>/train` and `<type>/test` (and `supplemental`, which is skipped);
  - a training name is `section_00_source_train_normal_0001_<attribute>.wav`, and what follows the number is the attribute string;
  - an evaluation test name is `section_00_NNNN.wav`, the name the ground truth uses. I checked this against the repository: the eight ground-truth files use these names, five types from 0000 and three from 0001.
- **How many attribute classes the real names give,** and how few clips some of them hold.
- **The official evaluator on Colab's Python.** It imports `distutils`, which newer Pythons no longer ship; it ran here under Python 3.13 with setuptools. If it does not run there, the notebook says so and its own numbers, which the tests show to be the evaluator's, are the result.
- **The run time** (about 2 hours is an estimate from notebooks 27 and 28).
- **How either system behaves on machine types met with one section only.** That is the experiment.
