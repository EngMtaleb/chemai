# P04 · Notebook 25 · tests before delivery (2026-11-12)

Nothing here is evidence about real data. Every test ran on synthetic clips, on a CPU, in my workspace. The notebook has not run on a GPU.

- **Notebook:** `P04_N25_Second_View_DCASE2020_Colab.ipynb`, MD5 e544b75a9c8864c92e90ccd901ab7d67 (12 cells, 1,405 lines of code; code SHA-256 321de9503d9e10b76cae797f1b6a23dae2cbab9e58b94380cd013336052baab4).
- **Builder:** `build_n25_second_view.py`, MD5 5febc66aed9fe1a3afa68880b8d1860f. It stops unless the numbers it quotes are in the rules file and in notebook 23's saved run.
- **Rules:** `RULES_second_view_written_before_running.md`, MD5 2f2eb0cbb183a715fbc87fb5950bdb16 (with Amendment 1).
- **Checker for a results folder:** `notebooks/check_n25_results.py`, MD5 ec2b47f88d9b982f28a47ab0b9051c36.
- **Every test below ran on this exact notebook file** (the last cycle, after the last change).

## What was tested
1. **The model at the reference's sizes** (160,000 samples per clip).
   - 313 frames, last kernel 8 × 20, feature map 512 × 8 × 20 before it.
   - 1,158,514 parameters with 28 classes, equal to a count by hand from the reference's layer definitions (MobileFaceNet 874,524; Tgram 280,406; ArcFace 3,584).
   - The state's key names follow the reference (`arcface.weight`, `tgramnet.conv_extrctor.weight`, …, `mobilefacenet.fc_out.bias`).
   - The margin: the logit at the labelled class is 30·cos(θ + 0.7), elsewhere 30·cos θ (largest difference 1.9e-6).
   - The unused `fc_out` layer gets no gradient, as in the reference.
2. **A full run on the plumbing fixtures** (notebook 23's synthetic test folder as the stand-in for Drive; 232 two-second clips; 3 seeds × 6 epochs).
   - G0(c): the frozen system came back exactly (16 machines and 96 evaluation scores, difference 0.0).
   - The evaluation set was scored once, and the official evaluator (the pinned file, run unmodified) gave the same 32 values (difference 0.0).
   - The checker recomputed every development number from the saved per-clip scores (160 clips × 7 systems, difference 0.0).
3. **Stop and resume.** A run was killed right after the save at epoch 4 of the second training, then started again.
   - The resumed run's features and scores equal the uninterrupted run's exactly (difference 0.0); same decision and same evaluation numbers.
   - Earlier builds of the notebook passed the same test with the kill in the first and in the third training.
4. **Failure paths.**
   - One of notebook 23's score files altered → the run stops at G0(a).
   - One embedding file with shuffled rows → the run stops at G0(c) (development difference 0.13, evaluation scores 0.47).
   - Not kept (forced in the test, and once naturally) → the evaluation set is not scored and no evaluation folder is made.
   - A wrong evaluator file → the result stands and the check is reported as not run, with the reason.
   - A second run of a finished folder → nothing is trained, the saved evaluation results are read, and the score files keep their times.
5. **Real-length clips** (the plumbing fixtures tiled to 10 s): the notebook's own path with 313 frames, the 8 × 20 kernel and 1,158,514 parameters, 2 epochs at batch 8.
6. **A structured synthetic set** (seven harmonic machines per type in coloured noise, three kinds of anomaly, 1,680 clips; a weak stand-in for the frozen system; 2 seeds × 40 epochs with the test's learning rate 5e-4).
   - The view learned: hold-out accuracy 0.60 and 0.57 at the saved epochs (21 and 23); training loss 13.3 → 5.2 and 15.5 → 4.0.
   - Development score: frozen stand-in 0.6512, view alone 0.7575, the five forms 0.8400–0.8700 (C selected and kept; confirmation 0.7963 against 0.7913 needed).
   - The evaluation set was scored once; the official evaluator gave the same 32 values. The checker recomputed 320 clips × 9 systems (difference 0.0).
   - Forms Z and ZP on the evaluation machines equal their formulas written by hand.
   - The spread of the log score on unseen normal clips was 0.16–0.20 for the view by LDN1 and 0.25–0.36 for the stand-in; for the paper's score it was above 1. That is the difference in scale that Z and ZP were added for.
7. **The mixed-precision and channels-last code paths, as far as a CPU walks them** (autocast in bfloat16, both memory layouts): no type errors, finite gradients, the same loss in both layouts without autocast.
8. **Lint:** ruff's F rules, clean.

## What was not tested
- **Any GPU.** Float16 autocast with the gradient scaler, cuDNN, the audio store on the GPU, and the timing of the two memory layouts with its fallback ran nowhere. I read these paths line by line; that is all.
- **Real data,** Drive, the downloads from Zenodo.
- **The run time.** The estimates in the rules are computed from layer sizes, not measured.

## What the synthetic tests changed (recorded in the rules as Amendment 1, before any run on real data)
- Forms Z and ZP (calibrated sums), one hold-out for the three seeds, gate G-VIEW reading both of the view's scores, and gate G0(a) not requiring notebook 23's main JSON to be byte-identical in Drive.
