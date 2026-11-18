# P04 · Notebook 35 · rules: temporal pooling for the expert on DCASE 2025 Task 2, and a self-weighted fusion

**Written 2026-11-17, before the notebook existed and before any of its scores.**

## Why
Test 3 (DCASE 2025 Task 2, eight machine types never heard) is the project's weakest result. Our best version is the expert
trained with attribute classes, without the trainee: 62.28 %. A second offline attempt (the two experts together) gave 61.44 %.
What is left to try needs the audio again, so it needs a notebook.

## Research map (numbers copied from the sources on 2026-11-17)
| Source | What it does | DCASE 2025 evaluation set |
|---|---|---|
| Challenge results page | first team (Wang_MYPS_task2_3) | 61.63 |
| Fujimura et al. (DCASE 2025; quoted in arXiv 2603.04605, Table VI) | pseudo labels, ensemble feature extractors | 60.0 |
| Wilkinghoff, Yadav and Tan, arXiv 2603.04605 | no training: frozen BEATs (three iterations on AudioSet), relative-deviation pooling with GeM (γ 16, p 3), nearest neighbour with local density normalisation | **62.4** (development 64.0) |
| the same paper, other encoders | OpenL3 58.6, EAT 58.5, Dasheng 57.9 | |
| Wilkinghoff and Tan, arXiv 2609.27362 | no training: 4 encoders × 52 poolings, ensemble weights learned without anomalies by minimising an AUC bound on pseudo-anomalies | **63.95** (NN + LDN); 63.16 with variance minimisation; equal weights 62.87; oracle 66.49 |
| This project, notebook 30 | fine-tuned expert with attribute classes, mean pooling | 62.28 |

The pooling formulas (arXiv 2603.04605): d_t = ‖x_t − mean‖, d̂_t = d_t / max d; RDP weight w_t ∝ (1 + d̂_t)^γ;
GeM(p) = (mean_t max(0, x_t)^p)^(1/p); RDP+GeM = (Σ w_t max(0, x_t)^p / Σ w_t)^(1/p) with p = 3.
The bound (arXiv 2609.27362): B = 1 + (Var z_in + Var z_out) / (E z_out − E z_in)², AUC ≥ 1/B.

**Target:** 63.95 %, the highest published number found for this evaluation set. Second line: 62.4 % (one encoder, no training).

**What is ours here.** Both papers pool frozen encoders. Top challenge systems fine-tune and pool by the mean. Nobody in this map
pools a fine-tuned encoder differently at test time, and nobody mixes trained and training-free listeners in one self-weighted
fusion. Those two are tried here: (1) the same poolings on our fine-tuned attribute-class expert, through its trained head;
(2) one fusion of that expert with the training-free listeners, weighted per machine type without labels.

## What had been seen before writing
Every number of notebook 30's record (five score sets on this evaluation set) and the two-experts attempt. So everything here is a
later attempt on an evaluation set already read. Nothing of any pooling other than the mean on these files.

## Data, models
Notebook 30's cells, unchanged: the 23 zips (MD5), every clip's 10 s, BEATs iter3+ and BEATs iter3 (SHA-256), ECHO.
Notebook 30's six trainings with attribute classes are read from Drive › `P04_FirstShot/attribute_classes` and are not changed.
No training in this notebook.

## Listeners
Tokens of the last layer, all time-frequency patches of the clip. Poolings (11): mean; max; GeM p 3, 6, 10; RDP γ 4, 10, 16, 19;
RDP+GeM (p 3) γ 9, 16.
- **Frozen** (LoRA off): BEATs iter3+ and BEATs iter3, each pooling. Four scorers against the type's 1,000 training clips:
  E0 log nearest Euclidean distance; E1 the same with local density normalisation (K 1, α 1); Ev with α chosen by variance
  minimisation over the references (grid 0 to 1.5, step 0.05); C1 this project's LDN1 (cosine, ratio form).
- **Expert with attribute classes** (LoRA on, six trainings): each pooling, then the trained head. Three forms, scored with LDN1:
  F (notebook 30's: three seeds' head outputs with ECHO, mean over the two encoders); Fx (the same without ECHO); P (the pooled
  backbone features, no head).
- **ECHO alone**, LDN1 (from notebook 30's saved embeddings).
That is 88 + 33 + 1 = 122 candidates.

## The score bank
For each candidate and type, saved without reading any evaluation label: the scores of the 200 test clips; the leave-one-out
scores of the 1,000 reference clips; the scores of 200 feature-mix pseudo-outliers (each dimension from one of two random
references); the scores of 20 training clips of each other type.
The notebook prints the development official score of every candidate. It does not read the evaluation ground truth.

## Gates
- Notebook 30's G0(a), G0(b), G0(c), inherited.
- **G0(q):** for each of the six trainings, the head output of the mean pooling equals notebook 30's saved embeddings (smallest cosine at least 0.999).
- **G0(r):** candidate `AT|F|mean` reproduces notebook 30's saved scores of the expert with attribute classes, development and
  evaluation (Spearman at least 0.999 per type, largest relative difference reported).

## The reading (offline, `notebooks/analyse_n35.py`, fixed here)
The official score Ω (harmonic mean of AUC source, AUC target and pAUC over the types), development and evaluation, of:
- **A0** `AT|F|mean`: must give 0.6228 on the evaluation set (0.5528 development). If not, stop.
- **A1** the published recipe on our files: frozen BEATs iter3, RDP+GeM γ 16, scorer Ev. Beside the paper's 62.4 / 64.0.
- **A2** the pooled expert: `AT|F` with the pooling of the highest development Ω.
- **A3** trained + training-free: A2's listener and A1's listener, each standardised by the median and robust spread of its own
  leave-one-out reference scores for that type, then added. (An LDN1 score is a ratio, so it enters in log, as in form ZP; the Euclidean scores are already logs.)
- **A4** self-weighted fusion: per type, softmax weights over 33 candidates (`AT|F` at the 11 poolings; BEATs iter3 and iter3+
  with scorer Ev at the 11 poolings), standardised as in A3, minimising the bound B with the feature-mix pseudo-outliers as z_out
  and the leave-one-out reference scores as z_in (Adam, 100 steps, learning rate 0.05, from equal weights).
- **A5** equal weights over the same 33 (the control for A4).
Reported beside, not candidates: every single candidate's Ω; A4 with the other-type clips as z_out.

**How the result is written.** All six numbers, development and evaluation, as they come out, beside 62.28, 62.4, 63.95 and 61.63.
"Our best version on Test 3" is the highest evaluation Ω among A0 to A5; it is a later attempt, chosen among six on an evaluation
set already read, and the record says so. A difference below 0.01 is written "no clear difference". If nothing exceeds 0.6228,
the deck stays as it is and the negative result is recorded with its numbers.

## What is not done
No training, no new encoder beyond notebook 30's, no tuning of a pooling exponent on evaluation clips, no weight fitted on labels.
