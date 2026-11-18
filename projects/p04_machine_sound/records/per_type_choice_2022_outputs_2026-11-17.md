# Test 2 (DCASE 2022 Task 2) · third attempt: choose per machine type — outputs

**Run:** 2026-11-17, offline, by `notebooks/attempt3_per_type_choice.py`, from the per-clip scores notebook 28 saved
(`run_2026-11-13_n28/`). No model was trained or re-scored. Run once.
**Rule, written first:** `RULES_per_type_choice_2022_written_before_running.md`. For each machine type, use whichever of
expert / trainee / fusion has the highest official type score on the **development** set.
**Source of every number below:** `per_type_choice_2022_choice.json` and `per_type_choice_2022_result.json`.

## The choice (development set)
ToyCar → expert · ToyTrain → fusion · bearing → fusion · fan → fusion · gearbox → expert · slider → fusion · valve → expert.
The trainee alone was chosen for no type. Development official score: 0.6961 as it is → 0.7037 with the choice (+0.76 points).

## Evaluation set
| | Official score |
|---|---|
| The system as it is (fusion everywhere) | 0.7063 |
| With the choice per machine type | **0.7150** |
| First team of the challenge | 0.7097 |
| Second team | 0.6822 |

With the choice: AUC source 0.8367, AUC target 0.7064, pAUC 0.6309.
Per type, chosen / fusion: ToyCar 0.9115 / 0.8296 · gearbox 0.8414 / 0.8272 · valve 0.8388 / 0.8448 · the other four unchanged.

## How to read it
- This attempt was made **after** the first evaluation result (0.7063) had been seen. It is a later version, not the first result.
- The choice itself used development clips only; no evaluation label entered it.
- +0.53 points over the first team is inside the noise of this test: in notebook 28's resampling the system as it is came
  first in 733 of 2,000 draws. The honest sentence is "level with the first team".
- Not a challenge entry: the 2022 challenge is closed.
