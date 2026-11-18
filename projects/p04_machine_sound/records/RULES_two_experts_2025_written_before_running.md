# Rules · Test 3 (DCASE 2025 Task 2), second attempt: the two experts together, without the trainee

Written on 2026-11-17, before the script ran.

## Why
On Test 3 the trainee is the cause of the low score (alone 0.4474; the fusion with it 0.5314). The expert alone reaches 0.5922,
and the expert trained with attribute classes 0.6228. These are two trainings of the same expert. The development set preferred
the first (0.5685 against 0.5528), the evaluation set the second. When it is not known which of two trainings is better, the
usual answer is to use both. That was never scored.

## What had been seen before writing
Every number of notebook 30's record, including the evaluation scores of both experts. So this is a later attempt, not a first result.

## The one candidate
`two experts` = form ZP with the trainee replaced by the second expert:
log(expert score) / σ_expert + log(expert-with-attribute-classes score) / σ_expert_AT, per machine type,
with the σ notebook 30 already saved (`calibration_scores`). Nothing is fitted, nothing is tuned, no weight is chosen.

## What is run
`notebooks/attempt2_two_experts_2025.py`, offline, from notebook 30's saved per-clip scores. First it must reproduce the
recorded 0.5922 and 0.6228. Then the candidate is scored once on the evaluation set (and on the development set, for reading).

## How the result is written
The number as it comes out, beside 0.6228 (the expert with attribute classes) and 61.63 % (the first team).
Our best version on Test 3 is whichever of the two is higher. No other combination is tried in this attempt.
