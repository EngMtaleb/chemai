# Plant-data check — protocol written BEFORE running any new model on plant data

Written: 2026-10-22, after the simulation comparison (results.json), before SACAC arrived.

## What has already been seen
- The 22-loop plant comparison (rule 12/14 · 2 false alarms; published random forest 11/14 · 3 false alarms)
  was run once in the original project. Those 22 loops are therefore NOT a fresh test any more.
- Plant data also shaped three choices (frozen-sensor threshold, no regularity gate, shape threshold check).

## What will be run, once
1. Every model below is tuned and fitted on the simulation development set only
   (GridSearchCV, GroupKFold by simulated loop, scoring macro F1) — exactly the `final_params` in results.json.
   - rule (regularity > 1 and shape > 0.6)
   - logistic regression, decision tree, random forest, gradient boosting
   - random forest without Harris (decided on physical grounds: the plant dead time is unknown)
   - the published random forest (fixed settings, for continuity)
2. Each is applied once to:
   a. the same 22 plant loops as the episode (stated labels, regularity > 1, ≥ 10 cycles), stiction vs other;
   b. all plant loops with a stated label, stiction vs other.
3. Reported for every model: caught / of, false alarms, precision, recall, F1, and the confusion counts.
   Nothing is re-tuned after seeing these numbers. All models are reported, not only the best.

## Decision rule (fixed now)
- The rule stays the shipped method unless a model beats it on (a) by MORE than two loops in caught-minus-false-alarms,
  because with 14 sticky valves a difference of one or two loops is within noise.
- Whatever the outcome, the episode states that these 22 loops had been seen before, so the plant numbers
  describe — they do not prove.
