# Rules · Test 2 (DCASE 2022), a third attempt: choose per machine type on the development set

Written on 2026-11-17, before the script below was run on any evaluation score.

## What is already known (and cannot be un-known)
- The kept system (the fusion) scored 70.63 % on the evaluation set; the first team has 70.97 %. In the resampling of notebook 28 the kept system is first in 733 of 2,000 draws: the two are not separable on this set.
- From notebook 28's table we already know that, on the evaluation set, the fusion is below the expert alone on ToyCar and gearbox. So this attempt is made **after** the evaluation result was seen. Whatever it gives is reported as a later attempt beside 70.63 %, not in its place.

## The idea
DCASE 2022 has the same seven machine types in the development set and in the evaluation set (other sections). So the choice between our three existing systems can be made per machine type on the development set — a choice challenge teams commonly make.

## The rule
- Candidates per type: the expert alone (`frozen`), the trainee alone (`view_paper`), the fusion (`ZP`). Nothing is retrained; the saved scores of notebook 28 are used.
- For each machine type, compute each candidate's development type score: the harmonic mean of AUC (source), AUC (target) and pAUC over the type's three development sections.
- Choose the candidate with the highest development type score. A tie goes to the fusion.
- On the evaluation set, each type's three sections take the chosen candidate's saved scores. The official score is computed once, with the same definitions as notebook 28.

## What will be said
- The new official score, the choices, and the difference to 70.63 % and to 70.97 %.
- Above 70.97 %: "a later attempt, chosen on the development set, is above the first team's score; the difference is inside the noise of this evaluation set".
- Not above: the number is reported as it is.
No other variant (weights, other candidates, other selection sets) is tried after this one is scored.
