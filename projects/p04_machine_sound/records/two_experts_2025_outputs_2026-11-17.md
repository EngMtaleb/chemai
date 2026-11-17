# Test 3 (DCASE 2025 Task 2) · second attempt: the two experts together — outputs

**Run:** 2026-11-17, offline, by `notebooks/attempt2_two_experts_2025.py`, from notebook 30's saved per-clip scores. Run once.
**Rule, written first:** `RULES_two_experts_2025_written_before_running.md`. **Source of the numbers:** `two_experts_2025_result.json`.
The script first reproduced the recorded 0.5922 and 0.6228.

| System (no trainee in any of them) | Development | Evaluation |
|---|---|---|
| The expert alone | 0.5685 | 0.5922 |
| The expert with attribute classes | 0.5528 | **0.6228** |
| The two experts together (the candidate) | 0.5603 | 0.6144 |
| First team of the challenge | | 0.6163 |

**Outcome: no gain.** The two together land between the two experts (0.6144), below the expert with attribute classes.
Our best version on Test 3 stays the expert with attribute classes: 62.28 %, 0.65 points above the first team's 61.63 %.
Per type the candidate is best only on BandSealer (0.6454 against 0.6335).

For the record: 0.6228 was not the system the rules of notebook 30 had chosen before scoring (that was the fusion, 0.5314),
and this attempt came after those results were seen. The deck presents it as "our best version". Not a challenge entry.
