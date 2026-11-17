"""Test 2 (DCASE 2022), third attempt: per machine type, choose among the expert, the trainee and the fusion on the development set.
Rules: records/RULES_per_type_choice_2022_written_before_running.md. Uses only notebook 28's saved scores."""
import csv, json, re, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(__file__).resolve().parent.parent / "results" / "run_2026-11-13_n28"; EV = R / "dcase2022_evaluation"
TYPES = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]; DEV_S, EVAL_S = ["section_00", "section_01", "section_02"], ["section_03", "section_04", "section_05"]
CAND = {"expert": ("frozen", "frozen"), "trainee": ("view_paper", "view"), "fusion": ("ZP", "kept")}          # development column, evaluation folder
hm = lambda v: len(v) / sum(1.0 / max(x, sys.float_info.epsilon) for x in v)


def three(y, s, dom):
    y, s, dom = np.asarray(y), np.asarray(s, float), np.asarray(dom); a = y == 1
    return [roc_auc_score(y[(dom == d) | a], s[(dom == d) | a]) for d in ("source", "target")] + [roc_auc_score(y, s, max_fpr=0.1)]


dev = {}
for t in TYPES:
    for q in DEV_S:
        rows = list(csv.DictReader(open(R / "development_scores" / f"scores_{t}_{q}.csv")))
        for c, (col, _) in CAND.items():
            dev.setdefault((t, c), []).extend(three([int(r["label"]) for r in rows], [float(r[col]) for r in rows], [r["domain"] for r in rows]))
dev_type = {t: {c: hm(dev[(t, c)]) for c in CAND} for t in TYPES}
choice = {t: max(CAND, key=lambda c: (round(dev_type[t][c], 12), c == "fusion")) for t in TYPES}
print("development type scores and the choice:")
for t in TYPES:
    print(f"  {t:<9}" + "".join(f"{c} {dev_type[t][c]:.4f}   " for c in CAND) + "→ " + choice[t])
print("development official score: as it is (fusion everywhere)", round(hm([v for t in TYPES for v in dev[(t, 'fusion')]]), 4), "· with the choice", round(hm([v for t in TYPES for v in dev[(t, choice[t])]]), 4))
(R.parent / "per_type_choice_2022_choice.json").write_text(json.dumps({"development_type_scores": dev_type, "choice": choice}, indent=1))

# ---- the evaluation set: scored once, after the choice is written
def ev_vals(t, folder):
    out = []
    for q in EVAL_S:
        num = lambda n: int(re.search(r"(\d{4})\.wav$", n).group(1))
        lab = {num(r[0]): int(float(r[1])) for r in csv.reader(open(EV / "ground_truth_data" / f"ground_truth_{t}_{q}_test.csv")) if r}
        dom = {num(r[0]): ("source" if float(r[1]) == 0 else "target") for r in csv.reader(open(EV / "ground_truth_domain" / f"ground_truth_{t}_{q}_test.csv")) if r}
        sc = {num(r[0]): float(r[1]) for r in csv.reader(open(EV / folder / f"anomaly_score_{t}_{q}_test.csv")) if r}
        assert sorted(sc) == sorted(lab) == sorted(dom)
        k = sorted(sc); out.append(three([lab[i] for i in k], [sc[i] for i in k], [dom[i] for i in k]))
    return out


ev = {(t, c): ev_vals(t, CAND[c][1]) for t in TYPES for c in CAND}
flat = lambda sel: [v for t in TYPES for sec in ev[(t, sel(t))] for v in sec]
as_is = hm(flat(lambda t: "fusion")); assert abs(as_is - 0.706310) < 5e-7, as_is                  # the recorded result is reproduced first
new = hm(flat(lambda t: choice[t])); vals = np.array(flat(lambda t: choice[t])).reshape(-1, 3)
res = {"rules": "records/RULES_per_type_choice_2022_written_before_running.md", "choice": choice, "development_type_scores": dev_type,
       "evaluation": {"as_it_is_fusion_everywhere": as_is, "with_the_choice": new, "AUC_source": hm(list(vals[:, 0])), "AUC_target": hm(list(vals[:, 1])), "pAUC": hm(list(vals[:, 2])),
                      "per_type": {t: {"chosen": choice[t], "score": hm([v for sec in ev[(t, choice[t])] for v in sec]), "fusion": hm([v for sec in ev[(t, 'fusion')] for v in sec])} for t in TYPES},
                      "expert_everywhere": hm(flat(lambda t: "expert")), "first_team": 0.7097, "difference_to_first_team": new - 0.7097, "difference_to_as_it_is": new - as_is}}
(R.parent / "per_type_choice_2022_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["evaluation"], indent=1))
