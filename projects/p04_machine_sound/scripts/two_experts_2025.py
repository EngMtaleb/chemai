"""Test 3 (DCASE 2025), second attempt: the two experts together, without the trainee.
Rules: records/RULES_two_experts_2025_written_before_running.md. Uses only notebook 30's saved scores."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path(__file__).resolve().parent.parent / "results" / "run_2026-11-14_n30"; EV = R / "dcase2025_evaluation"; Q = "section_00"
TE = ["AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera", "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"]
TD = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
FLOOR, EPS = 1e-6, sys.float_info.epsilon
hm = lambda v: len(v) / sum(1.0 / max(float(x), EPS) for x in v)
rows = lambda p: [r for r in csv.reader(open(p, newline="")) if r]


def three(y, s, dom):
    y, s, dom = np.asarray(y), np.asarray(s, float), np.asarray(dom)
    return [roc_auc_score(y[((dom == d) & (y == 0)) | (y == 1)], s[((dom == d) & (y == 0)) | (y == 1)]) for d in ("source", "target")] + [roc_auc_score(y, s, max_fpr=0.1)]


def sigma(t, key):
    lg = np.log(np.maximum(np.array([float(r[1]) for r in rows(R / "calibration_scores" / f"{key}_{t}.csv")[1:]]), FLOOR))
    return float(1.4826 * np.median(np.abs(lg - np.median(lg))))


two = lambda t, a, b: np.log(np.maximum(a, FLOOR)) / sigma(t, "frozen") + np.log(np.maximum(b, FLOOR)) / sigma(t, "frozen_AT")

# development set, for reading
dev = {}
for t in TD:
    d = list(csv.DictReader(open(R / "development_scores" / f"scores_{t}_{Q}.csv")))
    y, dom = [int(r["label"]) for r in d], [r["domain"] for r in d]; f, a = (np.array([float(r[c]) for r in d]) for c in ("frozen", "frozen_AT"))
    for nm, s in (("expert", f), ("expert_AT", a), ("two_experts", two(t, f, a))):
        dev.setdefault(nm, []).extend(three(y, s, dom))
DEV = {nm: round(hm(v), 4) for nm, v in dev.items()}; print("development official score:", DEV)
assert DEV["expert"] == 0.5685 and DEV["expert_AT"] == 0.5528, DEV

# evaluation set
ev, per = {}, {}
for t in TE:
    col = lambda folder, name: {r[0]: float(r[1]) for r in rows(EV / folder / name)}
    lab, dm = col("ground_truth_data", f"ground_truth_{t}_{Q}_test.csv"), col("ground_truth_domain", f"ground_truth_{t}_{Q}_test.csv")
    f, a = col("frozen", f"anomaly_score_{t}_{Q}_test.csv"), col("frozen_AT", f"anomaly_score_{t}_{Q}_test.csv")
    k = sorted(lab); assert k == sorted(dm) == sorted(f) == sorted(a) and len(k) == 200, (t, len(k))
    y, dom = [int(lab[i]) for i in k], ["source" if dm[i] == 0 else "target" for i in k]; fv, av = np.array([f[i] for i in k]), np.array([a[i] for i in k])
    for nm, s in (("expert", fv), ("expert_AT", av), ("two_experts", two(t, fv, av))):
        v = three(y, s, dom); ev.setdefault(nm, []).extend(v); per.setdefault(nm, {})[t] = hm(v)
OM = {nm: hm(v) for nm, v in ev.items()}
assert round(OM["expert"], 4) == 0.5922 and round(OM["expert_AT"], 4) == 0.6228, OM                # the record is reproduced first
v = np.array(ev["two_experts"]).reshape(-1, 3)
res = {"rules": "records/RULES_two_experts_2025_written_before_running.md", "development": DEV,
       "evaluation": {"expert": OM["expert"], "expert_AT": OM["expert_AT"], "two_experts": OM["two_experts"], "AUC_source": hm(list(v[:, 0])), "AUC_target": hm(list(v[:, 1])), "pAUC": hm(list(v[:, 2])),
                      "per_type": {t: {nm: per[nm][t] for nm in per} for t in TE}, "first_team": 0.6163, "fusion_with_trainee": 0.5314}}
(R.parent / "two_experts_2025_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["evaluation"], indent=1))
