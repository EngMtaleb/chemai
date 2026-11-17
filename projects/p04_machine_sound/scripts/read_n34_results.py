"""Reads notebook 34's returned results and recomputes every number from the per-clip score files.
python3 -I read_n34_results.py UNPACKED_DIR"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
D = Path(sys.argv[1]); J = json.loads((D / "array_simple.json").read_text())
LV, FR, PU = J["levels"], J["fronts"], J["pumps"]; FAMS = ["F1", "F2", "F3", "F4", "F1_raw"]
print("date", J["date"], "· test_run", J["test_run"], "· numpy", J["numpy"], "· sklearn", J["sklearn"], "· pumps", PU)
print("n_train", J["n_train"], "n_test", J["n_test"]); print("seconds", {k: v for k, v in J["seconds"].items() if k != "family_4_per_unit"})
print("G0a", J["G0a"]); [print("G0b", k, v) for k, v in J["G0b"].items()]
worst = 0.0; REC = {}
for lv in LV:
    for f in FR:
        for fam in FAMS:
            per = []
            for p in PU:
                u = np.load(D / "scores" / f"{lv}_{f}_{p}.npz", allow_pickle=False)
                a = float(roc_auc_score(u["label"], u[fam])); per.append(a)
                worst = max(worst, abs(round(a, 4) - J["results"][f"{lv}|{f}|{fam}"][p]))
            REC[f"{lv}|{f}|{fam}"] = float(np.mean(per)); worst = max(worst, abs(round(float(np.mean(per)), 4) - J["results"][f"{lv}|{f}|{fam}"]["avg"]))
print("recomputed from the 36 score files against the JSON: largest difference", round(worst, 5))
for fam in FAMS:
    print(f"\n{fam}: average AUC of {len(PU)} pumps      " + "   ".join(f"{f:>7s}" for f in FR))
    for lv in LV:
        print(f"   {lv:>5s}                         " + "   ".join(f"{REC[f'{lv}|{f}|{fam}']:7.4f}" for f in FR))
print("\nFamily 5, pre-trained alone (notebook 11):", json.dumps(J["family5_frozen_notebook11"]))
print("\nanswers at the loudest level:"); [print("  ", k, json.dumps(v)) for k, v in J["answers"].items()]
# labels and names are the same clips for every front end of a level and pump
ok = all(np.array_equal(np.load(D / "scores" / f"{lv}_CH0_{p}.npz")["name"], np.load(D / "scores" / f"{lv}_{f}_{p}.npz")["name"]) for lv in LV for f in FR for p in PU)
print("same test clips for the three front ends:", ok)
