"""Independent checks of results_exp9.json - written separately from run_exp9.py.

1. The rule's archive counts, recomputed by hand from the feature table (no chemai code).
2. No archive loop is in the rule-as-teacher pseudo-labels (names compared).
3. The tuned forest and tuned logistic regression, refitted from the recorded settings,
   give the recorded archive counts.
4. Every recorded F1 equals 2PR/(P+R) from its own counts.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
P2 = REPO / "projects" / "p02_control_loops"
HERE = Path(__file__).parent
R = json.loads((HERE / "results_exp9.json").read_text())
M = R["methods"]
ok = True


def check(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
p22 = p58[(p58.regularity > 1.0) & (p58.cycles >= 10.0)].reset_index(drop=True)

# 1 - the rule by hand
for df, key, n in ((p58, "a58", 58), (p22, "a22", 22)):
    flag = (df.regularity > 1.0) & (df["shape"] > 0.6)
    sticky = df.label == "stiction"
    caught, fa = int((flag & sticky).sum()), int((flag & ~sticky).sum())
    rec = M["the rule"][key]
    check(caught == rec["caught"] and fa == rec["false_alarms"],
          f"rule on {n}: by hand {caught} caught / {fa} false alarms; recorded {rec['caught']} / {rec['false_alarms']}")
check(M["the rule"]["a58"]["caught"] == 12 and M["the rule"]["a58"]["false_alarms"] == 2, "rule on 58 = episode (12 of 26, 2 FA)")
check(M["the rule"]["a22"]["caught"] == 12 and M["the rule"]["a22"]["false_alarms"] == 2, "rule on 22 = episode (12 of 14, 2 FA)")

# 2 - pseudo-labels never touch the archive
spec = importlib.util.spec_from_file_location("w8", P2 / "week8_physics.py")
w8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w8)
from chemai.config import ControlLoopConfig  # noqa: E402
cfg = ControlLoopConfig()
unl = w8.real_table(cfg, labelled=False)
check(len(set(unl.name) & set(plant.name)) == 0, f"unlabelled loops ({len(unl)}) share no name with the labelled archive ({len(plant)})")
check(unl.label.isna().all(), "every unlabelled loop really has no label")

# 3 - refit two tuned models from the recorded settings
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

dev = pd.read_csv(P2 / "results_week8_dev_features.csv")
new = pd.read_csv(P2 / "results_week8_test_features.csv")
train = pd.concat([dev, new], ignore_index=True)
BASE = ["harris", "regularity", "shape_value", "shape_missing", "cycles"]


def conv(v):
    return None if v == "None" else (int(v) if v.isdigit() else (float(v) if v.replace(".", "", 1).isdigit() else v))


def arch(pred, df):
    t, p = df.label.values == "stiction", np.asarray(pred) == "stiction"
    return int((t & p).sum()), int((~t & p).sum())


prm = {k: conv(v) for k, v in M["tuned random_forest"]["params"].items()}
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1, **prm)
rf.fit(w8.design(train, BASE), train.condition.values)
c, f = arch(rf.predict(w8.design(p58, BASE)), p58)
rec = M["tuned random_forest"]["a58"]
check((c, f) == (rec["caught"], rec["false_alarms"]), f"tuned forest refit {prm}: 58 -> {c}/{f}, recorded {rec['caught']}/{rec['false_alarms']}")

C = conv(M["tuned logistic_regression"]["params"]["logisticregression__C"])
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=C))
lr.fit(w8.design(train, BASE), train.condition.values)
c, f = arch(lr.predict(w8.design(p58, BASE)), p58)
rec = M["tuned logistic_regression"]["a58"]
check((c, f) == (rec["caught"], rec["false_alarms"]), f"tuned logistic regression refit C={C}: 58 -> {c}/{f}, recorded {rec['caught']}/{rec['false_alarms']}")

# 4 - F1 arithmetic
bad = []
for name, e in M.items():
    for key in ("a58", "a22") + (("sim",) if e["sim"] else ()):
        s = e[key]
        p = s["caught"] / (s["caught"] + s["false_alarms"]) if s["caught"] + s["false_alarms"] else 0
        r = s["caught"] / s["of"]
        f1 = 2 * p * r / (p + r) if p + r else 0
        if abs(f1 - s["f1"]) > 0.001:
            bad.append((name, key, s["f1"], round(f1, 3)))
check(not bad, f"F1 matches its counts everywhere {bad if bad else ''}")
check(all(e["a58"]["of"] == 26 and e["a22"]["of"] == 14 for e in M.values()), "every method scored on 26 and 14 sticky valves")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
