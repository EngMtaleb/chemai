"""Independent checks of results_exp10.json, using the Week 8 plant table (not real_meta) for the truth."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results_exp10.json").read_text())
plant = pd.read_csv(HERE.parents[1] / "results_week8_plant_features.csv")   # written by week8_physics.py
p58 = plant[plant.tier == "stated"].set_index("name")
p22 = p58[(p58.regularity > 1.0) & (p58.cycles >= 10.0)]
ok = True


def check(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


for kind, e in R["networks"].items():
    calls = pd.Series(e["archive_calls"])
    check(set(calls.index) == set(p58.index), f"{kind}: calls cover exactly the 58 stated loops")
    for df, key, n in ((p58, "a58", 58), (p22, "a22", 22)):
        c = calls.loc[df.index]
        sticky, flag = df.label == "stiction", c == "stiction"
        caught, fa = int((sticky & flag).sum()), int((~sticky & flag).sum())
        rec = e[key]
        check((caught, fa) == (rec["caught"], rec["false_alarms"]),
              f"{kind} on {n}: recount {caught} caught / {fa} FA; recorded {rec['caught']} / {rec['false_alarms']}")
    for key in ("sim", "a58", "a22"):
        s = e[key]
        p = s["caught"] / (s["caught"] + s["false_alarms"]) if s["caught"] + s["false_alarms"] else 0
        r = s["caught"] / s["of"]
        f1 = 2 * p * r / (p + r) if p + r else 0
        check(abs(f1 - s["f1"]) < 0.001, f"{kind} {key}: F1 {s['f1']} matches its counts ({f1:.3f})")
    check(e["sim"]["of"] == 96, f"{kind}: simulation scored on 96 sticky runs")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
