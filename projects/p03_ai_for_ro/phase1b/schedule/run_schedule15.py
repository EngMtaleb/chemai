"""declaration_v15.md (sha256 in declaration_v15.sha256): page 39's run (phase39/run_schedule.py, unchanged in its steps)
with the calibrated measurement and the new table, for one of two models:

    python3 -B run_schedule15.py every     ->  out/sched_every/   (page 39: every column)
    python3 -B run_schedule15.py routine   ->  out/sched_routine/ (the new page: the real record's columns only)

1 the model: XGBoost (phase 1's settings, five seeds) on the proof of concept's record; a leave-one-plant-out check;
2 the model's schedule on the 15 new plants; 3 the calendar, N as declared; 4 the simulator's own operator (reference);
5 the totals, per law, and the example train (the median change in days above the limit).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sched_lib15 as K  # noqa: E402

MODE = sys.argv[1]
K.CAUSES = {"every": ["temperature_c", "flux_lmh", "sdi", "turbidity_ntu"], "routine": ["temperature_c", "flux_lmh"]}[MODE]
K.FEATS = ["now", "state", "lag1", "lag3", "lag7", "lag14", "slope14", "days_since_cip"] + \
    [f"{c}{s}" for c in K.CAUSES for s in ("", "_m7")]

OUT = HERE / "out" / f"sched_{MODE}"
OUT.mkdir(parents=True, exist_ok=True)
dec = (HERE / "declaration_v15.md").read_bytes()
assert hashlib.sha256(dec).hexdigest() == (HERE / "declaration_v15.sha256").read_text().split()[0], "design changed"
sea = K.S.load_sea_temperature(K.SEA)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# 1 · the model
X, y, now, meta = K.training_pairs()
plants = np.array([m[0] for m in meta])
say("mode", MODE, "· inputs", len(K.FEATS), K.FEATS)
say("training pairs", len(y), "from", sorted(set(plants)))
check = {}
for p in sorted(set(plants)):
    tr, te = plants != p, plants == p
    mods = K.fit(X[tr], y[tr], now[tr])
    pred = np.clip(now[te] + np.mean([m.inplace_predict(X[te]) for m in mods], axis=0), 0, 1)
    check[p] = {"pairs": int(te.sum()), "mae_model_points": round(100 * float(np.mean(np.abs(pred - y[te]))), 2),
                "mae_today_points": round(100 * float(np.mean(np.abs(now[te] - y[te]))), 2)}
say("leave one plant out (dp_pct 7 days ahead, MAE in points):", json.dumps(check))
models = K.fit(X, y, now)

# 2 · the model's schedule
runs = {"model": [K.simulate_policy(s, law, "model", sea, models=models) for s, law in K.EVAL]}
c_model = sum(len(r["cleanings"]) for r in runs["model"])
say("model: cleanings", c_model, "· forecasts made", sum(r["n_pred"] for r in runs["model"]))


# 3 · the calendar: the largest N whose total cleanings is at least the model's
def total(N):
    return sum(len(K.simulate_policy(s, law, "calendar", sea, N=N)["cleanings"]) for s, law in K.EVAL)


counts = {}
lo, hi = 18, 500                                   # total(18) is the most cleanings a calendar can make
for N in (lo, hi):
    counts[N] = total(N)
assert counts[lo] >= c_model > counts[hi], (counts, c_model)
while hi - lo > 1:
    mid = (lo + hi) // 2
    counts[mid] = total(mid)
    lo, hi = (mid, hi) if counts[mid] >= c_model else (lo, mid)
N = lo
for n in range(max(18, N - 3), N + 4):              # check: the count does not rise again just past N
    counts.setdefault(n, total(n))
assert counts[N] >= c_model and all(counts[n] < c_model for n in range(N + 1, N + 4)), counts
say("calendar: every", N, "days ·", counts[N], "cleanings (model", c_model, ")")
runs["calendar"] = [K.simulate_policy(s, law, "calendar", sea, N=N) for s, law in K.EVAL]

# 4 · the simulator's own operator, for reference
runs["operator"] = [K.simulate_policy(s, law, "operator", sea) for s, law in K.EVAL]

# 5 · the totals
sc = pd.DataFrame([x for pol in runs.values() for r in pol for x in K.score(r)])
sc.to_csv(OUT / "scores.csv", index=False)


def summ(g):
    ty = g.running_days.sum() / 365.0                 # running train-years
    return {"trains": int(len(g)), "cleanings": int(g.cleanings.sum()), "days_above": int(g.days_above.sum()),
            "running_days": int(g.running_days.sum()), "cleanings_per_train_year": round(g.cleanings.sum() / ty, 2),
            "days_above_per_train_year": round(g.days_above.sum() / ty, 1),
            "share_of_running_days_above": round(100 * g.days_above.sum() / g.running_days.sum(), 1),
            "mean_rise_pct": round(100 * float((g.mean_rise * g.running_days).sum() / g.running_days.sum()), 1)}


res = {"N": int(N), "counts_by_N": {str(k): v for k, v in sorted(counts.items())}, "check": check,
       "total": {pol: summ(sc[sc.policy == pol]) for pol in runs},
       "by_law": {law: {pol: summ(sc[(sc.policy == pol) & (sc.law == law)]) for pol in runs}
                  for law in ("linear", "biofilm", "asymptotic")}}
for pol, v in res["total"].items():
    say(pol, json.dumps(v))
for law, d in res["by_law"].items():
    say(law, "· model", d["model"]["days_above"], "days above,", d["model"]["cleanings"], "cleanings · calendar",
        d["calendar"]["days_above"], "days above,", d["calendar"]["cleanings"], "cleanings")

# the example train: the median change in days above the limit (model minus calendar), over the 45 trains
a = sc[sc.policy == "model"].set_index(["seed", "train"]).days_above
b = sc[sc.policy == "calendar"].set_index(["seed", "train"]).days_above
diff = (a - b).sort_values(kind="mergesort")
key = diff.index[(len(diff) - 1) // 2]
say("example train (median change):", key, "change", int(diff[key]), "· changes:", diff.describe().round(1).to_dict())
ex = {"seed": int(key[0]), "train": key[1], "law": dict(K.EVAL)[int(key[0])], "N": int(N)}
for pol in ("calendar", "model"):
    r = next(r for r in runs[pol] if r["seed"] == key[0])
    d = r["daily"][r["daily"].train == key[1]].sort_values("day")
    ex["level"] = round(float(r["level"]), 4)
    ex[pol] = {"day": d.day.tolist(), "rise": [round(float(v) - 1.0, 4) for v in d.ratio], "running": d.running.tolist(),
               "cleanings": sorted(int(x) for x in r["cleanings"][r["cleanings"].train == key[1]].day),
               "days_above": int(sc[(sc.policy == pol) & (sc.seed == key[0]) & (sc.train == key[1])].days_above.iloc[0])}
res["example"] = {k: v for k, v in ex.items() if k not in ("calendar", "model")}
res["example"].update({pol: {"cleanings": ex[pol]["cleanings"], "days_above": ex[pol]["days_above"]}
                       for pol in ("calendar", "model")})
(OUT / "example.json").write_text(json.dumps(ex))
(OUT / "results.json").write_text(json.dumps(res, indent=1))
(OUT / "run.log").write_text("\n".join(log) + "\n")
say("written", OUT)
