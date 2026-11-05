"""Control investigation, step 3 - is it the yearly cycle? SDI and turbidity carry a yearly cycle (15-27 % of their variance on
the log scale, ctrl_check1.py), and a circular shift of 60-300 days moves that cycle but keeps it: the shifted columns still
tell the season. This re-runs the missed cell - XGBoost, phase 1's settings, the same folds, the same shift k per plant -
with the yearly cycle taken out before the shift: log x = a + b sin(2 pi d/365) + c cos(2 pi d/365) + r(d), fitted on the
plant's 772 days, and the column becomes exp(a + r(d - k)). Nothing else changes. If the gap closes, the cycle carried it.

    python3 -B ctrl_check3.py [target] [h]        (default flux_pct 14)
"""
import sys
import time

import numpy as np
import pandas as pd

import poc_lib as P

L = P.L
TARGET = sys.argv[1] if len(sys.argv) > 1 else "flux_pct"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SETS = {"0-4": (0, 1, 2, 3, 4), "5-9": (5, 6, 7, 8, 9), "10-14": (10, 11, 12, 13, 14)}


def plant_noseason(t, c, p):
    """P.plant(shift_instruments=True), with the yearly cycle removed before the same circular shift."""
    k = int(np.random.default_rng([P.PLANTS[p], 5]).integers(60, 301))            # the declared shift, unchanged
    d = t[t.plant == p].rename(columns={"day": "day_index"}).sort_values(["train", "day_index"]).reset_index(drop=True)
    led = c[c.plant == p].rename(columns={"day": "day_index"})
    pw = pd.read_csv(P.DATA / "plantwide.csv")
    daily = pw[pw.plant == p].set_index("day").sort_index()[P.INSTRUMENTS]
    x = np.arange(len(daily))
    X = np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * x / 365.0), np.cos(2 * np.pi * x / 365.0)])
    moved = {}
    for col in P.INSTRUMENTS:
        y = np.log(daily[col].to_numpy())
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        moved[col] = np.exp(beta[0] + np.roll(r, k))
    moved = pd.DataFrame(moved, index=daily.index)
    d = d.drop(columns=P.INSTRUMENTS).merge(moved, left_on="day_index", right_index=True, how="left")
    return d.sort_values(["train", "day_index"]).reset_index(drop=True), led, k


def run():
    t, c = P.load()
    rows = []
    for p in P.PLANTS:
        d, led, k = plant_noseason(t, c, p)
        days = P.build_days(d, led)
        feats = P.feature_sets(days)["record 2"]
        pairs = days[days.valid].reset_index(drop=True)
        for test, h, b, _, train, te in P.folds(pairs):
            if test != "rolling" or h != H:
                continue
            X, Xt = train[feats].to_numpy(float), te[feats].to_numpy(float)
            ych = (train[f"y_{TARGET}"] - train[TARGET]).to_numpy(float)
            now, y = te[TARGET].to_numpy(float), te[f"y_{TARGET}"].to_numpy(float)
            for name, seeds in SETS.items():
                f = np.clip(now + L.predict_xgb(L.fit_xgb(X, ych, seeds), Xt), 0, 1)
                rows.append(pd.DataFrame({"plant": p, "variant": "shifted, no yearly cycle", "seeds": name, "block": b,
                                          "train": te.train.values, "t": te.t.values, "now": now, "y": y, "XGBoost": f}))
        print(p, "shift", k, "done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(P.HERE / "out" / f"ctrl_check3_{TARGET}_{H}.csv", index=False)
    f2 = P.HERE / "out" / f"ctrl_check2_{TARGET}_{H}.csv"
    while not f2.exists():                                   # step 2 runs beside this one; its file closes the comparison
        time.sleep(30)
    prev = pd.read_csv(f2)
    both = pd.concat([prev, out], ignore_index=True)
    res = []
    for (v, s), g in both.groupby(["variant", "seeds"]):
        res.append({"variant": v, "seeds": s, "skill_pct": 100 * (1 - np.abs(g.y - g.XGBoost).mean() / np.abs(g.y - g.now).mean())})
    r = pd.DataFrame(res).pivot(index="seeds", columns="variant", values="skill_pct")
    for v in ("shifted", "shifted, no yearly cycle"):
        r[f"{v} - without"] = r[v] - r["without"]
    print(r.round(2).to_string())


if __name__ == "__main__":
    run()
