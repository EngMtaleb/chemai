"""Control investigation, step 2 - the seed spread (the declaration: when the control misses, nothing is reported until the
cause is found). The control's one miss: flux 14 days ahead, rolling test, XGBoost's skill with SDI and turbidity shifted is
2.64 points above its skill without them (the bar: within 2). This re-runs exactly that cell - XGBoost, phase 1's settings,
the same folds - with the declared seeds (0-4, to check this script reproduces the run) and with two other sets of five
seeds (5-9, 10-14). Nothing else changes. If the gap moves by more than the bar from the seeds alone, the bar sits inside
the test's own noise.

    python3 -B ctrl_check2.py [target] [h]        (default flux_pct 14)
"""
import sys

import numpy as np
import pandas as pd

import poc_lib as P

L = P.L
TARGET = sys.argv[1] if len(sys.argv) > 1 else "flux_pct"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SETS = {"0-4": (0, 1, 2, 3, 4), "5-9": (5, 6, 7, 8, 9), "10-14": (10, 11, 12, 13, 14)}


def run():
    t, c = P.load()
    rows = []
    for p in P.PLANTS:
        d, led, _ = P.plant(t, c, p)
        days = P.build_days(d, led)
        fs = P.feature_sets(days)
        minus = [f for f in fs["record 2"] if f not in P.extra_feats(P.INSTRUMENTS)]
        ds, leds, k = P.plant(t, c, p, shift_instruments=True)
        days_s = P.build_days(ds, leds)
        for variant, dd, feats in (("without", days, minus), ("shifted", days_s, fs["record 2"])):
            pairs = dd[dd.valid].reset_index(drop=True)
            for test, h, b, _, train, te in P.folds(pairs):
                if test != "rolling" or h != H:
                    continue
                X, Xt = train[feats].to_numpy(float), te[feats].to_numpy(float)
                ych = (train[f"y_{TARGET}"] - train[TARGET]).to_numpy(float)
                now, y = te[TARGET].to_numpy(float), te[f"y_{TARGET}"].to_numpy(float)
                for name, seeds in SETS.items():
                    f = np.clip(now + L.predict_xgb(L.fit_xgb(X, ych, seeds), Xt), 0, 1)
                    rows.append(pd.DataFrame({"plant": p, "variant": variant, "seeds": name, "block": b,
                                              "train": te.train.values, "t": te.t.values, "now": now, "y": y,
                                              "XGBoost": f}))
            print(p, variant, "done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(P.HERE / "out" / f"ctrl_check2_{TARGET}_{H}.csv", index=False)
    res = []
    for (v, s), g in out.groupby(["variant", "seeds"]):
        res.append({"variant": v, "seeds": s, "n": len(g),
                    "skill_pct": 100 * (1 - np.abs(g.y - g.XGBoost).mean() / np.abs(g.y - g.now).mean())})
    r = pd.DataFrame(res).pivot(index="seeds", columns="variant", values="skill_pct")
    r["shifted - without"] = r.shifted - r.without
    print(r.round(2).to_string())


if __name__ == "__main__":
    run()
