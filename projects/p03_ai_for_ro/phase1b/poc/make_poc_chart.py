"""Page 38 (v13, note 16): the proof of concept as a forecast chart, the way page 34 shows phase 1 - XGBoost with every logged
column (record 2, the run's best inputs), ΔP (dp_pct) 7 days ahead, rolling test, simulated plant 1, trains A, B and C.

    python3 -B make_poc_chart.py <deck folder>     ->  <deck folder>/poc_chart.js (const POCC)

- Test forecasts: the run's own (out/pred_trees.csv), unchanged.
- Training and validation curves, and the 90 % ranges: phase 1's chart method (make_p1_data.py, unchanged in kind) - for each
  test block, the method refitted on the first three quarters of that block's training pairs, the 90th percentile of its
  absolute error on the last quarter (by day) is the block's range; the curves shown are block 1's (the first block).
  Computed for this page, after the run; labelled so on the page.
"""
import json
import sys

import numpy as np
import pandas as pd

import poc_lib as P

L = P.L
OUT = P.HERE / "out"
PLANT, C, H = "P1", "dp_pct", 7


def main():
    deck = P.Path(sys.argv[1])
    t, c = P.load()
    d, led, _ = P.plant(t, c, PLANT)
    days = P.build_days(d, led)
    feats = P.feature_sets(days)["record 2"]
    pairs = days[days.valid].reset_index(drop=True)
    Ph = pairs[pairs.h == H]

    def split(train):
        cal_start = train.t.iloc[int(round(0.75 * len(train)))]
        return train[train.t + H < cal_start], train[train.t >= cal_start]

    q = []
    for b, (start, _) in enumerate(L.BLOCKS):
        tr_b = Ph[Ph.t + H < start].sort_values(["t", "train"])
        fit_b, cal_b = split(tr_b)
        (f_c,) = L.forecasts(fit_b, [cal_b], C, H, feats)
        q.append(float(np.quantile(np.abs(cal_b[f"y_{C}"].to_numpy() - f_c["XGBoost"]), 0.9)))
        print(f"block {b}: fit {len(fit_b)}, valid {len(cal_b)}, q90 {100 * q[-1]:.2f} points", flush=True)
    train0 = Ph[Ph.t + H < L.BLOCKS[0][0]].sort_values(["t", "train"])
    fit0, cal0 = split(train0)
    f_fit, f_cal = L.forecasts(fit0, [fit0, cal0], C, H, feats)
    fit0 = fit0.assign(pred=f_fit["XGBoost"], nc=fit0[C].to_numpy())
    cal0 = cal0.assign(pred=f_cal["XGBoost"], nc=cal0[C].to_numpy())

    R = pd.read_csv(OUT / "pred_trees.csv", keep_default_na=False, na_values=[""])
    R["held_out"] = R["held_out"].fillna("")
    R = R[(R.plant == PLANT) & (R.record == "record 2") & (R.test == "rolling") & (R.target == C) & (R.h == H)]
    assert len(R) and R.held_out.eq("").all()
    pc = lambda a: [round(100 * float(x), 2) for x in a]                           # noqa: E731
    mae = lambda y, p: float(100 * np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))  # noqa: E731
    out = {"plant": PLANT, "law": d.law.iloc[0] if "law" in d else None, "h": H, "blocks": L.BLOCKS,
           "q_by_block": [round(100 * x, 2) for x in q], "trains": {}}
    for tr in "ABC":
        g = d[d.train == tr].sort_values("day_index")
        fr, vr, te = fit0[fit0.train == tr].sort_values("t"), cal0[cal0.train == tr].sort_values("t"), R[R.train == tr].sort_values("t")
        qte = np.array([q[int(b)] for b in te.block])
        out["trains"][tr] = {
            "real": {"d": g.day_index.astype(int).tolist(), "v": pc(g[C])},
            "cip": led[led.train == tr].day_index.astype(int).tolist(),
            "fit": {"d": (fr.t + H).astype(int).tolist(), "p": pc(fr.pred)},
            "val": {"d": (vr.t + H).astype(int).tolist(), "p": pc(vr.pred)},
            "test": {"d": (te.t + H).astype(int).tolist(), "p": pc(te.XGBoost), "q": [round(100 * x, 2) for x in qte],
                     "nc": pc(te["no change"])},
            "mae": {"fit": mae(fr[f"y_{C}"], fr.pred), "val": mae(vr[f"y_{C}"], vr.pred), "test": mae(te.y, te.XGBoost)},
            "mae_nc": {"fit": mae(fr[f"y_{C}"], fr.nc), "val": mae(vr[f"y_{C}"], vr.nc), "test": mae(te.y, te["no change"])},
            "cover": float(100 * np.mean(np.abs(te.y.to_numpy() - te.XGBoost.to_numpy()) <= qte)),
            "valid_from": int((vr.t + H).min()), "n_test": int(len(te)),
        }
        o = out["trains"][tr]
        print(tr, {k: round(v, 2) for k, v in o["mae"].items()}, {k: round(v, 2) for k, v in o["mae_nc"].items()},
              f"cover {o['cover']:.1f} %", f"valid from day {o['valid_from']}", flush=True)
    allte = R.copy()
    out["plant_test"] = {"mae": mae(allte.y, allte.XGBoost), "mae_nc": mae(allte.y, allte["no change"]), "n": int(len(allte)),
                         "cover": float(100 * np.mean(np.abs(allte.y.to_numpy() - allte.XGBoost.to_numpy())
                                                      <= np.array([q[int(b)] for b in allte.block])))}
    print("plant 1, all trains:", {k: round(v, 2) if isinstance(v, float) else v for k, v in out["plant_test"].items()})
    js = "/* made by make_poc_chart.py (v13): page 38's forecast chart - nothing typed by hand */\nconst POCC = " + \
        json.dumps(out, separators=(",", ":")) + ";\n"
    (deck / "poc_chart.js").write_text(js)
    (OUT / "poc_chart.js").write_text(js)
    print("poc_chart.js", round(len(js) / 1024), "KB")


if __name__ == "__main__":
    main()
