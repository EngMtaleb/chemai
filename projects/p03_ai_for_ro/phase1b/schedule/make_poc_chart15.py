"""declaration_v15.md, page 38's chart: make_poc_chart.py's method with the model trained on all three plants' earlier
pairs; shown for simulated plant 1 (trains A, B, C), XGBoost with every logged column, dp_pct 7 days ahead, rolling test.

    python3 -B make_poc_chart15.py <deck folder>     ->  <deck folder>/poc_chart.js (const POCC)
"""
import json, sys
import numpy as np, pandas as pd
import poc_lib15 as Q
P, L = Q.P, Q.L
OUT = Q.HERE / "out"
PLANT, C, H = "P1", Q.C, Q.H


def main():
    deck = P.Path(sys.argv[1])
    t, c = Q.load()
    days, feats, pieces = Q.merged(t, c)
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
    R = pd.read_csv(OUT / "pred_trees15.csv")
    R = R[R.plant == PLANT]
    assert len(R)
    d, led, _ = pieces[PLANT]
    pc = lambda a: [round(100 * float(x), 2) for x in a]                           # noqa: E731
    mae = lambda y, p: float(100 * np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))  # noqa: E731
    out = {"plant": PLANT, "law": d.law.iloc[0], "h": H, "blocks": L.BLOCKS, "merged": True,
           "q_by_block": [round(100 * x, 2) for x in q], "trains": {}}
    for tr in "ABC":
        g = d[d.train == tr].sort_values("day_index")
        key = PLANT + tr
        fr, vr, te = (x[x.train == key].sort_values("t") for x in (fit0, cal0, R))
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
            "valid_from": int((vr.t + H).min()) if len(vr) else None, "n_test": int(len(te)),
        }
        o = out["trains"][tr]
        print(tr, {k: round(v, 2) for k, v in o["mae"].items()}, f"cover {o['cover']:.1f} %", f"n_test {o['n_test']}", flush=True)
    allte = R.copy()
    qall = np.array([q[int(b)] for b in allte.block])
    out["plant_test"] = {"mae": mae(allte.y, allte.XGBoost), "mae_nc": mae(allte.y, allte["no change"]), "n": int(len(allte)),
                         "cover": float(100 * np.mean(np.abs(allte.y.to_numpy() - allte.XGBoost.to_numpy()) <= qall))}
    Rall = pd.read_csv(OUT / "pred_trees15.csv")
    qa = np.array([q[int(b)] for b in Rall.block])
    out["all_test"] = {"n": int(len(Rall)), "cover": float(100 * np.mean(np.abs(Rall.y - Rall.XGBoost).to_numpy() <= qa))}
    print("plant 1:", out["plant_test"], "· three plants:", out["all_test"])
    js = "/* made by poc15/make_poc_chart15.py (v15): page 38's forecast chart - nothing typed by hand */\nconst POCC = " + \
        json.dumps(out, separators=(",", ":")) + ";\n"
    (deck / "poc_chart.js").write_text(js)
    (OUT / "poc_chart15.js").write_text(js)
    print("poc_chart.js", round(len(js) / 1024), "KB")


if __name__ == "__main__":
    main()
