"""Part B, development - from the first five minutes of a UF filtration cycle, how far
will the resistance climb by minute eleven?

Development: Project 4's segments before 2014-01-01 (its declared confirmation split:
fit 2012-13, score 2014-15). Everything is tuned and frozen here, grouped by episode;
no 2014-15 cycle is scored.

    python projects/p06_ml_ladder/stepB1_uf_dev.py
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from chemai.features.uf_forecast import CLOCK, PHYSICS, PLAIN
from chemai.models import deep
from chemai.models.ladder import regressors, tune

HERE = Path(__file__).parent
OUT = HERE / "results"
CUT = pd.Timestamp("2014-01-01")
STEPS = 8
SLOW_CAP = {"svr_rbf": 3000}
BASELINES = ["pred_line", "pred_hermia", "pred_history", "pred_history_flux"]


def load():
    rows = pd.read_parquet(OUT / "uf_rows.parquet")
    seg = pd.read_parquet(HERE.parent / "p04_tide_fouling" / "results" / "segments.parquet")
    seg["key"] = seg["run"] + "|" + seg["t_start"].astype("int64").astype(str)
    rows = rows.merge(seg[["key", "t_end", "fouling_pct_h"]], on="key", how="left")
    return rows.sort_values("t_start").reset_index(drop=True)


def history_steps(rows: pd.DataFrame) -> np.ndarray:
    """(n, STEPS, 7): the cycles before each one in its episode (within 3 h), oldest first,
    each known only after it ended; zero-padded at the front."""
    out = np.zeros((len(rows), STEPS, 7), np.float32)
    for _, g in rows.groupby("episode"):
        idx = g.index.to_numpy()
        ts, te = g["t_start"].to_numpy(), g["t_end"].to_numpy()
        feats = np.stack([g["fouling_pct_h"].to_numpy(float) / 20, g["y"].to_numpy(float) / 2,
                          g["lnr0"].to_numpy(float), g["flow_per_element"].to_numpy(float) / 10,
                          g["temp_c"].to_numpy(float) / 20, g["n_online"].to_numpy(float) / 3], 1)
        for j, i in enumerate(idx):
            prev = [k for k in range(max(0, j - 40), j)
                    if te[k] <= ts[j] and (ts[j] - te[k]) / np.timedelta64(1, "h") <= 3]
            prev = prev[-STEPS:]
            for p_, k in enumerate(prev):
                row = feats[k].copy()
                row[2] = row[2] - g["lnr0"].to_numpy(float)[j]              # resistance relative to now
                mins = (ts[j] - te[k]) / np.timedelta64(1, "m")
                out[i, STEPS - len(prev) + p_] = np.r_[row, mins / 60]
    return np.nan_to_num(out)


def dl_inputs(kind, rows, H, st):
    X = st.transform(rows[PLAIN].to_numpy(float))
    return [X, H] if kind == "gru" else [X]


def dl_model(kind, n_in):
    return {"mlp": lambda: deep.MLP(n_in), "gru": lambda: deep.HistoryGRU(7, n_in),
            "hermia": lambda: deep.HermiaNet(n_in)}[kind]()


def dl_fit(kind, rows, H, y, g, idx, seed):
    gs = np.unique(g[idx])
    va_g = np.random.default_rng(seed).choice(gs, max(1, len(gs) // 5), replace=False)
    va, tr = idx[np.isin(g[idx], va_g)], idx[~np.isin(g[idx], va_g)]
    st = deep.Standardiser().fit(rows[PLAIN].to_numpy(float)[tr])
    ins = dl_inputs(kind, rows, H, st)
    m = deep.train(dl_model(kind, ins[0].shape[1]), [a[tr] for a in ins], y[tr], [a[va] for a in ins], y[va],
                   loss="mae", seed=seed, lr=1e-3, patience=10, epochs=120)
    return m, st


def dl_predict(kind, m, st, rows, H):
    return deep.predict(m, dl_inputs(kind, rows, H, st))


def main():
    t0 = time.time()
    rows = load()
    H_all = history_steps(rows)
    d = (rows["t_start"] < CUT).to_numpy()
    dev, H = rows[d].reset_index(drop=True), H_all[d]
    y, g = dev["y"].to_numpy(float), dev["episode"].to_numpy()
    mae = lambda p: float(np.nanmean(np.abs(p - y)))                    # noqa: E731
    res = {"data": dict(rows=int(len(dev)), episodes=int(len(np.unique(g))), cut=str(CUT.date()),
                        y_median=round(float(np.median(y)), 3))}
    med = float(np.median(y))
    res["baselines"] = {"no_model_dev_median": dict(value=round(med, 4), mae=round(mae(np.full(len(y), med)), 4)),
                        "zero": dict(mae=round(mae(np.zeros(len(y))), 4))}
    for b in BASELINES:
        p = dev[b].to_numpy(float)
        res["baselines"][b] = dict(mae=round(float(np.nanmean(np.abs(p - y))), 4), coverage=round(float(np.isfinite(p).mean()), 3))
    oof = pd.DataFrame({"episode": g, "y": y})
    rungs = {}
    for fs_name, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        X = dev[feats]
        sc = {}
        for name, (model, grid) in regressors().items():
            t = tune(name, model, grid, X, y, g, "reg", max_rows=SLOW_CAP.get(name))
            sc[name] = dict(mae=round(-t.cv_score, 4), params=t.params, seconds=round(t.seconds))
            oof[f"{fs_name}:{name}"] = t.oof
            print(fs_name, name, sc[name]["mae"], t.params, flush=True)
        top3 = sorted([k for k in sc], key=lambda k: sc[k]["mae"])[:3]
        ens = oof[[f"{fs_name}:{k}" for k in top3]].mean(1).to_numpy()
        sc["ensemble_top3"] = dict(members=top3, mae=round(mae(ens), 4))
        oof[f"{fs_name}:ensemble_top3"] = ens
        rungs[fs_name] = sc
    yf = y.astype(np.float32)
    for kind in ("mlp", "gru", "hermia"):
        p = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(GroupKFold(5).split(dev, y, g)):
            m, st = dl_fit(kind, dev, H, yf, g, tr, seed=k)
            p[te] = dl_predict(kind, m, st, dev.iloc[te].reset_index(drop=True), H[te])
        rungs.setdefault("dl", {})[kind] = dict(mae=round(mae(p), 4))
        oof[f"dl:{kind}"] = p
        print("dl", kind, rungs["dl"][kind]["mae"], flush=True)
    res["rungs"] = rungs
    seven = list(regressors())
    pick = lambda d, ks: min(ks, key=lambda k: d[k]["mae"])           # noqa: E731
    res["frozen"] = {"ml": pick(rungs["ml"], seven), "ml_ensemble": "ensemble_top3",
                     "ml_physics": pick(rungs["ml_physics"], seven), "ml_physics_ensemble": "ensemble_top3",
                     "dl": pick(rungs["dl"], ["mlp", "gru"]), "dl_physics": "hermia"}
    # the clock, by machine learning: the frozen ML pick with hour-of-day and M2-phase features added
    name = res["frozen"]["ml"]
    model, grid = regressors()[name]
    t = tune(name, model, {k: [v] for k, v in rungs["ml"][name]["params"].items()}, dev[PLAIN + CLOCK], y, g, "reg")
    res["clock_check_dev"] = dict(model=name, mae_without=rungs["ml"][name]["mae"], mae_with_clock=round(-t.cv_score, 4))
    res["seconds"] = round(time.time() - t0)
    oof.to_parquet(OUT / "stepB1_dev_oof.parquet")
    (OUT / "stepB1_uf_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["baselines"]), "\n", json.dumps(res["frozen"]), res["clock_check_dev"], res["seconds"], "s")


if __name__ == "__main__":
    main()
