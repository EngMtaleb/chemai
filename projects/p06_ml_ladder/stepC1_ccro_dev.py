"""Part C, development - forecast a CCRO batch's live cost after minute 20.

Development: the constant-setpoint batches of datasets 3 (extremum seeking) and 4 (open loop)
that run 25 minutes or more. Tuning is grouped by experiment (four groups); the open-loop
group is the only one whose batches run far past minute 20, so its fold is also reported
alone: how each model does where it must extrapolate. Nothing here touches dataset 6.

    python projects/p06_ml_ladder/stepC1_ccro_dev.py
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from chemai.features.ccro_forecast import PHYSICS, PLAIN, physics_at
from chemai.models import deep
from chemai.models.ladder import regressors, tune

HERE = Path(__file__).parent
OUT = HERE / "results"
DEV = ("dataset3", "dataset4")
SLOW_CAP = {"svr_rbf": 4000, "k_nearest_neighbours": 12000}
FOLDS = 4                     # one per experiment: three extremum-seeking runs and the open loop


def table(feats: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    t = points.merge(feats, on="batch", how="inner")
    phys = physics_at(t, t["r"].to_numpy())
    return pd.concat([t, phys], axis=1)


def X_of(t: pd.DataFrame, fs: str) -> pd.DataFrame:
    return t[PLAIN + ["r"]] if fs == "ml" else t[PLAIN + PHYSICS + ["r"]]


def target_of(t: pd.DataFrame, fs: str) -> np.ndarray:
    return (t["ratio"] if fs == "ml" else t["resid"]).to_numpy(float)


def to_ratio(t: pd.DataFrame, pred: np.ndarray, fs: str) -> np.ndarray:
    return pred if fs == "ml" else t["line_ratio_at_r"].to_numpy(float) + pred


def dl_fit(kind, t, groups, idx, seed):
    gs = np.unique(groups[idx])
    va_g = np.random.default_rng(seed).choice(gs, 1, replace=False)
    va, tr = idx[np.isin(groups[idx], va_g)], idx[~np.isin(groups[idx], va_g)]
    st = deep.Standardiser(robust=True).fit(t[PLAIN].to_numpy(float)[tr])
    X = st.transform(t[PLAIN].to_numpy(float))
    r = t["r"].to_numpy(np.float32)
    sl = t["slope_rel"].to_numpy(np.float32)
    y = t["ratio"].to_numpy(np.float32)
    m = deep.PointMLP(X.shape[1]) if kind == "mlp" else deep.PowerLawNet(X.shape[1])
    ins = [X, r, sl]
    m = deep.train(m, [a[tr] for a in ins], y[tr], [a[va] for a in ins], y[va], loss="mae", seed=seed,
                   lr=1e-3, epochs=150, patience=12, batch=512)
    return m, st


def dl_predict(m, st, t):
    X = st.transform(t[PLAIN].to_numpy(float))
    return deep.predict(m, [X, t["r"].to_numpy(np.float32), t["slope_rel"].to_numpy(np.float32)])


def main():
    t0 = time.time()
    feats = pd.read_parquet(OUT / "ccro_feats.parquet")
    points = pd.read_parquet(OUT / "ccro_points.parquet")
    dev = table(feats[feats.dataset.isin(DEV)], points).reset_index(drop=True)
    g = dev["exp"].to_numpy()
    yr = dev["ratio"].to_numpy(float)
    ol = (dev["dataset"] == "dataset4").to_numpy()
    res = {"data": dict(batches=int(dev["batch"].nunique()), points=int(len(dev)),
                        groups=sorted(map(str, np.unique(g))), r_max_esc=round(float(dev.loc[~ol, "r"].max()), 2),
                        r_max_open_loop=round(float(dev.loc[ol, "r"].max()), 2))}
    mae = lambda p, m=None: float(np.mean(np.abs(p[m] - yr[m]))) if m is not None else float(np.mean(np.abs(p - yr)))  # noqa: E731
    line = dev["line_ratio_at_r"].to_numpy(float)
    res["baselines"] = {"square_root_law_line": dict(mae=round(mae(line), 4), mae_open_loop=round(mae(line, ol), 4)),
                        "flat_at_m20": dict(mae=round(mae(np.ones(len(yr))), 4))}
    oof = pd.DataFrame({"batch": dev["batch"], "r": dev["r"], "ratio": yr, "group": g})
    rungs = {}
    for fs in ("ml", "ml_physics"):
        X, y = X_of(dev, fs), target_of(dev, fs)
        sc = {}
        for name, (model, grid) in regressors().items():
            t = tune(name, model, grid, X, y, g, "reg", folds=FOLDS, max_rows=SLOW_CAP.get(name))
            p = to_ratio(dev, t.oof, fs)
            sc[name] = dict(mae=round(mae(p), 4), mae_open_loop=round(mae(p, ol), 4), params=t.params,
                            seconds=round(t.seconds))
            oof[f"{fs}:{name}"] = p
            print(fs, name, sc[name]["mae"], sc[name]["mae_open_loop"], t.params, flush=True)
        top3 = sorted([k for k in sc], key=lambda k: sc[k]["mae"])[:3]
        e = oof[[f"{fs}:{k}" for k in top3]].mean(1).to_numpy()
        sc["ensemble_top3"] = dict(members=top3, mae=round(mae(e), 4), mae_open_loop=round(mae(e, ol), 4))
        oof[f"{fs}:ensemble_top3"] = e
        rungs[fs] = sc
    for kind in ("mlp", "powerlaw"):
        p = np.full(len(yr), np.nan)
        for k, (tr, te) in enumerate(GroupKFold(FOLDS).split(dev, yr, g)):
            m, st = dl_fit(kind, dev, g, tr, seed=k)
            p[te] = dl_predict(m, st, dev.iloc[te])
        rungs.setdefault("dl", {})[kind] = dict(mae=round(mae(p), 4), mae_open_loop=round(mae(p, ol), 4))
        oof[f"dl:{kind}"] = p
        print("dl", kind, rungs["dl"][kind], flush=True)
    res["rungs"] = rungs
    seven = list(regressors())
    pick = lambda d, ks: min(ks, key=lambda k: d[k]["mae"])           # noqa: E731
    res["frozen"] = {"ml": pick(rungs["ml"], seven), "ml_ensemble": "ensemble_top3",
                     "ml_physics": pick(rungs["ml_physics"], seven), "ml_physics_ensemble": "ensemble_top3",
                     "dl": "mlp", "dl_physics": "powerlaw"}
    res["seconds"] = round(time.time() - t0)
    oof.to_parquet(OUT / "stepC1_dev_oof.parquet")
    (OUT / "stepC1_ccro_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["baselines"]), json.dumps(res["frozen"]), res["seconds"], "s")


if __name__ == "__main__":
    main()
