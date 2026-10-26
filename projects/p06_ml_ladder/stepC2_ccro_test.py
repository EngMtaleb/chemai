"""Part C, the test - run once, after DECLARED_BEFORE_RUN_C.md is committed.

Every rung frozen in step C1 is trained on all development batches and forecasts the
live cost of each test batch (dataset 6, Bayesian optimisation) after minute 20; the
same economics turns each forecast into a stop V*, scored against the batch's own best
exactly as Project 5 scored the square-root law (T4).

    python projects/p06_ml_ladder/stepC2_ccro_test.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import read_dataset
from chemai.features.ccro_batches import COLUMNS, batch_curve, cost_curve
from chemai.features.ccro_forecast import GRID_R, PLAIN, best_stop, physics_at
from chemai.models import deep
from chemai.models.ladder import fit_final, regressors

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stepC1_ccro_dev import DEV, SLOW_CAP, X_of, dl_fit, table, target_of, to_ratio  # noqa: E402

OUT = HERE / "results"
P5 = HERE.parent / "p05_when_to_stop" / "results"
DECL = HERE / "DECLARED_BEFORE_RUN_C.md"
B = (0.0, 1.0, 3.0, 10.0)


def declaration_commit() -> str:
    root = HERE.parents[1]
    rel = str(DECL.relative_to(root))
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel],
                           capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    if not DECL.exists() or dirty or not sha:
        raise SystemExit(f"{rel} must be committed and unchanged before the test runs")
    return sha


def grid_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Each test batch repeated on GRID_R, with the r-dependent physics features."""
    rep = feats.loc[feats.index.repeat(len(GRID_R))].reset_index(drop=True)
    rep["r"] = np.tile(GRID_R, len(feats))
    return pd.concat([rep, physics_at(rep, rep["r"].to_numpy())], axis=1)


def main():
    t0 = time.time()
    sha = declaration_commit()
    frozen = json.loads((OUT / "stepC1_ccro_dev.json").read_text())
    feats = pd.read_parquet(OUT / "ccro_feats.parquet")
    points = pd.read_parquet(OUT / "ccro_points.parquet")
    dev = table(feats[feats.dataset.isin(DEV)], points).reset_index(drop=True)
    tf = feats[feats.dataset == "dataset6"].reset_index(drop=True)
    tpts = table(tf, points).reset_index(drop=True)                       # the test batches' real curves
    grid = grid_table(tf)
    curves = {"square_root_law_line": grid["line_ratio_at_r"].to_numpy(float)}
    on_pts = {"square_root_law_line": tpts["line_ratio_at_r"].to_numpy(float)}
    for fs in ("ml", "ml_physics"):
        for name, (model, _) in regressors().items():
            m = fit_final(model, frozen["rungs"][fs][name]["params"], X_of(dev, fs), target_of(dev, fs),
                          SLOW_CAP.get(name))
            curves[f"{fs}:{name}"] = to_ratio(grid, m.predict(X_of(grid, fs)), fs)
            on_pts[f"{fs}:{name}"] = to_ratio(tpts, m.predict(X_of(tpts, fs)), fs)
        mem = frozen["rungs"][fs]["ensemble_top3"]["members"]
        curves[f"{fs}:ensemble_top3"] = np.mean([curves[f"{fs}:{k}"] for k in mem], axis=0)
        on_pts[f"{fs}:ensemble_top3"] = np.mean([on_pts[f"{fs}:{k}"] for k in mem], axis=0)
    g = dev["exp"].to_numpy()
    for kind in ("mlp", "powerlaw"):
        m, st = dl_fit(kind, dev, g, np.arange(len(dev)), seed=0)
        for tab, store in ((grid, curves), (tpts, on_pts)):
            X = st.transform(tab[PLAIN].to_numpy(float))
            store[f"dl:{kind}"] = deep.predict(m, [X, tab["r"].to_numpy(np.float32), tab["slope_rel"].to_numpy(np.float32)])

    # the truth, as Project 5 scored it: each batch's own best with its own brine; inside only
    sc = pd.read_parquet(P5 / "step3_test_scores.parquet")
    sc["batch"] = "dataset6|" + sc["exp"].astype(str) + "|" + sc["file"].astype(str)
    cfg = CCROConfig()
    raw = read_dataset("dataset6", columns=COLUMNS)
    bt = pd.read_parquet(P5 / "ccro_batches.parquet")
    bt = bt[bt.dataset == "dataset6"].set_index(["exp", "file"])
    Jc = {}
    for (exp, file), gr in raw.groupby(["exp", "file"], observed=True, sort=False):
        bid = f"dataset6|{exp}|{file}"
        if bid in set(tf["batch"]) and (exp, file) in bt.index:
            Jc[bid] = (batch_curve(gr, cfg), bt.loc[(exp, file)])
    n_r = len(GRID_R)
    rows = []
    for i, f in tf.iterrows():
        cur, brow = Jc[f["batch"]]
        filt = cur["phase"].to_numpy() == "filt"
        V = cur["V_m3"].to_numpy() * 1000
        for b in B:
            s = sc[(sc["batch"] == f["batch"]) & (sc["b"] == b)]
            if not len(s) or not bool(s["inside"].iloc[0]):
                continue
            J = cost_curve(cur, brow["own_drain_kwh"], brow["brine_l"], b)
            ok = filt & np.isfinite(J)
            jbest = float(np.nanmin(J[ok]))
            vbest = float(s["V_best_l"].iloc[0])
            rec = dict(batch=f["batch"], b=b, V_best_l=vbest, V_end_l=float(V[ok][-1]))
            preds = {"square_root_law_T4": float(s["law_V_star_l"].iloc[0])}
            for k, c in curves.items():
                preds[k] = best_stop(c[i * n_r:(i + 1) * n_r], f.to_dict(), b)
            for k, v in preds.items():
                rec[f"V:{k}"] = v
                if np.isfinite(v) and v <= V[ok][-1]:
                    j = int(np.flatnonzero(ok)[min(np.searchsorted(V[ok], v), ok.sum() - 1)])
                    rec[f"regret:{k}"] = 100 * (J[j] / jbest - 1)
                else:
                    rec[f"regret:{k}"] = np.nan
            rows.append(rec)
    R = pd.DataFrame(rows)
    methods = ["square_root_law_T4"] + list(curves)
    res = {"declaration_commit": sha, "test_batches": int(len(tf)), "by_b": {}}
    rng = np.random.default_rng(7)
    for b, gb in R.groupby("b"):
        d = {"inside": int(len(gb))}
        law_err = (100 * (gb["V:square_root_law_T4"] / gb["V_best_l"] - 1)).abs().to_numpy()
        for k in methods:
            err = (100 * (gb[f"V:{k}"] / gb["V_best_l"] - 1)).abs().to_numpy()
            ok = np.isfinite(err) & np.isfinite(law_err)
            diffs = []
            for _ in range(1000):
                ii = rng.integers(0, ok.sum(), ok.sum())
                diffs.append(np.median(err[ok][ii]) - np.median(law_err[ok][ii]))
            d[k] = dict(median_abs_err_pct=round(float(np.nanmedian(err)), 2),
                        within_10pct=round(float(np.nanmean(err <= 10)), 3),
                        median_ratio=round(float(np.nanmedian(gb[f"V:{k}"] / gb["V_best_l"])), 3),
                        regret_median_pct=round(float(np.nanmedian(gb[f"regret:{k}"])), 3),
                        beyond_batch=int(gb[f"regret:{k}"].isna().sum()),
                        minus_law=round(float(np.median(err[ok]) - np.median(law_err[ok])), 2),
                        minus_law_ci95=[round(float(np.quantile(diffs, 0.025)), 2), round(float(np.quantile(diffs, 0.975)), 2)])
        res["by_b"][f"b={b:g}"] = d
    yt = tpts["ratio"].to_numpy(float)
    res["curve_mae_after_minute_20"] = {k: round(float(np.mean(np.abs(v - yt))), 4) for k, v in on_pts.items()}
    res["seconds"] = round(time.time() - t0)
    R.to_parquet(OUT / "stepC2_test_stops.parquet")
    (OUT / "stepC2_ccro_test.json").write_text(json.dumps(res, indent=1, default=str))
    for b, d in res["by_b"].items():
        print(b, "inside", d["inside"])
        for k in methods:
            v = d[k]
            print(f"   {k:30s} err {v['median_abs_err_pct']:6.2f}%  ratio {v['median_ratio']:.3f}  regret {v['regret_median_pct']:.3f}%  vs law {v['minus_law']:+.2f} {v['minus_law_ci95']}  beyond {v['beyond_batch']}")
    print(json.dumps(res["curve_mae_after_minute_20"]), res["seconds"], "s")


if __name__ == "__main__":
    main()
