"""Part B, the test - run once, after DECLARED_BEFORE_RUN_B.md is committed.

Every rung frozen in step B1 is trained on all 2012-13 cycles and scored once on the
2014-15 cycles. Also: the clock check (does adding hour of day and the M2 tide phase
help the frozen ML pick?) and whether the four Hermia laws can be told apart at all
on a cycle this short.

    python projects/p06_ml_ladder/stepB2_uf_test.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from chemai.features.uf_forecast import CLOCK, LAWS, PHYSICS, PLAIN, fit_hermia, hermia_ln_ratio
from chemai.models.ladder import fit_final, group_bootstrap, regressors

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stepB1_uf_dev import CUT, SLOW_CAP, dl_fit, dl_predict, history_steps, load  # noqa: E402

OUT = HERE / "results"
DECL = HERE / "DECLARED_BEFORE_RUN_B.md"


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


def separability(curves: pd.DataFrame, keys: list[str], noise: pd.Series) -> dict:
    """Fit the four laws to each cycle's whole curve (0-12 min) and ask how far apart their
    fitted curves are at minute 12, against the cycle's own point-to-point noise."""
    cur = curves.copy()
    cur["key"] = cur["run"] + "|" + cur["t_start"].astype("int64").astype(str)
    grp = dict(tuple(cur[cur["key"].isin(keys)].groupby("key")))
    spread, ratio = [], []
    for k in keys:
        c = grp.get(k)
        if c is None:
            continue
        t, z = c["minute"].to_numpy(float), c["lnr"].to_numpy(float)
        m = t <= 12.0
        t, z = t[m], z[m] - z[m][0]
        h = fit_hermia(t, z)
        at = [float(hermia_ln_ratio(law, h[law][0], 12.0) + h[law][2]) for law in LAWS]
        s = (max(at) - min(at)) * 100
        spread.append(s)
        n = noise.get(k, np.nan)
        if np.isfinite(n) and n > 0:
            ratio.append(s / n)
    spread, ratio = np.array(spread), np.array(ratio)
    return dict(cycles=int(len(spread)), spread_at_12min_pct_median=round(float(np.median(spread)), 3),
                spread_over_noise_median=round(float(np.median(ratio)), 3),
                share_spread_above_noise=round(float((ratio > 1).mean()), 3))


def main():
    t0 = time.time()
    sha = declaration_commit()
    frozen = json.loads((OUT / "stepB1_uf_dev.json").read_text())
    rows = load()
    H_all = history_steps(rows)
    d = (rows["t_start"] < CUT).to_numpy()
    dev, Hd, test, Ht = rows[d].reset_index(drop=True), H_all[d], rows[~d].reset_index(drop=True), H_all[~d]
    y, g = dev["y"].to_numpy(float), dev["episode"].to_numpy()
    yt, gt = test["y"].to_numpy(float), test["episode"].to_numpy()
    res = {"declaration_commit": sha, "test": dict(rows=int(len(test)), episodes=int(len(np.unique(gt))),
                                                    period="2014-01-01 to 2015-04-01")}
    pred = pd.DataFrame(index=test.index)
    med = frozen["baselines"]["no_model_dev_median"]["value"]
    pred["no_model_dev_median"] = med
    for b in ("pred_line", "pred_hermia", "pred_history", "pred_history_flux"):
        pred[b] = test[b].to_numpy(float)
    for fs_name, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        for name, (model, _) in regressors().items():
            m = fit_final(model, frozen["rungs"][fs_name][name]["params"], dev[feats], y, SLOW_CAP.get(name))
            pred[f"{fs_name}:{name}"] = m.predict(test[feats])
        mem = frozen["rungs"][fs_name]["ensemble_top3"]["members"]
        pred[f"{fs_name}:ensemble_top3"] = pred[[f"{fs_name}:{k}" for k in mem]].mean(1)
    idx = np.arange(len(dev))
    hermia_w = None
    for kind in ("mlp", "gru", "hermia"):
        m, st = dl_fit(kind, dev, Hd, y.astype(np.float32), g, idx, seed=0)
        pred[f"dl:{kind}"] = dl_predict(kind, m, st, test, Ht)
        if kind == "hermia":
            import torch
            with torch.no_grad():
                w, k = m.parts(torch.as_tensor(st.transform(test[PLAIN].to_numpy(float))))
            hermia_w = w.numpy()
    # the clock check
    name = frozen["frozen"]["ml"]
    model, _ = regressors()[name]
    mc = fit_final(model, frozen["rungs"]["ml"][name]["params"], dev[PLAIN + CLOCK], y, SLOW_CAP.get(name))
    pred[f"clock:{name}"] = mc.predict(test[PLAIN + CLOCK])

    ok = np.isfinite(yt)
    scores = {}
    ref = {"no_model": pred["no_model_dev_median"].to_numpy(), "hermia": pred["pred_hermia"].to_numpy(),
           "history": pred["pred_history"].to_numpy()}
    for c in pred.columns:
        p = pred[c].to_numpy(float)
        m_ = ok & np.isfinite(p)
        sc = dict(mae=round(float(np.mean(np.abs(p[m_] - yt[m_]))), 4), coverage=round(float(m_.mean()), 3))
        for rname, rp_ in ref.items():
            both = m_ & np.isfinite(rp_)

            def diff(i, p=p, rp_=rp_, both=both):
                i = i[both[i]]
                return float(np.mean(np.abs(p[i] - yt[i])) - np.mean(np.abs(rp_[i] - yt[i])))
            lo, hi = group_bootstrap(gt, diff, n=500)
            sc[f"minus_{rname}"] = round(diff(np.arange(len(yt))), 4)
            sc[f"minus_{rname}_ci95"] = [round(lo, 4), round(hi, 4)]
        scores[c] = sc
    res["scores"] = scores
    res["clock_check"] = dict(model=name, mae_without=scores[f"ml:{name}"]["mae"], mae_with=scores[f"clock:{name}"]["mae"])
    res["hermia_weights_test_mean"] = dict(zip(LAWS, np.round(hermia_w.mean(0), 3).tolist()))
    res["hermia_weights_share_dominant"] = round(float((hermia_w.max(1) > 0.5).mean()), 3)
    res["separability"] = separability(pd.read_parquet(OUT / "uf_curves.parquet"), test["key"].tolist(),
                                       test.set_index("key")["early_resid"])
    res["seconds"] = round(time.time() - t0)
    pd.concat([test[["key", "episode", "t_start", "y"]], pred], axis=1).to_parquet(OUT / "stepB2_test_predictions.parquet")
    (OUT / "stepB2_uf_test.json").write_text(json.dumps(res, indent=1, default=str))
    for c, s in scores.items():
        print(f"{c:34s} MAE {s['mae']:.3f}  vs no-model {s['minus_no_model']:+.3f} {s['minus_no_model_ci95']}")
    print(res["clock_check"], res["hermia_weights_test_mean"], res["separability"], res["seconds"], "s")


if __name__ == "__main__":
    main()
