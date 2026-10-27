"""Post hoc checks - run AFTER the three tests, prompted by an independent audit of this folder.

Nothing here re-runs a declared test or replaces a declared result: the scores in
results/stepA2_ro_test.json, stepB2_uf_test.json and stepC2_ccro_test.json stand as declared.
These checks size three weaknesses the audit found (VALIDATION.md, section 7):

  B1  Part B's flow per element and temperature are means over the whole cycle, so they read past
      minute 5. Refit the frozen sklearn models with first-5-minute values and score again.
  B2  How narrow is Part B's pass? Drop each test episode in turn and recompute the interval; and
      list the episodes that straddle the 2014-01-01 split.
  C1  Three of Part C's minute-20 inputs read a little past minute 20 (a centred 15-sample
      conductivity median; flux, recirculation and temperature as medians of the whole filtration
      phase). Refit the frozen sklearn models with values known at minute 20 and score again.

Each check first rebuilds the declared numbers from the same code path, to show it scores the
same way; the networks are not refitted.

    python projects/p06_ml_ladder/posthoc_checks.py     # needs data/ as Projects 4 and 5 download it
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from chemai.config import CCROConfig, UFPilotConfig, UFSegmentConfig
from chemai.data.nawi_ccro import read_dataset
from chemai.data.uf_pilot import read_run, run_root
from chemai.features.ccro_batches import COLUMNS, batch_curve, cost_curve, phase_of
from chemai.features.ccro_forecast import EARLY, GRID_R, best_stop, physics_at
from chemai.features.uf_forecast import PHYSICS, PLAIN, water_viscosity
from chemai.features.uf_segments import cut_segments, seconds
from chemai.models.ladder import fit_final, group_bootstrap, regressors

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stepB1_uf_dev import CUT, SLOW_CAP as SLOW_B, load  # noqa: E402
from stepC1_ccro_dev import DEV, SLOW_CAP as SLOW_C, X_of, table, target_of, to_ratio  # noqa: E402

OUT = HERE / "results"
P4 = HERE.parent / "p04_tide_fouling" / "results"
P5 = HERE.parent / "p05_when_to_stop" / "results"


# ---------------------------------------------------------------- Part B
def first5_conditions() -> pd.DataFrame:
    """Flow and temperature of every UF segment over its first 5 minutes (same cutting as Project 4)."""
    runs = pd.read_parquet(P4 / "runs.parquet")
    root, scfg = run_root(cfg=UFPilotConfig()), UFSegmentConfig()
    out = []
    for r in runs.itertuples():
        df = read_run(root / r.split / r.file)
        for a, b in cut_segments(df, scfg):
            seg = df.iloc[a:b + 1]
            t = (seconds(seg.index) - seconds(seg.index)[0]) / 60
            f5 = t <= 5.0
            out.append(dict(key=f"{r.file}|{pd.Timestamp(seg.index[0]).value}",
                            q_f5=float(seg["q_uf"].to_numpy(float)[f5].mean()),
                            temp_f5=float(seg["temp_c"].to_numpy(float)[f5].mean())))
    return pd.DataFrame(out)


def score_b(rows: pd.DataFrame, frozen: dict) -> dict:
    """Every frozen sklearn model of Part B, trained on 2012-13 and scored on 2014-15, exactly as
    stepB2 scores it (MAE; minus the no-model median, interval by episode)."""
    d = (rows["t_start"] < CUT).to_numpy()
    dev, test = rows[d].reset_index(drop=True), rows[~d].reset_index(drop=True)
    y, yt, gt = dev["y"].to_numpy(float), test["y"].to_numpy(float), test["episode"].to_numpy()
    base = np.full(len(yt), frozen["baselines"]["no_model_dev_median"]["value"])
    pred = {}
    for fs, feats in (("ml", PLAIN), ("ml_physics", PLAIN + PHYSICS)):
        for name, (model, _) in regressors().items():
            m = fit_final(model, frozen["rungs"][fs][name]["params"], dev[feats], y, SLOW_B.get(name))
            pred[f"{fs}:{name}"] = m.predict(test[feats])
        mem = frozen["rungs"][fs]["ensemble_top3"]["members"]
        pred[f"{fs}:ensemble_top3"] = np.mean([pred[f"{fs}:{k}"] for k in mem], axis=0)
    ok = np.isfinite(yt)
    out = {}
    for k, p in pred.items():
        both = ok & np.isfinite(p)

        def diff(i, p=p, both=both):
            i = i[both[i]]
            return float(np.mean(np.abs(p[i] - yt[i])) - np.mean(np.abs(base[i] - yt[i])))
        lo, hi = group_bootstrap(gt, diff, n=500)
        out[k] = dict(mae=round(float(np.mean(np.abs(p[both] - yt[both]))), 4),
                      minus_no_model=round(diff(np.arange(len(yt))), 4), ci95=[round(lo, 4), round(hi, 4)])
    return out


def part_b1(frozen: dict, declared: dict) -> dict:
    rows = load()
    c5 = first5_conditions()
    fix = rows.merge(c5, on="key", how="left")
    gap_q = 100 * np.log(fix["flow_per_element"] * fix["n_online"].clip(lower=1) / fix["q_f5"])
    gap_t = fix["temp_c"] - fix["temp_f5"]
    period = np.where(fix["t_start"] < CUT, "dev", "test")
    gaps = {p: dict(flow_gap_pct_sd=round(float(gap_q[period == p].std()), 3),
                    flow_gap_corr_with_y=round(float(np.corrcoef(gap_q[period == p].fillna(0), fix["y"][period == p])[0, 1]), 3),
                    temp_gap_c_sd=round(float(gap_t[period == p].std()), 3)) for p in ("dev", "test")}
    fix["flow_per_element"] = fix["q_f5"] / fix["n_online"].clip(lower=1)
    fix["temp_c"] = fix["temp_f5"]
    fix["viscosity"] = water_viscosity(fix["temp_c"]) * 1e3
    fix["flux_ratio"] = fix["flow_per_element"] / fix["prev_flow"]
    fix["pred_history_flux"] = fix["pred_history"] * fix["flux_ratio"]
    fix = fix.drop(columns=["q_f5", "temp_f5"])
    as_declared = score_b(rows, frozen)
    first5 = score_b(fix, frozen)
    reproduces = all(abs(as_declared[k]["mae"] - declared["scores"][k]["mae"]) < 1e-4 for k in as_declared)
    return dict(unmatched_cycles=int(rows["key"].isin(c5["key"]).eq(False).sum()), gaps=gaps,
                reproduces_declared_scores=bool(reproduces), as_declared=as_declared, first5=first5)


def part_b2(frozen: dict) -> dict:
    p = pd.read_parquet(OUT / "stepB2_test_predictions.parquet")
    yt, gt = p["y"].to_numpy(float), p["episode"].to_numpy()
    pick = f"ml:{frozen['frozen']['ml']}"
    pr, base = p[pick].to_numpy(float), p["no_model_dev_median"].to_numpy(float)
    ok = np.isfinite(yt) & np.isfinite(pr)

    def interval(mask, seed=7):
        idx_all = np.flatnonzero(mask)
        g = gt[idx_all]

        def diff(i):
            i = idx_all[i]
            i = i[ok[i]]
            return float(np.mean(np.abs(pr[i] - yt[i])) - np.mean(np.abs(base[i] - yt[i])))
        lo, hi = group_bootstrap(g, diff, n=500, seed=seed)
        return round(diff(np.arange(len(idx_all))), 4), [round(lo, 4), round(hi, 4)]
    full = interval(np.ones(len(yt), bool))
    seeds = [interval(np.ones(len(yt), bool), seed=s)[1] for s in range(20)]          # the declared seed is 7
    drop = {}
    for e in np.unique(gt):
        d, ci = interval(gt != e)
        drop[int(e)] = dict(minus_no_model=d, ci95=ci, verdict="pass" if ci[1] < 0 else "tie" if ci[0] <= 0 else "loss")
    ties = sorted(e for e, v in drop.items() if v["verdict"] != "pass")
    rows = load()
    side = rows.assign(dev=rows["t_start"] < CUT).groupby("episode").agg(dev=("dev", "sum"), n=("dev", "size"),
                                                                       first=("t_start", "min"), last=("t_start", "max"))
    straddle = {int(e): dict(dev_cycles=int(r["dev"]), test_cycles=int(r["n"] - r["dev"]),
                             first_cycle=str(r["first"]), last_cycle=str(r["last"]))
                for e, r in side.iterrows() if 0 < r["dev"] < r["n"]}
    t = rows.loc[rows["t_start"] >= CUT, "t_start"]
    return dict(pick=pick, full=dict(minus_no_model=full[0], ci95=full[1]),
                bootstrap_seeds_0_to_19=dict(passes=int(sum(hi < 0 for _, hi in seeds)), upper_bounds=[hi for _, hi in seeds]),
                test_episodes=int(len(drop)), episodes_whose_removal_makes_a_tie=ties, by_dropped_episode=drop,
                episodes_straddling_the_split=straddle, test_period=[str(t.min()), str(t.max())])


# ---------------------------------------------------------------- Part C
def minute20_features(feats: pd.DataFrame, cfg: CCROConfig) -> tuple[pd.DataFrame, dict]:
    """The same minute-20 features, with conductivity smoothed by a trailing median and flux,
    recirculation and temperature taken as medians of the filtration samples up to minute 20."""
    fixed = feats.copy().set_index("batch")
    curves = {}
    for name in ("dataset3", "dataset4", "dataset6"):
        df = read_dataset(name, columns=COLUMNS)
        for (exp, file), g in df.groupby(["exp", "file"], observed=True, sort=False):
            bid = f"{name}|{exp}|{file}"
            if bid not in fixed.index:
                continue
            cur = batch_curve(g, cfg)
            gs = g.sort_values("row")
            ph = gs["state"].map(phase_of).to_numpy()
            gk = gs[np.isin(ph, ("startup", "stab", "filt"))]
            c_tr = gk["reject_conductivity_ms_per_cm"].astype(float).rolling(cfg.conductivity_smooth_samples, min_periods=1).median().to_numpy()
            filt = cur["phase"].to_numpy() == "filt"
            el, V, live = cur["elapsed_min"].to_numpy(), cur["V_m3"].to_numpy(), cur["live_kwh_m3"].to_numpy()
            w = filt & (el >= EARLY[0]) & (el <= EARLY[1]) & np.isfinite(live)
            i20 = int(np.flatnonzero(filt & (el <= EARLY[1]))[-1])
            i5 = int(np.flatnonzero(filt & (el >= EARLY[0]))[0])
            V20 = float(V[i20])
            dcdv = np.polyfit(V[w], c_tr[w], 1)[0] if np.isfinite(c_tr[w]).all() else np.nan
            early = filt & (el <= EARLY[1])
            fixed.loc[bid, "c5"], fixed.loc[bid, "c20"] = float(c_tr[i5]), float(c_tr[i20])
            fixed.loc[bid, "dc_rel"] = dcdv * V20 / float(c_tr[i20]) if np.isfinite(dcdv) else np.nan
            fixed.loc[bid, "c_ratio"] = float(c_tr[i20] / c_tr[i5]) if c_tr[i5] > 0 else np.nan
            fixed.loc[bid, "flux"] = float(np.nanmedian(gk["flux_lmh"].to_numpy(float)[early]))
            fixed.loc[bid, "q_recirc"] = float(np.nanmedian(gk["reject_flow_l_per_min"].to_numpy(float)[early]))
            fixed.loc[bid, "temp_c"] = float(np.nanmedian(gk["feed_temperature_c"].to_numpy(float)[early]))
            if name == "dataset6":
                curves[bid] = cur
    return fixed.reset_index(), curves


def score_c(F: pd.DataFrame, points: pd.DataFrame, frozen: dict, curves_t: dict) -> dict:
    """Every frozen sklearn model of Part C, scored per brine price exactly as stepC2 scores it."""
    dev = table(F[F.dataset.isin(DEV)], points).reset_index(drop=True)
    tf = F[F.dataset == "dataset6"].reset_index(drop=True)
    rep = tf.loc[tf.index.repeat(len(GRID_R))].reset_index(drop=True)
    rep["r"] = np.tile(GRID_R, len(tf))
    grid = pd.concat([rep, physics_at(rep, rep["r"].to_numpy())], axis=1)
    curves = {}
    for fs in ("ml", "ml_physics"):
        for name, (model, _) in regressors().items():
            m = fit_final(model, frozen["rungs"][fs][name]["params"], X_of(dev, fs), target_of(dev, fs), SLOW_C.get(name))
            curves[f"{fs}:{name}"] = to_ratio(grid, m.predict(X_of(grid, fs)), fs)
        mem = frozen["rungs"][fs]["ensemble_top3"]["members"]
        curves[f"{fs}:ensemble_top3"] = np.mean([curves[f"{fs}:{k}"] for k in mem], axis=0)
    sc = pd.read_parquet(P5 / "step3_test_scores.parquet")
    sc["batch"] = "dataset6|" + sc["exp"].astype(str) + "|" + sc["file"].astype(str)
    n_r, rows = len(GRID_R), []
    for i, f in tf.iterrows():
        for b in (0.0, 1.0, 3.0, 10.0):
            s = sc[(sc["batch"] == f["batch"]) & (sc["b"] == b)]
            if not len(s) or not bool(s["inside"].iloc[0]) or f["batch"] not in curves_t:
                continue
            rec = dict(b=b, V_best=float(s["V_best_l"].iloc[0]), T4=float(s["law_V_star_l"].iloc[0]))
            for k, c in curves.items():
                rec[k] = best_stop(c[i * n_r:(i + 1) * n_r], f.to_dict(), b)
            rows.append(rec)
    R = pd.DataFrame(rows)
    rng = np.random.default_rng(7)
    out = {}
    for b, g in R.groupby("b"):
        law = (100 * (g["T4"] / g["V_best"] - 1)).abs().to_numpy()
        d = {"inside": int(len(g)), "square_root_law_T4": round(float(np.nanmedian(law)), 2)}
        for k in curves:
            err = (100 * (g[k] / g["V_best"] - 1)).abs().to_numpy()
            ok = np.isfinite(err) & np.isfinite(law)
            diffs = []
            for _ in range(1000):
                ii = rng.integers(0, ok.sum(), ok.sum())
                diffs.append(np.median(err[ok][ii]) - np.median(law[ok][ii]))
            d[k] = dict(median_abs_err_pct=round(float(np.nanmedian(err)), 2),
                        minus_law=round(float(np.median(err[ok]) - np.median(law[ok])), 2),
                        ci95=[round(float(np.quantile(diffs, 0.025)), 2), round(float(np.quantile(diffs, 0.975)), 2)])
        out[f"b={b:g}"] = d
    return out


def part_c1(frozen: dict, declared: dict) -> dict:
    cfg = CCROConfig()
    feats = pd.read_parquet(OUT / "ccro_feats.parquet")
    points = pd.read_parquet(OUT / "ccro_points.parquet")
    fixed, curves_t = minute20_features(feats, cfg)
    as_declared = score_c(feats, points, frozen, curves_t)
    minute20 = score_c(fixed, points, frozen, curves_t)
    reproduces = all(abs(as_declared[b][k]["median_abs_err_pct"] - declared["by_b"][b][k]["median_abs_err_pct"]) < 0.006
                     for b in as_declared for k in as_declared[b] if isinstance(as_declared[b][k], dict))
    return dict(reproduces_declared_scores=bool(reproduces), as_declared=as_declared, minute20=minute20)


def main():
    t0 = time.time()
    fb = json.loads((OUT / "stepB1_uf_dev.json").read_text())
    fc = json.loads((OUT / "stepC1_ccro_dev.json").read_text())
    res = {"note": "post hoc, after the declared tests; nothing here replaces a declared result",
           "B1_first5_flow_and_temperature": part_b1(fb, json.loads((OUT / "stepB2_uf_test.json").read_text())),
           "B2_narrowness": part_b2(fb),
           "C1_minute20_inputs": part_c1(fc, json.loads((OUT / "stepC2_ccro_test.json").read_text()))}
    res["seconds"] = round(time.time() - t0)
    (OUT / "posthoc_checks.json").write_text(json.dumps(res, indent=1, default=str))
    b1, b2, c1 = res["B1_first5_flow_and_temperature"], res["B2_narrowness"], res["C1_minute20_inputs"]
    print("B1 reproduces:", b1["reproduces_declared_scores"], "| unmatched cycles:", b1["unmatched_cycles"])
    for k in ("ml:extra_trees", "ml:ensemble_top3", "ml_physics:extra_trees", "ml_physics:ensemble_top3", "ml_physics:svr_rbf"):
        print(f"  {k:28s} declared {b1['as_declared'][k]['mae']:.4f} {b1['as_declared'][k]['ci95']}   first 5 min {b1['first5'][k]['mae']:.4f} {b1['first5'][k]['ci95']}")
    print("B2", b2["full"], "passes under", b2["bootstrap_seeds_0_to_19"]["passes"], "of 20 seeds; ties when dropped:",
          b2["episodes_whose_removal_makes_a_tie"], "straddling:", b2["episodes_straddling_the_split"])
    print("C1 reproduces:", c1["reproduces_declared_scores"])
    for b in c1["minute20"]:
        print(f"  {b}: law {c1['minute20'][b]['square_root_law_T4']}", " | ".join(
            f"{k} {c1['as_declared'][b][k]['median_abs_err_pct']}->{c1['minute20'][b][k]['median_abs_err_pct']}"
            for k in ("ml:ridge", "ml:ensemble_top3", "ml_physics:extra_trees", "ml_physics:ensemble_top3")))
    print(res["seconds"], "s")


if __name__ == "__main__":
    main()
