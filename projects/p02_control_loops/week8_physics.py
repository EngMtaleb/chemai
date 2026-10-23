"""Project 2 - Week 8: physics-informed diagnosis, run once (PROTOCOL_physics.md).

Run:  python projects/p02_control_loops/week8_physics.py
      (needs data/sacac/ and data/isdb/isdb10.mat)

The rule beat every model trained on simulation when the models were tested on
plant loops (PROTOCOL_plant.md). Here the physics of the valve goes into the
method in four ways, all fixed before this run:

  1  physics alone   S read from the trend: how far CO travels after a reversal
                     before CV answers (chemai.features.stiction_physics)
  2  physics-informed models   the five model families, with that reading added
                     and Harris computed in simulation exactly as on plant data
  3  hybrid          the rule decides where it is clear, the best physics-informed
                     model decides the rest
  4  rule as teacher the rule labels the unlabelled real loops where it is clear;
                     a forest learns from simulation plus those loops

Every method is reported - on the simulation test (new plants), on the 22 plant
loops of the episode and on all 58 loops with a stated label - and judged by
the decision rule written before the run. Nothing is re-tuned afterwards.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import generate_dataset, load_isdb, load_sacac
from chemai.features import delay_samples, harris_index
from chemai.features.stiction_physics import reversal_travel
from chemai.models import baseline_predict, loop_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week8")
log.setLevel(logging.INFO)

HERE = Path(__file__).parent
OUT = HERE / "results_week8_physics.json"
_spec = importlib.util.spec_from_file_location("compare_models", HERE / "compare_models.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

PLANT_DELAY = 2          # the plant dead time is unknown: the smallest delay, as in week2_classifier
MIN_SAMPLES = 200
BASE = ["harris", "regularity", "shape_value", "shape_missing", "cycles"]
PHYS = ["harris_p", "regularity", "shape_value", "shape_missing", "cycles",
        "s_travel", "travel_lag", "travel_missing"]
PHYS_NO_HARRIS = [f for f in PHYS if f != "harris_p"]


# ======================================================================= feature tables

def sim_table(cfg: ControlLoopConfig) -> pd.DataFrame:
    specs, runs = generate_dataset(8, 2, cfg)
    by_id = {s.loop_id: s for s in specs}
    rows = []
    for (loop_id, seed), d in runs.items():
        s = by_id[loop_id]
        f = loop_features(d.sp, d.pv, d.op, s.loop_type,
                          delay_samples(s.theta + s.ts / s.substeps, s.ts), cfg)
        r = reversal_travel(d.op.to_numpy(), d.pv.to_numpy())
        rows.append({"loop": loop_id, "seed": seed, "condition": s.condition, "loop_type": s.loop_type,
                     "true_s": s.stiction_s, "true_j": s.stiction_j,
                     "slip_to_load": s.slip_to_load if s.condition == "stiction" else np.nan,
                     **f.as_row(),
                     # Harris exactly as it is computed on plant data: the same assumed delay
                     "harris_p": harris_index(d.sp.to_numpy() - d.pv.to_numpy(), PLANT_DELAY, cfg.harris_ar_order),
                     **r.as_row(), "s_travel_co": r.s_co})
    return pd.DataFrame(rows)


def real_table(cfg: ControlLoopConfig, labelled: bool) -> pd.DataFrame:
    rows = []

    def add(name, label, tier, loop_type, sp, pv, op):
        if not np.isfinite(pv).all() or len(pv) < MIN_SAMPLES:
            return
        f = loop_features(sp, pv, op, loop_type, PLANT_DELAY, cfg)
        r = reversal_travel(op, pv)
        rows.append({"name": name, "label": label, "tier": tier, "loop_type": loop_type,
                     **f.as_row(), "harris_p": f.harris, **r.as_row(), "s_travel_co": r.s_co})

    for rec in load_sacac(DATA_DIR / "sacac"):
        if (rec.label is not None) != labelled:
            continue
        d = rec.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        add(rec.name, rec.label, "stated" if labelled else None, rec.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.in_sacac or (x.label is not None) != labelled:
            continue
        d = x.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        add(x.key, x.label, x.tier, x.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
    return pd.DataFrame(rows)


def design(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Feature table -> model input. A missing shape or reading becomes a neutral value plus a flag."""
    out = df.copy()
    out["shape_missing"] = out["shape"].isna().astype(int)
    out["shape_value"] = out["shape"].fillna(0.5)
    out["travel_missing"] = out["s_travel"].isna().astype(int)
    out["s_travel"] = out["s_travel"].fillna(0.0)
    out["travel_lag"] = out["travel_lag"].fillna(0.0)
    return out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ======================================================================= helpers

def binary(pred) -> np.ndarray:
    return np.where(np.asarray(pred) == "stiction", "stiction", "other")


def truth_plant(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.label.values == "stiction", "stiction", "other")


def clear_zone(df: pd.DataFrame, cfg: ControlLoopConfig) -> np.ndarray:
    """'stiction' / 'other' where the rule is clear, '' elsewhere (the grey zone)."""
    reg = df.regularity.values > cfg.regularity_threshold
    shape = df["shape"].values
    out = np.full(len(df), "", dtype=object)
    out[reg & (shape > cfg.shape_triangular)] = "stiction"
    out[reg & (shape < cfg.shape_sinusoidal)] = "other"
    return out


def nested(dev: pd.DataFrame, est, grid, feats, y):
    """Out-of-fold predictions with settings tuned inside each training fold, then the final fit."""
    X, g = design(dev, feats), dev.loop.values
    oof = np.empty(len(dev), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, g):
        gs = GridSearchCV(clone(est), grid, cv=GroupKFold(4), scoring="f1_macro", n_jobs=2)
        gs.fit(X.iloc[tr], y[tr], groups=g[tr])
        oof[te] = gs.predict(X.iloc[te])
    gs = GridSearchCV(clone(est), grid, cv=GroupKFold(5), scoring="f1_macro", n_jobs=2)
    gs.fit(X, y, groups=g)
    return oof, gs.best_estimator_, {k: str(v) for k, v in gs.best_params_.items()}


def scores_all(name, dev_pred, test_pred, plant22_pred, plant58_pred, dev, test, p22, p58, res):
    entry = {}
    if dev_pred is not None:
        entry["dev_oof"] = cmp.stiction_binary(dev.condition.values, dev_pred)
    entry["test"] = cmp.stiction_binary(test.condition.values, test_pred)
    entry["test_f1_ci"] = cmp.bootstrap_ci(test, test_pred, cmp.stiction_f1)
    entry["plant22"] = cmp.stiction_binary(truth_plant(p22), binary(plant22_pred))
    entry["plant58"] = cmp.stiction_binary(truth_plant(p58), binary(plant58_pred))
    res["methods"][name] = entry
    t, a, b = entry["test"], entry["plant22"], entry["plant58"]
    log.info("%-30s test F1 %.3f %s | 22 loops: %d/%d FA %d | 58 loops: %d/%d FA %d", name, t["f1"],
             entry["test_f1_ci"], a["caught"], a["of"], a["false_alarms"], b["caught"], b["of"], b["false_alarms"])


# ======================================================================= the run

def main() -> dict:
    t0 = time.time()
    cfg = ControlLoopConfig()
    dev = sim_table(cfg)
    test = sim_table(dataclasses.replace(cfg, random_state=2026))
    test["loop"] = "T-" + test["loop"]
    real = real_table(cfg, labelled=True)
    unlabelled = real_table(cfg, labelled=False)
    p58 = real[real.tier == "stated"].reset_index(drop=True)
    p22 = p58[(p58.regularity > cfg.regularity_threshold) & (p58.cycles >= cfg.min_cycles)].reset_index(drop=True)
    log.info("dev %d runs, test %d runs, real labelled %d (stated %d, episode set %d), unlabelled %d  (%.0fs)",
             len(dev), len(test), len(real), len(p58), len(p22), len(unlabelled), time.time() - t0)

    res = {"protocol": "PROTOCOL_physics.md (with amendment 1)", "n": {
        "dev_runs": len(dev), "test_runs": len(test), "plant_stated": len(p58), "plant_episode": len(p22),
        "unlabelled_real": len(unlabelled)}, "methods": {}}
    yd = dev.condition.values

    # ---- 0 · the rule, and the models without physics (as in compare_models.py)
    rule = lambda df: baseline_predict(df, cfg).values                                   # noqa: E731
    scores_all("rule", rule(dev), rule(test), rule(p22), rule(p58), dev, test, p22, p58, res)
    for name, (est, grid, feats) in cmp.MODELS.items():
        feats = BASE if feats == cmp.FEATURES else [f for f in BASE if f != "harris"]
        oof, model, params = nested(dev, est, grid, feats, yd)
        scores_all(f"ml_{name}", oof, model.predict(design(test, feats)), model.predict(design(p22, feats)),
                   model.predict(design(p58, feats)), dev, test, p22, p58, res)
        res["methods"][f"ml_{name}"]["params"] = params

    # ---- 1 · physics alone: S read from the trend, one threshold chosen on development data
    s_dev = dev.s_travel.fillna(0.0).values
    grid_t = np.unique(np.round(np.quantile(s_dev[s_dev > 0], np.linspace(0.05, 0.95, 37)), 3))
    f1s = [cmp.stiction_f1(yd, np.where(s_dev > t, "stiction", "other")) for t in grid_t]
    theta = float(grid_t[int(np.argmax(f1s))])
    phys = lambda df: np.where(df.s_travel.fillna(0.0).values > theta, "stiction", "other")   # noqa: E731
    scores_all("physics_alone", phys(dev), phys(test), phys(p22), phys(p58), dev, test, p22, p58, res)
    res["methods"]["physics_alone"]["threshold_s_std"] = theta

    # ---- 2 · physics-informed models
    best_name, best_f1, best_model, best_oof, best_feats = None, -1.0, None, None, None
    for name, (est, grid, feats) in cmp.MODELS.items():
        feats = PHYS if feats == cmp.FEATURES else PHYS_NO_HARRIS
        oof, model, params = nested(dev, est, grid, feats, yd)
        scores_all(f"pi_{name}", oof, model.predict(design(test, feats)), model.predict(design(p22, feats)),
                   model.predict(design(p58, feats)), dev, test, p22, p58, res)
        res["methods"][f"pi_{name}"]["params"] = params
        if hasattr(model, "feature_importances_"):
            res["methods"][f"pi_{name}"]["feature_importances"] = dict(zip(feats, np.round(model.feature_importances_, 3).tolist()))
        f1 = res["methods"][f"pi_{name}"]["dev_oof"]["f1"]
        if f1 > best_f1:
            best_name, best_f1, best_model, best_oof, best_feats = name, f1, model, oof, feats
    res["best_physics_informed"] = best_name

    # ---- 3 · hybrid: the rule where it is clear, the best physics-informed model elsewhere
    def hybrid(df, model_pred):
        z = clear_zone(df, cfg)
        return np.where(z == "", binary(model_pred), z)
    scores_all("hybrid", hybrid(dev, best_oof), hybrid(test, best_model.predict(design(test, best_feats))),
               hybrid(p22, best_model.predict(design(p22, best_feats))),
               hybrid(p58, best_model.predict(design(p58, best_feats))), dev, test, p22, p58, res)
    res["methods"]["hybrid"]["grey_zone_share"] = {
        "test": round(float((clear_zone(test, cfg) == "").mean()), 3),
        "plant22": round(float((clear_zone(p22, cfg) == "").mean()), 3),
        "plant58": round(float((clear_zone(p58, cfg) == "").mean()), 3)}

    # ---- 4 · rule as teacher: pseudo-labels on the unlabelled real loops where the rule is clear
    z = clear_zone(unlabelled, cfg)
    pseudo = unlabelled[z != ""].copy()
    pseudo["target"] = z[z != ""]
    rf_params = res["methods"]["pi_random_forest"]["params"]
    est, grid, _ = cmp.MODELS["random_forest"]
    fixed = {k: next(g for g in grid[k] if str(g) == v) for k, v in rf_params.items()}
    teacher = clone(est).set_params(**fixed)
    X_train = pd.concat([design(dev, PHYS), design(pseudo, PHYS)], ignore_index=True)
    y_train = np.concatenate([binary(dev.condition.values), pseudo.target.values])
    teacher.fit(X_train, y_train)
    scores_all("rule_as_teacher", None, teacher.predict(design(test, PHYS)), teacher.predict(design(p22, PHYS)),
               teacher.predict(design(p58, PHYS)), dev, test, p22, p58, res)
    res["methods"]["rule_as_teacher"]["pseudo_labels"] = pd.Series(pseudo.target).value_counts().to_dict()

    # ---- the valve truth: S read from the trend against the true S (simulation test)
    st = test[test.condition == "stiction"].copy()
    st["vis"] = pd.cut(st.slip_to_load, [0, 1, 2, np.inf], labels=["buried", "marginal", "visible"])
    valve = {"all": {"runs": int(st.s_travel_co.notna().sum()),
                     "spearman": round(float(spearmanr(st.true_s, st.s_travel_co, nan_policy="omit").correlation), 3),
                     "median_abs_error_pct": round(float(np.nanmedian(np.abs(st.s_travel_co - st.true_s))), 2)}}
    for k, g in st.groupby("vis", observed=True):
        valve[str(k)] = {"runs": int(g.s_travel_co.notna().sum()),
                         "spearman": round(float(spearmanr(g.true_s, g.s_travel_co, nan_policy="omit").correlation), 3),
                         "median_abs_error_pct": round(float(np.nanmedian(np.abs(g.s_travel_co - g.true_s))), 2)}
    valve["healthy_valves_median_reading_pct"] = round(float(test[test.condition != "stiction"].s_travel_co.median()), 2)
    valve["points"] = st[["true_s", "s_travel_co", "loop_type", "slip_to_load"]].round(3).to_dict("records")
    res["valve_truth"] = valve
    log.info("S from the trend vs true S (simulation test): %s", {k: v for k, v in valve.items() if k != "points"})

    # ---- the decision rule, as written before the run
    r = res["methods"]["rule"]
    lift = lambda e: e["caught"] - e["false_alarms"]                                      # noqa: E731
    verdicts = {}
    for name, e in res["methods"].items():
        if name == "rule":
            continue
        ok1 = e["test_f1_ci"][0] > r["test"]["f1"]
        ok2 = lift(e["plant58"]) - lift(r["plant58"]) > 2
        ok3 = lift(e["plant22"]) - lift(r["plant22"]) >= -1
        verdicts[name] = {"sim_test": ok1, "plant58": ok2, "plant22": ok3, "replaces_rule": bool(ok1 and ok2 and ok3)}
    res["verdicts"] = verdicts
    winners = [k for k, v in verdicts.items() if v["replaces_rule"]]
    log.info("methods that replace the rule under the protocol: %s", winners or "none")
    res["feature_medians"] = {
        "simulation_dev": dev[["harris", "harris_p", "regularity", "shape", "s_travel"]].median().round(3).to_dict(),
        "plant_labelled": real[["harris", "regularity", "shape", "s_travel"]].median().round(3).to_dict()}
    res["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    dev.to_csv(HERE / "results_week8_dev_features.csv", index=False)
    test.to_csv(HERE / "results_week8_test_features.csv", index=False)
    real.to_csv(HERE / "results_week8_plant_features.csv", index=False)
    log.info("saved %s in %ds", OUT.name, res["seconds"])
    return res


if __name__ == "__main__":
    main()
