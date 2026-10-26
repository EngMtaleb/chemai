"""Project 2 - Week 8, part 1: five models against the rule, on simulation.

Run:  python projects/p02_control_loops/compare_models.py
      then python projects/p02_control_loops/plant_check.py (needs data/)

Protocol (fixed before running):
  * development set: the episode's simulated dataset (cfg.random_state = 42).
  * outer GroupKFold(5) by simulated loop -> out-of-fold predictions;
    hyperparameters tuned INSIDE each training fold only (inner GroupKFold(4),
    scoring = macro F1). This is nested cross-validation: the fold that scores
    a model never helped choose its settings.
  * held-out test: NEW simulated plants (cfg.random_state = 2026), touched once,
    after tuning on the whole development set.
  * the rule (regularity > 1 and shape > 0.6) answers one question: stiction or not.
    So every model is also scored on that one question (stiction vs the rest).
  * a variant without Harris is included because the plant dead time is
    unknown - a physical reason, decided before any plant data is used here.

Result, recorded in VALIDATION.md 17.4: every model beats the rule on
simulation; on the plant loops none beats it (PROTOCOL_plant.md, plant_check.py).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from chemai.config import ControlLoopConfig
from chemai.models import (FEATURES, baseline_predict, build_model, design_matrix,
                           simulation_features)

warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "results_compare_models.json")
CLASSES = ["healthy", "stiction", "tuning_tight", "tuning_sluggish", "external_oscillation"]
NO_HARRIS = [f for f in FEATURES if f != "harris"]

MODELS = {
    "logistic_regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        {"logisticregression__C": [0.1, 1.0, 10.0]}, FEATURES),
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=0),
        {"max_depth": [2, 3, 4, 6, None], "min_samples_leaf": [1, 3, 5]}, FEATURES),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1),
        {"max_depth": [None, 6], "min_samples_leaf": [1, 3, 5]}, FEATURES),
    "gradient_boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
        {"learning_rate": [0.05, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300]}, FEATURES),
    "random_forest_no_harris": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1),
        {"max_depth": [None, 6], "min_samples_leaf": [1, 3, 5]}, NO_HARRIS),
}


def stiction_binary(truth, pred) -> dict:
    t = np.asarray(truth) == "stiction"
    p = np.asarray(pred) == "stiction"
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "of": tp + fn, "false_alarms": fp,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def multiclass(truth, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=CLASSES, zero_division=0)
    return {"accuracy": round(accuracy_score(truth, pred), 3),
            "macro_f1": round(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0), 3),
            "per_class": {c: {"precision": round(p[i], 3), "recall": round(r[i], 3),
                              "f1": round(f[i], 3), "n": int(n[i])} for i, c in enumerate(CLASSES)},
            "confusion": confusion_matrix(truth, pred, labels=CLASSES).tolist()}


def visibility(df, pred) -> dict:
    st = df[df.condition == "stiction"].copy()
    st["found"] = np.asarray(pred)[df.condition.values == "stiction"] == "stiction"
    st["vis"] = pd.cut(st.slip_to_load, [0, 1, 2, np.inf], labels=["buried", "marginal", "visible"])
    g = st.groupby("vis", observed=True).found.agg(["size", "mean"])
    return {k: {"runs": int(v["size"]), "recall": round(float(v["mean"]), 3)} for k, v in g.iterrows()}


def bootstrap_ci(df, pred, fn, n=1000, seed=0) -> list:
    """95% interval by resampling whole simulated loops (runs of a loop move together)."""
    rng = np.random.default_rng(seed)
    loops = df.loop.unique()
    idx_by_loop = {l: np.where(df.loop.values == l)[0] for l in loops}
    pred = np.asarray(pred)
    vals = []
    for _ in range(n):
        pick = rng.choice(loops, size=len(loops), replace=True)
        idx = np.concatenate([idx_by_loop[l] for l in pick])
        vals.append(fn(df.condition.values[idx], pred[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def macro_f1(t, p):
    return f1_score(t, p, labels=CLASSES, average="macro", zero_division=0)


def stiction_f1(t, p):
    return stiction_binary(t, p)["f1"]


def main():
    t0 = time.time()
    cfg = ControlLoopConfig()
    dev = simulation_features(cfg)
    test = simulation_features(dataclasses.replace(cfg, random_state=2026))
    test["loop"] = "T-" + test["loop"]
    print(f"simulated dev {len(dev)} runs / {dev.loop.nunique()} loops, "
          f"test {len(test)} runs / {test.loop.nunique()} loops in {time.time()-t0:.0f}s", flush=True)
    Xd_all, Xt_all = design_matrix(dev), design_matrix(test)
    yd, yt, gd = dev.condition.values, test.condition.values, dev.loop.values

    res = {"protocol": __doc__, "n_dev_runs": len(dev), "n_dev_loops": int(dev.loop.nunique()),
           "n_test_runs": len(test), "n_test_loops": int(test.loop.nunique()),
           "class_counts_dev": dev.condition.value_counts().to_dict(), "models": {}}

    # the rule: stiction or not
    rule_dev, rule_test = baseline_predict(dev, cfg).values, baseline_predict(test, cfg).values
    res["models"]["rule"] = {
        "dev_oof": {"stiction": stiction_binary(yd, rule_dev), "visibility": visibility(dev, rule_dev)},
        "test": {"stiction": stiction_binary(yt, rule_test), "visibility": visibility(test, rule_test),
                 "stiction_f1_ci": bootstrap_ci(test, rule_test, stiction_f1)}}

    # the episode's forest, fixed settings, for continuity with the 0.83
    outer = GroupKFold(5)
    pub = np.empty(len(dev), dtype=object)
    for tr, te in outer.split(Xd_all, yd, gd):
        pub[te] = clone(build_model()).fit(Xd_all.iloc[tr], yd[tr]).predict(Xd_all.iloc[te])
    pub_test = clone(build_model()).fit(Xd_all, yd).predict(Xt_all)
    res["models"]["random_forest_published"] = {
        "dev_oof": {**multiclass(yd, pub), "stiction": stiction_binary(yd, pub),
                    "visibility": visibility(dev, pub)},
        "test": {**multiclass(yt, pub_test), "stiction": stiction_binary(yt, pub_test),
                 "visibility": visibility(test, pub_test),
                 "macro_f1_ci": bootstrap_ci(test, pub_test, macro_f1),
                 "stiction_f1_ci": bootstrap_ci(test, pub_test, stiction_f1)}}
    print("published forest done", flush=True)

    for name, (est, grid, feats) in MODELS.items():
        t1 = time.time()
        Xd, Xt = Xd_all[feats], Xt_all[feats]
        oof = np.empty(len(dev), dtype=object)
        chosen = []
        for tr, te in outer.split(Xd, yd, gd):
            gs = GridSearchCV(clone(est), grid, cv=GroupKFold(4), scoring="f1_macro", n_jobs=2)
            gs.fit(Xd.iloc[tr], yd[tr], groups=gd[tr])
            oof[te] = gs.predict(Xd.iloc[te])
            chosen.append(gs.best_params_)
        gs = GridSearchCV(clone(est), grid, cv=GroupKFold(5), scoring="f1_macro", n_jobs=2)
        gs.fit(Xd, yd, groups=gd)
        pt = gs.predict(Xt)
        entry = {"features": feats, "grid": {k: [str(x) for x in v] for k, v in grid.items()},
                 "params_per_outer_fold": [{k: str(v) for k, v in c.items()} for c in chosen],
                 "final_params": {k: str(v) for k, v in gs.best_params_.items()},
                 "inner_cv_macro_f1": round(gs.best_score_, 3),
                 "dev_oof": {**multiclass(yd, oof), "stiction": stiction_binary(yd, oof),
                             "visibility": visibility(dev, oof)},
                 "test": {**multiclass(yt, pt), "stiction": stiction_binary(yt, pt),
                          "visibility": visibility(test, pt),
                          "macro_f1_ci": bootstrap_ci(test, pt, macro_f1),
                          "stiction_f1_ci": bootstrap_ci(test, pt, stiction_f1)}}
        est_final = gs.best_estimator_
        if hasattr(est_final, "feature_importances_"):
            entry["feature_importances"] = dict(zip(feats, np.round(est_final.feature_importances_, 3).tolist()))
        res["models"][name] = entry
        print(f"{name}: dev macroF1 {entry['dev_oof']['macro_f1']} acc {entry['dev_oof']['accuracy']} | "
              f"test macroF1 {entry['test']['macro_f1']} stictionF1 {entry['test']['stiction']['f1']} "
              f"({time.time()-t1:.0f}s)", flush=True)

    res["seconds"] = round(time.time() - t0)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    dev.to_csv(OUT.replace(".json", "_dev_features.csv"), index=False)
    test.to_csv(OUT.replace(".json", "_test_features.csv"), index=False)
    print("saved", OUT, res["seconds"], "s")


if __name__ == "__main__":
    main()
