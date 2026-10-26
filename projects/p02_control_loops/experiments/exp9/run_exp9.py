"""Experiment 9 - run once, following PROTOCOL_exp9.md (written before this run).

Train on ALL the simulation (240 plants, 480 runs), test on the whole labelled archive
(58 loops, 26 sticky valves; the episode's 22 inside them). Stages, in order:
A default models · B tuned models · C ensembles · D physics-informed · E the rule.
No archive label is used for training, tuning or selection.
"""
from __future__ import annotations

import importlib.util
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
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
from chemai.config import ControlLoopConfig          # noqa: E402
from chemai.models import baseline_predict            # noqa: E402

warnings.filterwarnings("ignore")
P2 = REPO / "projects" / "p02_control_loops"
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("w8", P2 / "week8_physics.py")
w8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w8)
warnings.filterwarnings("ignore")

T0 = time.time()
CFG = ControlLoopConfig()
BASE = ["harris", "regularity", "shape_value", "shape_missing", "cycles"]
PHYS = list(w8.PHYS)


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def stiction_f1(y_true, y_pred) -> float:
    return f1_score(np.asarray(y_true) == "stiction", np.asarray(y_pred) == "stiction", zero_division=0)


SCORER = make_scorer(stiction_f1)

FAM = {
    "logistic_regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
    "svm": (
        make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=0)),
        {"svc__C": [0.3, 1, 3, 10, 30], "svc__gamma": ["scale", 0.03, 0.1, 0.3]}),
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=0),
        {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 3, 5, 10]}),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1),
        {"max_depth": [None, 4, 8], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", None]}),
    "gradient_boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
        {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300],
         "l2_regularization": [0.0, 1.0]}),
}

import os                                              # noqa: E402
SMOKE = bool(os.environ.get("EXP9_SMOKE"))            # code check only: tiny grids, archive numbers hidden, nothing saved
if SMOKE:
    FAM = {n: (clone(e).set_params(n_estimators=20) if n == "random_forest" else e,
               {k: v[:1] for k, v in g.items()}) for n, (e, g) in FAM.items()}

# ======================================================================= data
dev = pd.read_csv(P2 / "results_week8_dev_features.csv")
sim_new = pd.read_csv(P2 / "results_week8_test_features.csv")
assert sim_new.loop.str.startswith("T-").all() and not dev.loop.str.startswith("T-").any()
train = pd.concat([dev, sim_new], ignore_index=True)
Y = train.condition.values
G = train.loop.values
CLASSES = np.unique(Y)
plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
p22 = p58[(p58.regularity > CFG.regularity_threshold) & (p58.cycles >= CFG.min_cycles)].reset_index(drop=True)
assert len(train) == 480 and train.loop.nunique() == 240
assert len(p58) == 58 and (p58.label == "stiction").sum() == 26
assert len(p22) == 22 and (p22.label == "stiction").sum() == 14
log(f"training: {len(train)} runs / {train.loop.nunique()} plants · archive: 58 loops (26 sticky), 22 loops (14 sticky)")
unl = w8.real_table(CFG, labelled=False)
log(f"unlabelled real loops: {len(unl)}")


def binary(pred) -> np.ndarray:
    return np.where(np.asarray(pred) == "stiction", "stiction", "other")


def truth(df) -> np.ndarray:
    return np.where(df.label.values == "stiction", "stiction", "other")


def counts(t, p) -> dict:
    t, p = np.asarray(t) == "stiction", np.asarray(p) == "stiction"
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "of": tp + fn, "false_alarms": fp,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def boot_ci(t, p, n=2000, seed=0) -> list:
    rng = np.random.default_rng(seed)
    t, p = np.asarray(t), np.asarray(p)
    idx = rng.integers(0, len(t), size=(n, len(t)))
    vals = [stiction_f1(t[i], p[i]) for i in idx]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


RES = {"protocol": "PROTOCOL_exp9.md", "n": {"train_runs": len(train), "train_plants": int(train.loop.nunique()),
                                             "archive58": 58, "archive22": 22, "unlabelled_real": len(unl)},
       "methods": {}}


def record(stage, name, sim_pred, a58, a22, extra=None):
    e = {"stage": stage}
    e["sim"] = counts(Y, sim_pred) if sim_pred is not None else None
    e["a58"] = counts(truth(p58), binary(a58))
    e["a58"]["f1_ci"] = boot_ci(truth(p58), binary(a58))
    e["a22"] = counts(truth(p22), binary(a22))
    if extra:
        e.update(extra)
    RES["methods"][name] = e
    s = e["sim"]["f1"] if e["sim"] else float("nan")
    if SMOKE:
        log(f"{stage} {name:34s} ran (archive numbers hidden in the code check)")
        return
    log(f"{stage} {name:34s} sim F1 {s:.3f} | 58: {e['a58']['caught']}/26 FA {e['a58']['false_alarms']} "
        f"F1 {e['a58']['f1']:.3f} {e['a58']['f1_ci']} | 22: {e['a22']['caught']}/14 FA {e['a22']['false_alarms']}")


# ======================================================================= helpers
def tune(est, grid, X, y, g):
    gs = GridSearchCV(clone(est), grid, cv=GroupKFold(4), scoring=SCORER, n_jobs=2)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, {k: str(v) for k, v in gs.best_params_.items()}


def proba(m, X):
    assert list(m.classes_) == list(CLASSES)
    return m.predict_proba(X)


def vote(models, X):
    p = np.mean([proba(m, X) for m in models], axis=0)
    return CLASSES[np.argmax(p, axis=1)]


def stack_fit(models, X, y, g):
    """Meta-model on group-out-of-fold probabilities of the given (already tuned) models."""
    K = len(CLASSES)
    oof = np.zeros((len(X), len(models) * K))
    for tr, te in GroupKFold(4).split(X, y, g):
        for j, m in enumerate(models):
            mm = clone(m).fit(X.iloc[tr], y[tr])
            oof[te, j * K:(j + 1) * K] = proba(mm, X.iloc[te])
    return LogisticRegression(max_iter=5000, class_weight="balanced").fit(oof, y)


def stack_predict(meta, models, X):
    return meta.predict(np.hstack([proba(m, X) for m in models]))


def stage(feats, tuned, ensembles, tag):
    """Nested out-of-fold predictions on the simulation, then the final fit on all of it."""
    X = w8.design(train, feats)
    names = list(FAM) + (["voting", "stacking"] if ensembles else [])
    oof = {n: np.empty(len(train), dtype=object) for n in names}
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, Y, G)):
        fitted = []
        for name, (est, grid) in FAM.items():
            m = tune(est, grid, X.iloc[tr], Y[tr], G[tr])[0] if tuned else clone(est).fit(X.iloc[tr], Y[tr])
            fitted.append(m)
            oof[name][te] = m.predict(X.iloc[te])
        if ensembles:
            oof["voting"][te] = vote(fitted, X.iloc[te])
            meta = stack_fit(fitted, X.iloc[tr], Y[tr], G[tr])
            oof["stacking"][te] = stack_predict(meta, fitted, X.iloc[te])
        log(f"  {tag}: outer fold {k + 1}/5 done")
    final, params = {}, {}
    for name, (est, grid) in FAM.items():
        if tuned:
            final[name], params[name] = tune(est, grid, X, Y, G)
        else:
            final[name], params[name] = clone(est).fit(X, Y), "defaults"
    X58, X22 = w8.design(p58, feats), w8.design(p22, feats)
    preds = {n: (final[n].predict(X58), final[n].predict(X22)) for n in FAM}
    if ensembles:
        models = list(final.values())
        preds["voting"] = (vote(models, X58), vote(models, X22))
        meta = stack_fit(models, X, Y, G)
        preds["stacking"] = (stack_predict(meta, models, X58), stack_predict(meta, models, X22))
    return oof, preds, final, params


# ======================================================================= A · default models
oofA, prA, _, _ = stage(BASE, tuned=False, ensembles=False, tag="A")
for n in FAM:
    record("A", f"default {n}", oofA[n], *prA[n])

# ======================================================================= B + C · tuned models and ensembles
oofB, prB, finB, parB = stage(BASE, tuned=True, ensembles=True, tag="B+C")
for n in FAM:
    record("B", f"tuned {n}", oofB[n], *prB[n], extra={"params": parB[n]})
for n in ("voting", "stacking"):
    record("C", f"ensemble {n}", oofB[n], *prB[n])

# ======================================================================= D · physics-informed
oofD, prD, finD, parD = stage(PHYS, tuned=True, ensembles=True, tag="D")
for n in FAM:
    record("D", f"physics {n}", oofD[n], *prD[n], extra={"params": parD[n]})
record("D", "physics ensemble voting", oofD["voting"], *prD["voting"])
record("D", "physics ensemble stacking", oofD["stacking"], *prD["stacking"])

sim_scores = {n: stiction_f1(Y, oofD[n]) for n in FAM}
best = max(sim_scores, key=sim_scores.get)
RES["hybrid_model"] = {"chosen": best, "by_sim_f1": {k: round(v, 3) for k, v in sim_scores.items()}}


def hybrid(df, model_pred):
    z = w8.clear_zone(df, CFG)
    return np.where(z == "", binary(model_pred), z)


record("D", f"hybrid (rule + physics {best})", hybrid(train, oofD[best]),
       hybrid(p58, prD[best][0]), hybrid(p22, prD[best][1]),
       extra={"grey_zone_share": {"sim": round(float((w8.clear_zone(train, CFG) == "").mean()), 3),
                                  "a58": round(float((w8.clear_zone(p58, CFG) == "").mean()), 3),
                                  "a22": round(float((w8.clear_zone(p22, CFG) == "").mean()), 3)}})

z = w8.clear_zone(unl, CFG)
pseudo = unl[z != ""].copy()
pseudo["target"] = z[z != ""]
rf_est, rf_grid = FAM["random_forest"]
rf_fixed = {k: next(g for g in rf_grid[k] if str(g) == v) for k, v in parD["random_forest"].items()}
teacher = clone(rf_est).set_params(**rf_fixed)
teacher.fit(pd.concat([w8.design(train, PHYS), w8.design(pseudo, PHYS)], ignore_index=True),
            np.concatenate([binary(Y), pseudo.target.values]))
record("D", "rule as teacher", None, teacher.predict(w8.design(p58, PHYS)), teacher.predict(w8.design(p22, PHYS)),
       extra={"pseudo_labels": pd.Series(pseudo.target).value_counts().to_dict(), "params": parD["random_forest"]})

# ======================================================================= E · the rule
rule = lambda df: baseline_predict(df, CFG).values                                       # noqa: E731
record("E", "the rule", rule(train), rule(p58), rule(p22))

# ======================================================================= reading rule
r58, r22 = RES["methods"]["the rule"]["a58"], RES["methods"]["the rule"]["a22"]
lift = lambda e: e["caught"] - e["false_alarms"]                                         # noqa: E731
for name, e in RES["methods"].items():
    if name == "the rule":
        continue
    d58, d22 = lift(e["a58"]) - lift(r58), lift(e["a22"]) - lift(r22)
    e["vs_rule"] = {"lift58": d58, "lift22": d22, "clearly_beats_rule": bool(d58 > 2 and d22 >= -1)}
RES["seconds"] = round(time.time() - T0)
if SMOKE:
    log("code check finished - nothing saved")
    sys.exit(0)
(HERE / "results_exp9.json").write_text(json.dumps(RES, indent=2, default=str))
rows = []
for name, e in RES["methods"].items():
    rows.append({"stage": e["stage"], "method": name, "sim_f1": e["sim"]["f1"] if e["sim"] else None,
                 "a58_caught": e["a58"]["caught"], "a58_fa": e["a58"]["false_alarms"], "a58_f1": e["a58"]["f1"],
                 "a58_ci": e["a58"]["f1_ci"], "a22_caught": e["a22"]["caught"], "a22_fa": e["a22"]["false_alarms"],
                 "a22_f1": e["a22"]["f1"], "lift58_vs_rule": e.get("vs_rule", {}).get("lift58"),
                 "clearly_beats_rule": e.get("vs_rule", {}).get("clearly_beats_rule")})
pd.DataFrame(rows).to_csv(HERE / "results_exp9_table.csv", index=False)
log("saved results_exp9.json and results_exp9_table.csv")
