"""Experiment 12 - feature models, run once under PROTOCOL_exp12.md.

Seven families (default and tuned), the ensemble of the seven tuned, the ANN with its hybrid and rule-as-teacher
variants, and S alone. Data, folds, weights, scorer and groups exactly as experiment 11.
EXP12_SMOKE=1: tiny grids, archive numbers hidden, nothing saved.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
from chemai.config import ControlLoopConfig          # noqa: E402
from chemai.models import baseline_predict            # noqa: E402

warnings.filterwarnings("ignore")
P2 = REPO / "projects" / "p02_control_loops"
HERE = Path(__file__).parent
E11 = HERE.parent / "exp11"
_spec = importlib.util.spec_from_file_location("w8", P2 / "week8_physics.py")
w8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w8)
warnings.filterwarnings("ignore")
SMOKE = bool(os.environ.get("EXP12_SMOKE"))
T0 = time.time()
CFG = ControlLoopConfig()
BASE = ["harris", "regularity", "shape_value", "shape_missing", "cycles"]
PHYS = list(w8.PHYS)
MAP = {"stiction": "stiction", "tuning_sluggish": "tuning_sluggish", "tuning_tight": "tuning_tight",
       "external_oscillation": "external_oscillation", "healthy": "healthy", "no_oscillation": "healthy"}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def stiction_f1(y_true, y_pred) -> float:
    return f1_score(np.asarray(y_true) == "stiction", np.asarray(y_pred) == "stiction", zero_division=0)


SCORER = make_scorer(stiction_f1)


class XGB(BaseEstimator, ClassifierMixin):
    """XGBoost with string labels and balanced sample weights (the other families use class_weight='balanced')."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3):
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate

    def fit(self, X, y):
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self.m_ = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
                                tree_method="hist", random_state=0, n_jobs=1, verbosity=0)
        self.m_.fit(np.asarray(X, dtype=float), yi, sample_weight=compute_sample_weight("balanced", yi))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# family -> (default estimator, the estimator the grid tunes, grid). For the first five, both are experiment 11's.
FAM = {
    "logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                            {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
    "svm": (make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=0)),
            {"svc__C": [0.3, 1, 3, 10, 30], "svc__gamma": ["scale", 0.03, 0.1, 0.3]}),
    "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=0),
                      {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 3, 5, 10]}),
    "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1),
                      {"max_depth": [None, 4, 8], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", None]}),
    "gradient_boosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300],
                           "l2_regularization": [0.0, 1.0]}),
    "xgboost": (XGB(),
                {"n_estimators": [100, 300], "max_depth": [2, 3, 6], "learning_rate": [0.03, 0.1, 0.3]}),
    "lightgbm": (LGBMClassifier(class_weight="balanced", random_state=0, n_jobs=1, verbose=-1),
                 {"n_estimators": [100, 300], "num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.1],
                  "min_child_samples": [5, 20]}),
}
FIVE = ["logistic_regression", "svm", "decision_tree", "random_forest", "gradient_boosting"]
MLP = (make_pipeline(StandardScaler(), MLPClassifier(solver="adam", max_iter=1500, random_state=0)),
       {"mlpclassifier__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
        "mlpclassifier__alpha": [0.0001, 0.001, 0.01, 0.1]})
if SMOKE:
    FAM = {n: (clone(e).set_params(n_estimators=20) if n == "random_forest" else e, {k: v[:1] for k, v in g.items()})
           for n, (e, g) in FAM.items()}
    MLP = (MLP[0], {k: v[:1] for k, v in MLP[1].items()})

# ---------------------------------------------------------------- data (exactly as experiment 11)
sim = pd.concat([pd.read_csv(P2 / "results_week8_dev_features.csv"),
                 pd.read_csv(P2 / "results_week8_test_features.csv")], ignore_index=True)
sim["group"] = sim.loop
plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
folds = pd.read_csv(E11 / "folds_exp11.csv")
p58 = p58.merge(folds[["name", "group", "fold"]], on="name", how="left")
assert len(sim) == 480 and len(p58) == 58 and p58.fold.notna().all() and (p58.label == "stiction").sum() == 26
unl = w8.real_table(CFG, labelled=False)
z = w8.clear_zone(unl, CFG)
pseudo = unl[z != ""].copy()
pseudo["target"] = z[z != ""]
assert len(pseudo) == 11
log(f"simulation {len(sim)} runs · archive 58 (26 sticky) in 5 folds · pseudo-labelled real loops {len(pseudo)}")


def binary(pred):
    return np.where(np.asarray(pred) == "stiction", "stiction", "other")


def tune(est, grid, X, y, g):
    gs = GridSearchCV(clone(est), grid, cv=GroupKFold(4), scoring=SCORER, n_jobs=2)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, gs.best_params_


def vote(models, X):
    cl = models[0].classes_
    assert all(list(m.classes_) == list(cl) for m in models)
    return cl[np.argmax(np.mean([m.predict_proba(X) for m in models], axis=0), axis=1)]


def training_sets(real_tr):
    """Five-class table (simulation + mapped real loops, repeated) and binary table (for the teacher)."""
    s5 = sim.assign(target=sim.condition.values)
    sb = sim.assign(target=binary(sim.condition.values))
    if real_tr is None or len(real_tr) == 0:
        return s5, sb, 0, 0
    mapped = real_tr[real_tr.label.isin(MAP)]
    k5 = int(round(len(sim) / len(mapped)))
    r5 = pd.concat([mapped.assign(target=mapped.label.map(MAP), group="R:" + mapped.group)] * k5, ignore_index=True)
    kb = int(round(len(sim) / len(real_tr)))
    rb = pd.concat([real_tr.assign(target=np.where(real_tr.label.values == "stiction", "stiction", "other"),
                                   group="R:" + real_tr.group)] * kb, ignore_index=True)
    return (pd.concat([s5, r5], ignore_index=True), pd.concat([sb, rb], ignore_index=True), k5, kb)


def s_threshold(table):
    """S alone: one threshold on S, the best stiction F1 on the training rows."""
    s = w8.design(table, PHYS).s_travel.values
    yb = binary(table.target.values)
    grid_t = np.unique(np.round(np.quantile(s[s > 0], np.linspace(0.05, 0.95, 37)), 3))
    f1s = [stiction_f1(yb, np.where(s > t, "stiction", "other")) for t in grid_t]
    return float(grid_t[int(np.argmax(f1s))])


def fit_predict(real_tr, test):
    t5, tb, k5, kb = training_sets(real_tr)
    X, y, g = w8.design(t5, BASE), t5.target.values, t5.group.values
    XP = w8.design(t5, PHYS)
    Xt, XPt = w8.design(test, BASE), w8.design(test, PHYS)
    zt = w8.clear_zone(test, CFG)
    out, info = {}, {"k5": k5, "kb": kb, "train_rows": len(t5), "params": {}}
    tuned = {}
    for n, (est, grid) in FAM.items():
        out[f"{n} · default"] = clone(est).fit(X, y).predict(Xt)
        tuned[n] = tune(est, grid, X, y, g)
        out[f"{n} · tuned"] = tuned[n][0].predict(Xt)
        info["params"][n] = {k: str(v) for k, v in tuned[n][1].items()}
        log(f"    {n} done")
    out["ensemble · 7 tuned"] = vote([tuned[n][0] for n in FAM], Xt)
    out["check · ensemble of the 5 (experiment 11)"] = vote([tuned[n][0] for n in FIVE], Xt)
    m7, p7 = tune(MLP[0], MLP[1], XP, y, g)
    p_ann = m7.predict(XPt)
    out["ANN"] = p_ann
    out["ANN · hybrid"] = np.where(zt == "", binary(p_ann), zt)
    teacher = clone(MLP[0]).set_params(**p7)
    teacher.fit(pd.concat([w8.design(tb, PHYS), w8.design(pseudo, PHYS)], ignore_index=True),
                np.concatenate([tb.target.values, pseudo.target.values]))
    out["ANN · rule as teacher"] = teacher.predict(XPt)
    theta = s_threshold(tb)
    out["S alone"] = np.where(XPt.s_travel.values > theta, "stiction", "other")
    info["params"]["ann"] = {k: str(v) for k, v in p7.items()}
    info["params"]["s_alone_threshold_std"] = theta
    return {k: binary(v) for k, v in out.items()}, info


predA, infoA = fit_predict(None, p58)
log("condition A (simulation only) done")
METHODS = list(predA)
predB = {m: np.empty(58, dtype=object) for m in METHODS}
infoB = []
for f in range(5):
    te = p58.fold.values == f
    pr, inf = fit_predict(p58[~te], p58[te])
    for m in METHODS:
        predB[m][te] = pr[m]
    inf["fold"] = f
    infoB.append(inf)
    log(f"condition B fold {f + 1}/5 done (real loops repeated x{inf['k5']}, training rows {inf['train_rows']})")
rule = binary(baseline_predict(p58, CFG).values)


def counts(t, p) -> dict:
    t, p = np.asarray(t) == "stiction", np.asarray(p) == "stiction"
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "of": tp + fn, "false_alarms": fp, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def boot_ci(t, p, n=2000, seed=0) -> list:
    rng = np.random.default_rng(seed)
    t, p = np.asarray(t), np.asarray(p)
    vals = [stiction_f1(t[i], p[i]) for i in rng.integers(0, len(t), size=(n, len(t)))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


# reproduction checks against experiment 11 (the same code path must give the same calls)
E11R = json.loads((E11 / "results_exp11_features.json").read_text())
checks = {}
for mine, theirs in [("random_forest · default", "ML · random forest (default)"), ("random_forest · tuned", "tuning · random forest"),
                     ("check · ensemble of the 5 (experiment 11)", "ensemble · soft voting of 5 tuned"), ("ANN", "ANN · physics features")]:
    for cond, pred in (("A", predA), ("B", predB)):
        checks[f"{mine} {cond}"] = bool(list(pred[mine]) == list(E11R["calls"][theirs][cond]))
if not SMOKE:
    log("reproduction of experiment 11: " + ", ".join(f"{k} {'same' if v else 'DIFFERENT'}" for k, v in checks.items()))

truth = binary(p58.label.values)
RES = {"protocol": "PROTOCOL_exp12.md", "reproduction_checks": checks, "condition_A_info": infoA, "condition_B_info": infoB,
       "methods": {}, "calls": {"names": p58.name.tolist()}}
for m in METHODS + ["the rule"]:
    pa, pb = (rule, rule) if m == "the rule" else (predA[m], predB[m])
    A, B = counts(truth, pa), counts(truth, pb)
    A["f1_ci"], B["f1_ci"] = boot_ci(truth, pa), boot_ci(truth, pb)
    lift = lambda e: e["caught"] - e["false_alarms"]                                  # noqa: E731
    RES["methods"][m] = {"sim_only": A, "sim_plus_real": B, "vs_rule_B": lift(B) - 10, "clearly_beats_rule_B": bool(lift(B) - 10 > 2)}
    RES["calls"][m] = {"A": list(pa), "B": list(pb)}
    if SMOKE:
        log(f"{m:44s} ran (archive numbers hidden in the code check)")
    else:
        log(f"{m:44s} sim only: {A['caught']}/26 FA {A['false_alarms']} F1 {A['f1']:.3f} | + real: {B['caught']}/26 "
            f"FA {B['false_alarms']} F1 {B['f1']:.3f} {B['f1_ci']}")
if SMOKE:
    log("code check finished - nothing saved")
    sys.exit(0)
(HERE / "results_exp12_features.json").write_text(json.dumps(RES, indent=2, default=str))
log("saved results_exp12_features.json")
