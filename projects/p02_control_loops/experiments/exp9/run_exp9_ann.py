"""Experiment 9, addendum 1: a neural network (MLP), run once under PROTOCOL_exp9.md addendum 1.

Same training data (all 480 simulated runs), same archive (58 / 22), same nested CV, same bar.
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
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
from chemai.config import ControlLoopConfig          # noqa: E402

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


MLP = make_pipeline(StandardScaler(), MLPClassifier(solver="adam", max_iter=1500, random_state=0))
GRID = {"mlpclassifier__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
        "mlpclassifier__alpha": [0.0001, 0.001, 0.01, 0.1]}

train = pd.concat([pd.read_csv(P2 / "results_week8_dev_features.csv"),
                   pd.read_csv(P2 / "results_week8_test_features.csv")], ignore_index=True)
Y, G = train.condition.values, train.loop.values
plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
p22 = p58[(p58.regularity > CFG.regularity_threshold) & (p58.cycles >= CFG.min_cycles)].reset_index(drop=True)
assert len(train) == 480 and len(p58) == 58 and len(p22) == 22


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
    vals = [stiction_f1(t[i], p[i]) for i in rng.integers(0, len(t), size=(n, len(t)))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def truth(df):
    return np.where(df.label.values == "stiction", "stiction", "other")


def tune(X, y, g):
    gs = GridSearchCV(clone(MLP), GRID, cv=GroupKFold(4), scoring=make_scorer(stiction_f1), n_jobs=2)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, gs.best_params_


rule_lift58, rule_lift22 = 12 - 2, 12 - 2           # the rule on the archive (checked in check_exp9.py)
out = {"protocol": "PROTOCOL_exp9.md, addendum 1", "methods": {}}
for tag, feats in (("neural network · episode features", BASE), ("neural network · physics features", PHYS)):
    X = w8.design(train, feats)
    oof = np.empty(len(train), dtype=object)
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, Y, G)):
        m, _ = tune(X.iloc[tr], Y[tr], G[tr])
        oof[te] = m.predict(X.iloc[te])
        log(f"  {tag}: outer fold {k + 1}/5")
    final, params = tune(X, Y, G)
    X58, X22 = w8.design(p58, feats), w8.design(p22, feats)
    e = {"sim": counts(Y, oof), "params": {k: str(v) for k, v in params.items()}}
    pred58 = np.where(final.predict(X58) == "stiction", "stiction", "other")
    e["a58"] = counts(truth(p58), pred58)
    e["a58"]["f1_ci"] = boot_ci(truth(p58), pred58)
    e["a22"] = counts(truth(p22), np.where(final.predict(X22) == "stiction", "stiction", "other"))
    seeds = []
    for s in range(5):
        ms = clone(final).set_params(mlpclassifier__random_state=s).fit(X, Y)
        c = counts(truth(p58), np.where(ms.predict(X58) == "stiction", "stiction", "other"))
        seeds.append((c["caught"], c["false_alarms"]))
    e["seeds_a58_caught_fa"] = seeds
    d58 = e["a58"]["caught"] - e["a58"]["false_alarms"] - rule_lift58
    d22 = e["a22"]["caught"] - e["a22"]["false_alarms"] - rule_lift22
    e["vs_rule"] = {"lift58": d58, "lift22": d22, "clearly_beats_rule": bool(d58 > 2 and d22 >= -1)}
    out["methods"][tag] = e
    log(f"{tag:36s} sim F1 {e['sim']['f1']:.3f} | 58: {e['a58']['caught']}/26 FA {e['a58']['false_alarms']} "
        f"F1 {e['a58']['f1']:.3f} {e['a58']['f1_ci']} | 22: {e['a22']['caught']}/14 FA {e['a22']['false_alarms']} "
        f"| seeds 0-4 on 58: {seeds} | {e['params']}")
out["seconds"] = round(time.time() - T0)
(HERE / "results_exp9_ann.json").write_text(json.dumps(out, indent=2, default=str))
log("saved results_exp9_ann.json")
