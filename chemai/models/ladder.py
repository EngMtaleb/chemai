"""The model ladder: seven machine-learning models, tuned, then an ensemble -
the same steps for every question, so the questions can be compared.

Rules every rung follows:

  * tuning sees development data only, split by GROUP (a plant, an episode, an
    experiment) - never by row, because rows of one group share one membrane;
  * a model's development score is out-of-fold: the fold that scores a row never
    helped fit or tune the model that predicts it... for the fit; the settings are
    chosen on the same grouped folds, so that score is mildly optimistic - the
    score that decides is the test, run once;
  * the ensemble is fixed before the test: the mean of the three best models'
    probabilities (or predictions), ranked on development.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR

warnings.filterwarnings("ignore")

SEED = 7


def _pre(scale: bool = True) -> list:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def classifiers(seed: int = SEED) -> dict:
    """Seven classifiers and their grids (small on purpose: tuning is not the point)."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "logistic_regression": (Pipeline(_pre() + [("m", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
                                {"m__C": [0.01, 0.1, 1.0, 10.0]}),
        "k_nearest_neighbours": (Pipeline(_pre() + [("m", KNeighborsClassifier())]),
                                 {"m__n_neighbors": [25, 75, 151], "m__weights": ["uniform", "distance"]}),
        "svm_rbf": (Pipeline(_pre() + [("k", Nystroem(n_components=400, random_state=seed)),
                                       ("m", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto"),
                                                                    method="sigmoid", cv=3))]),
                    {"k__gamma": [0.01, 0.05], "m__estimator__C": [0.1, 1.0]}),
        "random_forest": (Pipeline(_pre(False) + [("m", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=seed))]),
            {"m__max_depth": [6, 12], "m__min_samples_leaf": [5, 30]}),
        "extra_trees": (Pipeline(_pre(False) + [("m", ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=seed))]),
            {"m__max_depth": [6, 12], "m__min_samples_leaf": [5, 30]}),
        "xgboost": (XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  n_jobs=2, random_state=seed, eval_metric="logloss", verbosity=0),
                    {"max_depth": [3, 5], "min_child_weight": [1, 10]}),
        "lightgbm": (LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, n_jobs=2, random_state=seed, verbose=-1),
                     {"num_leaves": [7, 31], "min_child_samples": [20, 80]}),
    }


def regressors(seed: int = SEED) -> dict:
    """Seven regressors and their grids."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    return {
        "ridge": (Pipeline(_pre() + [("m", Ridge())]), {"m__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "k_nearest_neighbours": (Pipeline(_pre() + [("m", KNeighborsRegressor())]),
                                 {"m__n_neighbors": [25, 75, 151], "m__weights": ["uniform", "distance"]}),
        "svr_rbf": (Pipeline(_pre() + [("m", SVR(cache_size=500))]),
                    {"m__C": [0.3, 3.0], "m__epsilon": [0.05, 0.2]}),
        "random_forest": (Pipeline(_pre(False) + [("m", RandomForestRegressor(n_estimators=300, n_jobs=2,
                                                                                random_state=seed))]),
                          {"m__max_depth": [6, 12], "m__min_samples_leaf": [5, 30]}),
        "extra_trees": (Pipeline(_pre(False) + [("m", ExtraTreesRegressor(n_estimators=300, n_jobs=2,
                                                                            random_state=seed))]),
                        {"m__max_depth": [6, 12], "m__min_samples_leaf": [5, 30]}),
        "xgboost": (XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=2, random_state=seed, verbosity=0),
                    {"max_depth": [3, 5], "min_child_weight": [1, 10]}),
        "lightgbm": (LGBMRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, n_jobs=2, random_state=seed, verbose=-1),
                     {"num_leaves": [7, 31], "min_child_samples": [20, 80]}),
    }


@dataclass
class Tuned:
    name: str
    params: dict
    cv_score: float
    oof: np.ndarray
    seconds: float


def _predict(model, X, task: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "cls" else model.predict(X)


def _score(y, p, task: str, weight=None) -> float:
    if task == "cls":
        return float(average_precision_score(y, p))
    return -float(mean_absolute_error(y, p, sample_weight=weight))      # higher is better


def tune(name: str, model, grid: dict, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
         task: str = "cls", folds: int = 5, max_rows: int | None = None, seed: int = SEED) -> Tuned:
    """Grid search on grouped folds; returns the best settings and their out-of-fold predictions.

    `max_rows` caps the rows a slow model trains on inside a fold (a random subsample
    of the training fold's rows; the scored fold is always complete)."""
    t0 = time.time()
    cv = list(GroupKFold(folds).split(X, y, groups))
    rng = np.random.default_rng(seed)
    best = (-np.inf, None, None)
    for params in ParameterGrid(grid):
        oof = np.full(len(y), np.nan)
        for tr, te in cv:
            if max_rows and len(tr) > max_rows:
                tr = rng.choice(tr, max_rows, replace=False)
            m = clone(model).set_params(**params)
            m.fit(X.iloc[tr], y[tr])
            oof[te] = _predict(m, X.iloc[te], task)
        s = _score(y, oof, task)
        if s > best[0]:
            best = (s, params, oof)
    return Tuned(name, best[1], best[0], best[2], time.time() - t0)


def fit_final(model, params: dict, X: pd.DataFrame, y: np.ndarray, max_rows: int | None = None, seed: int = SEED):
    idx = np.arange(len(y))
    if max_rows and len(idx) > max_rows:
        idx = np.random.default_rng(seed).choice(idx, max_rows, replace=False)
    m = clone(model).set_params(**params)
    m.fit(X.iloc[idx], y[idx])
    return m


def threshold_at_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """The lowest threshold whose precision on (y, p) is at least `target`
    (development data); the most recall that precision allows."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(prec >= target)
    if not len(ok):
        return float(ps[0]) + 1e-9
    k = ok[-1]
    return float(ps[k])


def threshold_best_f1(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(ps[int(np.argmax(f1))])


def rp(y: np.ndarray, flag: np.ndarray) -> dict:
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, recall=round(rec, 3), precision=round(prec, 3),
                f1=round(2 * rec * prec / max(rec + prec, 1e-9), 3))


def group_bootstrap(groups: np.ndarray, stat, n: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95 % interval of stat(index) by resampling whole groups."""
    rng = np.random.default_rng(seed)
    g = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(list).to_list()
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(g), len(g))
        idx = np.concatenate([g[i] for i in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (np.nan, np.nan)
