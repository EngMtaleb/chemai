"""Experiment 12 - the post-hoc check (POSTHOC_exp12.md): the two-class teacher networks without the 11 rule-labelled loops.
Condition B only. Everything else is copied from run_exp12_dl.py.

Original docstring: the networks with the physics (PROTOCOL_exp12.md, with amendment 1), run once.

Hybrids: the rule's clear zone where it is clear, the experiment-11 network call elsewhere (nothing retrained).
Rule as teacher: the CNN and the LSTM with a two-class output, trained like the experiment-11 forest teacher.
EXP12_SMOKE=1: one epoch, one seed, archive numbers hidden, nothing saved.
"""
from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"
import json            # noqa: E402
import sys             # noqa: E402
import time            # noqa: E402
from pathlib import Path  # noqa: E402

import keras           # noqa: E402
import numpy as np     # noqa: E402
import pandas as pd    # noqa: E402
from keras import layers  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

SMOKE = bool(os.environ.get("EXP12_SMOKE"))
EPOCHS = 1 if SMOKE else 30
SEEDS = (0,) if SMOKE else (0, 1, 2)
HERE = Path(__file__).parent
E10, E11 = HERE.parent / "exp10", HERE.parent / "exp11"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


D = np.load(E10 / "inputs_exp10.npz")
P = np.load(HERE / "pseudo_windows_exp12.npz")
sim = pd.read_csv(E10 / "sim_meta_exp10.csv")
real = pd.read_csv(E10 / "real_meta_exp10.csv").merge(pd.read_csv(E11 / "folds_exp11.csv")[["name", "fold"]], on="name")
pmeta = pd.read_csv(HERE / "pseudo_meta_exp12.csv")
zone = real[["name"]].merge(pd.read_csv(HERE / "zone58_exp12.csv"), on="name", how="left").zone.fillna("").values
assert len(real) == 58 and real.fold.notna().all() and len(pmeta) == 11
y_run = (sim.condition.values == "stiction").astype(int)          # binary: sticky or not
g_run = sim.loop.values
tr_w, tr_run = D["tr_w"], D["tr_run"]
r_w, r_run = D["real_w"], D["real_run"]
p_w, p_run = P["w"], P["run"]
p_y = (pmeta.target.values == "stiction").astype(int)[p_run]
p_wt = (12.0 / pmeta.windows.values)[p_run]                       # amendment 1: a pseudo loop weighs one simulated run
truth = (real.label == "stiction").values
r_yb = truth.astype(int)


def build(kind, seed):
    keras.utils.set_random_seed(seed)
    inp = keras.Input((192, 2))
    if kind == "lstm":
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(32)(x)
        x = layers.Dense(32, activation="relu")(x)
    else:
        x = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    m = keras.Model(inp, layers.Dense(2, activation="softmax")(x))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return m


def fit_teacher(kind, seed, real_train_idx):
    """Simulated windows (15 % of plants held out for early stopping) + all labelled real loops given + the pseudo loops."""
    plants = np.unique(g_run)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(plants, size=int(round(0.15 * len(plants))), replace=False))
    is_val_run = np.array([g in val for g in g_run])
    m_tr, m_va = ~is_val_run[tr_run], is_val_run[tr_run]
    X, y, w = [tr_w[m_tr]], [y_run[tr_run[m_tr]]], [np.ones(int(m_tr.sum()))]          # post hoc: no pseudo windows
    if len(real_train_idx):
        sel = np.isin(r_run, real_train_idx)
        n_real = int(sel.sum())
        X.append(r_w[sel]); y.append(r_yb[r_run[sel]]); w.append(np.full(n_real, m_tr.sum() / n_real))
    model = build(kind, seed)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(w),
              validation_data=(tr_w[m_va], y_run[tr_run[m_va]]), epochs=EPOCHS, batch_size=64,
              callbacks=[es], verbose=0, shuffle=True)
    return model


def loop_p(model, idx):
    sel = np.isin(r_run, idx)
    p = model.predict(r_w[sel], batch_size=512, verbose=0)[:, 1]
    out, cnt = np.zeros(len(real)), np.zeros(len(real))
    np.add.at(out, r_run[sel], p)
    np.add.at(cnt, r_run[sel], 1)
    return out / np.maximum(cnt, 1)


def counts(t, p) -> dict:
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "of": tp + fn, "false_alarms": fp, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def boot_ci(t, p, n=2000, seed=0) -> list:
    rng = np.random.default_rng(seed)
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    vals = [f1_score(t[i], p[i], zero_division=0) for i in rng.integers(0, len(t), size=(n, len(t)))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]



RES = {"posthoc": "POSTHOC_exp12.md", "names": real.name.tolist(), "methods": {}}
for kind in ("cnn", "lstm"):
    pB = np.zeros(len(real))
    for f in range(5):
        te = np.where(real.fold.values == f)[0]
        tr = np.where(real.fold.values != f)[0]
        pB[te] = np.mean([loop_p(fit_teacher(kind, s, tr), te)[te] for s in SEEDS], axis=0)
        log(f"  {kind} two-class without the 11: fold {f + 1}/5 done")
    c = pB > 0.5
    B = counts(truth, c); B["f1_ci"] = boot_ci(truth, c)
    RES["methods"][f"{kind} · two-class, without the 11"] = {"sim_plus_real": B, "calls_B": [bool(v) for v in c]}
    log(f"{kind} two-class without the 11 · + real: {B['caught']}/26 FA {B['false_alarms']} F1 {B['f1']:.3f} {B['f1_ci']}")
if SMOKE:
    log("code check finished - nothing saved"); sys.exit(0)
(HERE / "results_exp12_posthoc.json").write_text(json.dumps(RES, indent=2, default=str))
log("saved results_exp12_posthoc.json")
