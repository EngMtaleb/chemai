"""Experiment 11 - CNN and LSTM (methods 8-9), run once under PROTOCOL_exp11.md.

Inputs and network settings exactly as experiment 10 (../exp10/inputs_exp10.npz). Early stopping uses
15 % of the simulated training plants, as in experiment 10, so every real training loop stays in training.
Real windows are weighted so that, together, they weigh as much as the simulated windows.
EXP11_SMOKE=1: one epoch, archive numbers hidden, nothing saved.
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

SMOKE = bool(os.environ.get("EXP11_SMOKE"))
EPOCHS = 1 if SMOKE else 30
SEEDS = (0,) if SMOKE else (0, 1, 2)
HERE = Path(__file__).parent
E10 = HERE.parent / "exp10"
T0 = time.time()
MAP = {"stiction": "stiction", "tuning_sluggish": "tuning_sluggish", "tuning_tight": "tuning_tight",
       "external_oscillation": "external_oscillation", "healthy": "healthy", "no_oscillation": "healthy"}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


D = np.load(E10 / "inputs_exp10.npz")
sim = pd.read_csv(E10 / "sim_meta_exp10.csv")
real = pd.read_csv(E10 / "real_meta_exp10.csv").merge(pd.read_csv(HERE / "folds_exp11.csv")[["name", "fold"]], on="name")
assert len(real) == 58 and real.fold.notna().all()
CLASSES = sorted(sim.condition.unique())
CI = {c: i for i, c in enumerate(CLASSES)}
STI = CI["stiction"]
y_run = sim.condition.map(CI).values
g_run = sim.loop.values
tr_w, tr_run = D["tr_w"], D["tr_run"]
r_w, r_run = D["real_w"], D["real_run"]
truth = (real.label == "stiction").values


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
    m = keras.Model(inp, layers.Dense(len(CLASSES), activation="softmax")(x))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return m


def fit(kind, seed, real_train_idx):
    """Simulation windows (15 % of plants held for early stopping) + the windows of the given real loops."""
    plants = np.unique(g_run)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(plants, size=int(round(0.15 * len(plants))), replace=False))
    is_val_run = np.array([g in val for g in g_run])
    m_tr, m_va = ~is_val_run[tr_run], is_val_run[tr_run]
    X, y, w = [tr_w[m_tr]], [y_run[tr_run[m_tr]]], [np.ones(int(m_tr.sum()))]
    if len(real_train_idx):
        sel = np.isin(r_run, real_train_idx)
        n_real = int(sel.sum())
        X.append(r_w[sel])
        y.append(real.label.map(MAP).map(CI).values[r_run[sel]].astype(int))
        w.append(np.full(n_real, m_tr.sum() / n_real))
    model = build(kind, seed)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    h = model.fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(w),
                  validation_data=(tr_w[m_va], y_run[tr_run[m_va]]), epochs=EPOCHS, batch_size=64,
                  callbacks=[es], verbose=0, shuffle=True)
    return model, len(h.history["loss"])


def loop_proba(model, idx):
    sel = np.isin(r_run, idx)
    p = model.predict(r_w[sel], batch_size=512, verbose=0)
    out, cnt = np.zeros((len(real), len(CLASSES))), np.zeros(len(real))
    np.add.at(out, r_run[sel], p)
    np.add.at(cnt, r_run[sel], 1)
    return out / np.maximum(cnt, 1)[:, None]


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


mapped = real.label.isin(MAP).values
all_idx = np.arange(len(real))
RES = {"protocol": "PROTOCOL_exp11.md", "networks": {}, "names": real.name.tolist()}
for kind in ("cnn", "lstm"):
    # condition A: simulation only
    pA = np.mean([loop_proba(fit(kind, s, np.array([], int))[0], all_idx) for s in SEEDS], axis=0)
    log(f"  {kind}: condition A done")
    # condition B: + mapped real loops of the other folds
    pB = np.zeros_like(pA)
    for f in range(5):
        te = np.where(real.fold.values == f)[0]
        tr = np.where((real.fold.values != f) & mapped)[0]
        pB[te] = np.mean([loop_proba(fit(kind, s, tr)[0], te)[te] for s in SEEDS], axis=0)
        log(f"  {kind}: condition B fold {f + 1}/5 done ({len(tr)} real training loops)")
    cA, cB = pA.argmax(1) == STI, pB.argmax(1) == STI
    A, B = counts(truth, cA), counts(truth, cB)
    A["f1_ci"], B["f1_ci"] = boot_ci(truth, cA), boot_ci(truth, cB)
    RES["networks"][kind] = {"sim_only": A, "sim_plus_real": B,
                             "calls_A": [CLASSES[i] for i in pA.argmax(1)], "calls_B": [CLASSES[i] for i in pB.argmax(1)],
                             "vs_rule_B": B["caught"] - B["false_alarms"] - 10,
                             "clearly_beats_rule_B": bool(B["caught"] - B["false_alarms"] - 10 > 2)}
    if SMOKE:
        log(f"{kind}: ran (archive numbers hidden in the code check)")
    else:
        log(f"{kind:5s} sim only: {A['caught']}/26 FA {A['false_alarms']} F1 {A['f1']:.3f} | + real: "
            f"{B['caught']}/26 FA {B['false_alarms']} F1 {B['f1']:.3f} {B['f1_ci']}")
if SMOKE:
    log("code check finished - nothing saved")
    sys.exit(0)
(HERE / "results_exp11_dl.json").write_text(json.dumps(RES, indent=2, default=str))
log("saved results_exp11_dl.json")
