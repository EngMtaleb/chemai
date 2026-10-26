"""Experiment 10 - LSTM and 1D-CNN on CO and CV, run once under PROTOCOL_exp10.md.

Inputs come from prep_exp10.py. Settings are fixed in the protocol; nothing is tuned.
EXP10_SMOKE=1 runs one epoch per network as a code check, hides archive numbers and saves nothing.
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
from sklearn.model_selection import GroupKFold  # noqa: E402

SMOKE = bool(os.environ.get("EXP10_SMOKE"))
EPOCHS = 1 if SMOKE else 30
KINDS = [k for k in os.environ.get("EXP10_KINDS", "cnn,lstm").split(",") if k]
HERE = Path(__file__).parent
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


D = np.load(HERE / "inputs_exp10.npz")
sim = pd.read_csv(HERE / "sim_meta_exp10.csv")
real = pd.read_csv(HERE / "real_meta_exp10.csv")
CLASSES = sorted(sim.condition.unique())
STI = CLASSES.index("stiction")
y_run = sim.condition.map({c: i for i, c in enumerate(CLASSES)}).values
g_run = sim.loop.values
tr_w, tr_run = D["tr_w"], D["tr_run"]
si_w, si_run = D["sim_inf_w"], D["sim_inf_run"]
r_w, r_run = D["real_w"], D["real_run"]
assert len(sim) == 480 and len(real) == 58 and (real.label == "stiction").sum() == 26 and real.in22.sum() == 22
log(f"train windows {len(tr_w)} · simulation windows {len(si_w)} · archive windows {len(r_w)} · classes {CLASSES}")


def build(kind: str, seed: int) -> keras.Model:
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


def fit(kind: str, runs: np.ndarray, seed: int):
    """Train on the windows of `runs`; 15 % of their plants (by plant) are the early-stopping set."""
    plants = np.unique(g_run[runs])
    rng = np.random.default_rng(seed)
    val = set(rng.choice(plants, size=max(1, int(round(0.15 * len(plants)))), replace=False))
    is_val = np.array([g_run[r] in val for r in runs])
    m_tr, m_va = np.isin(tr_run, runs[~is_val]), np.isin(tr_run, runs[is_val])
    model = build(kind, seed)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    h = model.fit(tr_w[m_tr], y_run[tr_run[m_tr]], validation_data=(tr_w[m_va], y_run[tr_run[m_va]]),
                  epochs=EPOCHS, batch_size=64, callbacks=[es], verbose=0, shuffle=True)
    return model, len(h.history["loss"])


def record_proba(model, W, idx, n):
    p = model.predict(W, batch_size=512, verbose=0)
    out, cnt = np.zeros((n, len(CLASSES))), np.zeros(n)
    np.add.at(out, idx, p)
    np.add.at(cnt, idx, 1)
    return out / np.maximum(cnt, 1)[:, None]


def counts(t, p) -> dict:
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "of": tp + fn, "false_alarms": fp,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def boot_ci(t, p, n=2000, seed=0) -> list:
    rng = np.random.default_rng(seed)
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    vals = [f1_score(t[i], p[i], zero_division=0) for i in rng.integers(0, len(t), size=(n, len(t)))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


truth58 = (real.label == "stiction").values
in22 = real.in22.values.astype(bool)
sim_truth = (y_run == STI)
OUT = {"protocol": "PROTOCOL_exp10.md", "classes": CLASSES, "networks": {}}

for kind in KINDS:
    e = {"epochs": {}}
    # ---- simulation score: GroupKFold(5) by plant, out of fold, seed 0
    oof = np.zeros((len(sim), len(CLASSES)))
    all_runs = np.arange(len(sim))
    for k, (tr, te) in enumerate(GroupKFold(5).split(all_runs, y_run, g_run)):
        model, ep = fit(kind, tr, 0)
        sel = np.isin(si_run, te)
        p = record_proba(model, si_w[sel], si_run[sel], len(sim))
        oof[te] = p[te]
        e["epochs"][f"fold{k + 1}"] = ep
        log(f"  {kind}: fold {k + 1}/5 ({ep} epochs)")
    e["sim"] = counts(sim_truth, oof.argmax(1) == STI)
    e["sim"]["accuracy_5_classes"] = round(float((oof.argmax(1) == y_run).mean()), 3)
    # ---- final: all 480 runs, seeds 0-2; the archive call is the mean probability
    probs, per_seed = [], []
    for seed in (0, 1, 2):
        model, ep = fit(kind, all_runs, seed)
        p = record_proba(model, r_w, r_run, len(real))
        probs.append(p)
        call = p.argmax(1) == STI
        per_seed.append({"seed": seed, "epochs": ep, "a58": counts(truth58, call)})
        log(f"  {kind}: final seed {seed} ({ep} epochs)")
    pm = np.mean(probs, axis=0)
    call = pm.argmax(1) == STI
    e["a58"] = counts(truth58, call)
    e["a58"]["f1_ci"] = boot_ci(truth58, call)
    e["a22"] = counts(truth58[in22], call[in22])
    e["per_seed"] = per_seed
    e["archive_calls"] = dict(zip(real.name, [CLASSES[i] for i in pm.argmax(1)]))
    d58 = e["a58"]["caught"] - e["a58"]["false_alarms"] - (12 - 2)
    d22 = e["a22"]["caught"] - e["a22"]["false_alarms"] - (12 - 2)
    e["vs_rule"] = {"lift58": d58, "lift22": d22, "clearly_beats_rule": bool(d58 > 2 and d22 >= -1)}
    OUT["networks"][kind] = e
    if SMOKE:
        log(f"{kind}: code path ran (archive numbers hidden in the code check)")
        continue
    seeds_txt = " ".join(f"s{s['seed']}:{s['a58']['caught']}/{s['a58']['false_alarms']}" for s in per_seed)
    log(f"{kind:5s} sim F1 {e['sim']['f1']:.3f} (5-class acc {e['sim']['accuracy_5_classes']}) | 58: {e['a58']['caught']}/26 "
        f"FA {e['a58']['false_alarms']} F1 {e['a58']['f1']:.3f} {e['a58']['f1_ci']} | 22: {e['a22']['caught']}/14 "
        f"FA {e['a22']['false_alarms']} | seeds {seeds_txt}")

OUT["seconds"] = round(time.time() - T0)
if SMOKE:
    log("code check finished - nothing saved")
    sys.exit(0)
(HERE / "results_exp10.json").write_text(json.dumps(OUT, indent=2, default=str))
log("saved results_exp10.json")
