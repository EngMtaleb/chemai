"""Experiment 10 - build the network inputs (PROTOCOL_exp10.md, 'Input'). Produces no result.

Simulation: the same 480 runs as Week 8 (generate_dataset(8, 2), random_state 42 and 2026).
Archive: the 58 stated-label loops, loaded exactly as week8_physics.real_table does.
"""
from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
from chemai.config import DATA_DIR, ControlLoopConfig                 # noqa: E402
from chemai.data import generate_dataset, load_isdb, load_sacac        # noqa: E402
from chemai.features.loop_performance import oscillation_regularity    # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
P2 = REPO / "projects" / "p02_control_loops"
PTS, WIN, STRIDE, N_TRAIN, N_INF, MIN_SAMPLES = 32, 192, 96, 12, 64, 200
RNG = np.random.default_rng(0)


def rescale(sp, pv, op):
    sp, pv, op = (np.asarray(v, dtype=float) for v in (sp, pv, op))
    n = len(pv)
    period, _ = oscillation_regularity(detrend(sp - pv))
    fallback = not (np.isfinite(period) and 4 <= period <= n / 3)
    if fallback:
        period = n / 12
    co, cv = detrend(op), detrend(pv - sp)
    L = max(int(round(n * PTS / period)), 2)
    xn = np.linspace(0, n - 1, L)
    X = np.stack([np.interp(xn, np.arange(n), co), np.interp(xn, np.arange(n), cv)], axis=1)
    return X.astype(np.float32), float(period), bool(fallback)


def window(X, s):
    w = X[s:s + WIN].copy()
    mu, sd = w.mean(axis=0), w.std(axis=0)
    w = np.where(sd > 1e-9, (w - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    if len(w) < WIN:
        w = np.vstack([np.zeros((WIN - len(w), 2)), w])
    return w.astype(np.float32)


def starts(L):
    return list(range(0, max(L - WIN, 0) + 1, STRIDE))


def inference_windows(X):
    st = starts(len(X))
    idx = np.unique(np.linspace(0, len(st) - 1, min(N_INF, len(st))).round().astype(int))
    return np.stack([window(X, st[i]) for i in idx])


# ---------------------------------------------------------------- simulation
cfg = ControlLoopConfig()
sim_meta, tr_w, tr_run, inf_w, inf_run = [], [], [], [], []
for rs, prefix in ((42, ""), (2026, "T-")):
    specs, runs = generate_dataset(8, 2, dataclasses.replace(cfg, random_state=rs))
    by_id = {s.loop_id: s for s in specs}
    for (loop_id, seed), d in runs.items():
        s = by_id[loop_id]
        X, period, fb = rescale(d.sp, d.pv, d.op)
        k = len(sim_meta)
        st = starts(len(X))
        pick = RNG.choice(len(st), size=min(N_TRAIN, len(st)), replace=False)
        for i in pick:
            tr_w.append(window(X, st[i])); tr_run.append(k)
        w = inference_windows(X)
        inf_w.append(w); inf_run += [k] * len(w)
        sim_meta.append({"loop": prefix + loop_id, "seed": seed, "condition": s.condition,
                         "loop_type": s.loop_type, "period": period, "fallback": fb, "n": len(d)})
    print(f"simulation random_state {rs}: {len(runs)} runs", flush=True)
sim_meta = pd.DataFrame(sim_meta)

# consistency with Week 8: the same runs give the same regularity
w8 = pd.concat([pd.read_csv(P2 / "results_week8_dev_features.csv"),
                pd.read_csv(P2 / "results_week8_test_features.csv")], ignore_index=True)
m = sim_meta.merge(w8[["loop", "seed", "condition"]], on=["loop", "seed"], how="left", suffixes=("", "_w8"))
assert len(sim_meta) == 480 and (m.condition == m.condition_w8).all(), "runs differ from Week 8"

# ---------------------------------------------------------------- archive (as week8_physics.real_table)
plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
p22 = set(p58[(p58.regularity > cfg.regularity_threshold) & (p58.cycles >= cfg.min_cycles)].name)
want = set(p58.name)
real_meta, r_w, r_run = [], [], []


def add(name, label, loop_type, sp, pv, op):
    if name not in want or not np.isfinite(pv).all() or len(pv) < MIN_SAMPLES:
        return
    X, period, fb = rescale(sp, pv, op)
    w = inference_windows(X)
    k = len(real_meta)
    r_w.append(w); r_run.extend([k] * len(w))
    real_meta.append({"name": name, "label": label, "loop_type": loop_type, "in22": name in p22,
                      "period": period, "fallback": fb, "n": len(pv), "windows": len(w)})


for rec in load_sacac(DATA_DIR / "sacac"):
    if rec.label is None:
        continue
    d = rec.data
    sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
    add(rec.name, rec.label, rec.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
    if x.in_sacac or x.label is None:
        continue
    d = x.data
    sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
    add(x.key, x.label, x.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
real_meta = pd.DataFrame(real_meta)
assert len(real_meta) == 58 and set(real_meta.name) == want, "archive differs from Week 8"
assert (real_meta.label == "stiction").sum() == 26 and real_meta.in22.sum() == 22
lab = p58.set_index("name").label
assert (real_meta.set_index("name").label == lab.loc[real_meta.name]).all()

np.savez_compressed(HERE / "inputs_exp10.npz",
                    tr_w=np.stack(tr_w), tr_run=np.array(tr_run),
                    sim_inf_w=np.concatenate(inf_w), sim_inf_run=np.array(inf_run),
                    real_w=np.concatenate(r_w), real_run=np.array(r_run))
sim_meta.to_csv(HERE / "sim_meta_exp10.csv", index=False)
real_meta.to_csv(HERE / "real_meta_exp10.csv", index=False)
print(f"train windows {len(tr_w)}, simulation inference windows {sum(len(w) for w in inf_w)}, "
      f"archive windows {sum(len(w) for w in r_w)}")
print("period fallback: simulation", int(sim_meta.fallback.sum()), "of 480 · archive", int(real_meta.fallback.sum()), "of 58")
print("records shorter than one window: simulation",
      int((sim_meta.n * PTS / sim_meta.period < WIN).sum()), "· archive", int((real_meta.n * PTS / real_meta.period < WIN).sum()))
