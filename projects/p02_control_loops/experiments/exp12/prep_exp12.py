"""Experiment 12 - inputs for the network variants (PROTOCOL_exp12.md). Produces no result.

1. The 11 rule-labelled unlabelled real loops (week8_physics.clear_zone), as CO/CV windows built exactly like
   experiment 10's archive windows.
2. The rule's clear zone on the 58 labelled loops, for the hybrids.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

REPO = Path(__file__).resolve().parents[4]   # the repository root
sys.path.insert(0, str(REPO))
from chemai.config import DATA_DIR, ControlLoopConfig                 # noqa: E402
from chemai.data import load_isdb, load_sacac                          # noqa: E402
from chemai.features.loop_performance import oscillation_regularity    # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
P2 = REPO / "projects" / "p02_control_loops"
_spec = importlib.util.spec_from_file_location("w8", P2 / "week8_physics.py")
w8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w8)
PTS, WIN, STRIDE, N_INF, MIN_SAMPLES = 32, 192, 96, 64, 200          # as prep_exp10.py
CFG = ControlLoopConfig()


def rescale(sp, pv, op):                                               # copied from prep_exp10.py
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


# ---------------------------------------------------------------- 1 · the pseudo-labelled loops
unl = w8.real_table(CFG, labelled=False)
z = w8.clear_zone(unl, CFG)
pseudo = unl[z != ""].assign(target=z[z != ""])[["name", "target"]].reset_index(drop=True)
assert len(pseudo) == 11 and (pseudo.target == "stiction").sum() == 9
want = dict(zip(pseudo.name, pseudo.target))
meta, ws, run = [], [], []


def add(name, sp, pv, op):
    if name not in want or not np.isfinite(pv).all() or len(pv) < MIN_SAMPLES:
        return
    X, period, fb = rescale(sp, pv, op)
    w = inference_windows(X)
    k = len(meta)
    ws.append(w); run.extend([k] * len(w))
    meta.append({"name": name, "target": want[name], "period": period, "fallback": fb, "n": len(pv), "windows": len(w)})


for rec in load_sacac(DATA_DIR / "sacac"):
    if rec.label is not None:
        continue
    d = rec.data
    sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
    add(rec.name, sp, d.pv.to_numpy(), d.op.to_numpy())
for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
    if x.in_sacac or x.label is not None:
        continue
    d = x.data
    sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
    add(x.key, sp, d.pv.to_numpy(), d.op.to_numpy())
meta = pd.DataFrame(meta)
assert len(meta) == 11 and set(meta.name) == set(pseudo.name), "pseudo loops differ from week8_physics"
np.savez_compressed(HERE / "pseudo_windows_exp12.npz", w=np.concatenate(ws), run=np.array(run))
meta.to_csv(HERE / "pseudo_meta_exp12.csv", index=False)

# ---------------------------------------------------------------- 2 · the rule's clear zone on the 58
plant = pd.read_csv(P2 / "results_week8_plant_features.csv")
p58 = plant[plant.tier == "stated"].reset_index(drop=True)
pd.DataFrame({"name": p58.name, "zone": w8.clear_zone(p58, CFG)}).to_csv(HERE / "zone58_exp12.csv", index=False)
print(f"pseudo loops {len(meta)} ({int((meta.target == 'stiction').sum())} sticky), windows {sum(meta.windows)}, "
      f"period fallback {int(meta.fallback.sum())}; zone58 written")
