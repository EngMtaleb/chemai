"""Phase 3 addition - the shared harness (declaration_addendum.md).

The tables, folds and scores are phase 3's own: p3lib.py is imported unchanged from the phase-3 folder, and the folds are
built by the same lines as run.py. This file adds only what the new methods need: the networks' inputs and their fitting.
Prophet's fitting lives in run_prophet.py (it runs in its own environment).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True                      # write nothing into the phase-3 folder when importing p3lib
HERE = Path(__file__).resolve().parent
PHASE3 = HERE.parent
sys.path.insert(0, str(PHASE3))
import p3lib as L  # noqa: E402  (unchanged phase-3 functions)

DATA, OUT3 = PHASE3 / "data", PHASE3 / "out"
SHA = {"derived_daily_indicator_table.csv": "50ad81cb632dfc74e710bad70ce9bf08a014061d0ef7b602be25fc465f1f6374",
       "cip_campaign_ledger.csv": "4f65351d8aaa96d8c804c63d3257978dd22628553e7ab4e12d08f1aa0a375ca7"}
NEW = ["Prophet", "Prophet + level shift", "LSTM", "RNN"]


# ------------------------------------------------------------------ phase 3's tables and folds
def load():
    for name, sha in SHA.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == sha, name
    d = pd.read_csv(DATA / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
    led = pd.read_csv(DATA / "cip_campaign_ledger.csv")
    return d, led


def tables(d, led):
    days = L.build_days(d, led)
    pairs = days[days.valid].reset_index(drop=True)
    return days, pairs, L.features(days)


def folds(pairs):
    """every fold of both tests, built by the same lines as run.py: (test, h, block, held-out train, train rows, test rows)."""
    for h in L.HORIZONS:
        P = pairs[pairs.h == h]
        for b, (start, _) in enumerate(L.BLOCKS):
            train = P[P.t + h < start].sort_values(["t", "train"])
            test = P[P.block == b]
            yield "rolling", h, b, "", train, test
            for k_out in "ABC":                                     # an unseen train, in the future
                tr_k, te_k = train[train.train != k_out], test[test.train == k_out]
                if te_k.empty:
                    continue
                yield "unseen_train", h, b, k_out, tr_k, te_k


# ------------------------------------------------------------------ the networks' inputs
LOOKBACK = 17                                         # days t-16 .. t: the span XGBoost's lags reach (14 days + 2 tolerance)
SEQ_CH = L.IND + ["record", "in_cycle"]
STATIC = ["days_since_cip", "n_samples", "dp_pct_slope14", "flux_pct_slope14", "dp_pct_state", "flux_pct_state",
          "dp_pct_law", "flux_pct_law"]
FLAGS = ["no_cip_yet", "slope_missing", "law_missing"]


def sequences(d, led):
    """(train, t) -> (17, 6): the four indicators carried forward from the latest record at or before each day (before a
    train's first record: its first record), a flag 'record on this day', a flag 'day in the current cleaning cycle'."""
    out = {}
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        cips = np.sort(led[led.train == tr].day_index.values)
        days_, vals, first = g.index.to_numpy(), g[L.IND].to_numpy(float), int(g.index.min())
        for t in days_:
            last = cips[cips <= t].max() if (cips <= t).any() else -1
            s = np.zeros((LOOKBACK, len(SEQ_CH)), dtype=np.float32)
            for j, tau in enumerate(range(int(t) - LOOKBACK + 1, int(t) + 1)):
                pos = np.searchsorted(days_, tau, side="right") - 1
                s[j, :4] = vals[max(pos, 0)]
                s[j, 4] = float(pos >= 0 and days_[pos] == tau)
                s[j, 5] = float(tau > last and tau >= first)
            out[(tr, int(t))] = s
    return out


def static_inputs(df, first_day):
    """XGBoost's 8 inputs that are not indicator values, with its missing values filled and flagged (3 flags)."""
    X = df[STATIC].astype(float).copy()
    no_cip = X.days_since_cip.isna()
    X.loc[no_cip, "days_since_cip"] = (df.t - df.train.map(first_day))[no_cip]
    slope_m, law_m = X.dp_pct_slope14.isna(), X.dp_pct_law.isna()
    assert (slope_m == X.flux_pct_slope14.isna()).all() and (law_m == X.flux_pct_law.isna()).all()
    X[["dp_pct_slope14", "flux_pct_slope14"]] = X[["dp_pct_slope14", "flux_pct_slope14"]].fillna(0.0)
    X.loc[law_m, "dp_pct_law"] = df.dp_pct[law_m]
    X.loc[law_m, "flux_pct_law"] = df.flux_pct[law_m]
    assert not X.isna().any().any()
    F = np.column_stack([no_cip, slope_m, law_m]).astype(float)
    return np.column_stack([X.to_numpy(float), F]).astype(np.float32)


def net_inputs(df, seqs, first_day):
    S = np.stack([seqs[(tr, int(t))] for tr, t in zip(df.train, df.t)])
    return S, static_inputs(df, first_day)


# ------------------------------------------------------------------ the networks and their fitting
HIDDEN, DENSE, LR, WD, BATCH, MAX_EPOCHS, PATIENCE, CLIP, VAL_FRAC = 32, 16, 1e-3, 1e-4, 64, 150, 15, 1.0, 0.8


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def make_net(kind, n_seq=len(SEQ_CH), n_static=len(STATIC) + len(FLAGS)):
    torch = _torch()
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = (nn.LSTM(n_seq, HIDDEN, batch_first=True) if kind == "LSTM"
                        else nn.RNN(n_seq, HIDDEN, batch_first=True, nonlinearity="tanh"))
            self.head = nn.Sequential(nn.Linear(HIDDEN + n_static, DENSE), nn.ReLU(), nn.Linear(DENSE, 1))

        def forward(self, s, x):
            o, _ = self.rnn(s)
            return self.head(torch.cat([o[:, -1], x], 1)).squeeze(-1)

    return Net()


class Scaler:
    """mean and standard deviation of the rows the network is fitted on (training rows only)."""
    def __init__(self, S, X, dy):
        flat = S.reshape(-1, S.shape[2])
        self.ms, self.ss = flat.mean(0), flat.std(0)
        self.mx, self.sx = X.mean(0), X.std(0)
        self.ss[self.ss < 1e-12], self.sx[self.sx < 1e-12] = 1.0, 1.0
        self.my, self.sy = float(dy.mean()), float(dy.std()) or 1.0

    def tensors(self, S, X):
        torch = _torch()
        return (torch.tensor((S - self.ms) / self.ss, dtype=torch.float32),
                torch.tensor((X - self.mx) / self.sx, dtype=torch.float32))


def _epoch(net, opt, St, Xt, Yt, rng):
    torch = _torch()
    net.train()
    perm = rng.permutation(len(Yt))
    for b in range(0, len(perm), BATCH):
        j = torch.as_tensor(perm[b:b + BATCH])
        opt.zero_grad()
        loss = ((net(St[j], Xt[j]) - Yt[j]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()


def _change(net, sc, S, X):
    torch = _torch()
    net.eval()
    with torch.no_grad():
        St, Xt = sc.tensors(S, X)
        return net(St, Xt).numpy().astype(float) * sc.sy + sc.my


def fit_predict_seed(kind, seed, A_fit, A_val, A_all, A_test):
    """one seed: stage 1 (fit rows, early stopping on the validation slice) gives E*; stage 2 refits on all the fold's
    training rows for E* epochs and forecasts the test rows. A_* = (S, X, now, y)."""
    torch = _torch()
    S, X, now, y = A_fit
    sc = Scaler(S, X, y - now)
    St, Xt = sc.tensors(S, X)
    Yt = torch.tensor(((y - now) - sc.my) / sc.sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net = make_net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    Sv, Xv, now_v, y_v = A_val
    best_mae, best_ep, ep = np.inf, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        _epoch(net, opt, St, Xt, Yt, rng)
        mae = float(np.mean(np.abs(y_v - np.clip(now_v + _change(net, sc, Sv, Xv), 0, 1))))
        if mae < best_mae:
            best_mae, best_ep = mae, ep
        elif ep - best_ep >= PATIENCE:
            break
    stopped = ep
    # stage 2: same seed, same starting weights, all training rows, E* epochs
    S2, X2, now2, y2 = A_all
    sc2 = Scaler(S2, X2, y2 - now2)
    St2, Xt2 = sc2.tensors(S2, X2)
    Yt2 = torch.tensor(((y2 - now2) - sc2.my) / sc2.sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net2 = make_net(kind)
    opt2 = torch.optim.Adam(net2.parameters(), lr=LR, weight_decay=WD)
    rng2 = np.random.default_rng(seed)
    for _ in range(best_ep):
        _epoch(net2, opt2, St2, Xt2, Yt2, rng2)
    S_te, X_te = A_test[0], A_test[1]
    return _change(net2, sc2, S_te, X_te), {"best_epoch": best_ep, "stopped_epoch": stopped, "val_mae_pp": 100 * best_mae}


def validation_split(train, h):
    """the validation slice (training rows from the day of the row at 80 %) and the stage-1 fit rows (target day before it)."""
    v_start = train.t.iloc[int(round(VAL_FRAC * len(train)))]
    return train[train.t + h < v_start], train[train.t >= v_start], int(v_start)
