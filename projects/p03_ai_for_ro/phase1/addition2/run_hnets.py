"""Phase 1 addition 2 - hybrid LSTM (method 10) and hybrid RNN (method 11): every fold of both tests, as declared in
declaration_addendum2.md. Also, for the deck (descriptive): stage-1 curves and a 90 % range computed afterwards, and
the same for addition 1's LSTM and RNN on the rolling folds for dp_pct, 7 days ahead.

    python3 -B run_hnets.py [part] [n_parts]      (jobs split round-robin over parts; parts can run side by side)
    python3 -B run_hnets.py collect

The fitting is addition 1's fit_predict_seed, line for line, with two additions that change no number it returns: each
seed keeps a copy of its weights at the best epoch, and gives stage 1's forecasts on its fit rows and validation slice.
For the hybrids the base is the fouling law (p3lib.simple(...)["fouling law"]) instead of today.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "addition"))
import p3add as A  # noqa: E402  (addition 1's harness, unchanged; it imports phase 1's p3lib unchanged)
L = A.L
PARTS = HERE / "parts"
KINDS = {"hybrid LSTM": ("LSTM", "law"), "hybrid RNN": ("RNN", "law"), "LSTM": ("LSTM", "now"), "RNN": ("RNN", "now")}


def fit_seed(kind, seed, A_fit, A_val, A_all, A_test):
    """addition 1's fit_predict_seed with A_* = (S, X, base, y); returns stage 2's change on the test rows, the log, and
    stage 1's changes on the fit rows and the validation slice at the best epoch."""
    torch = A._torch()
    S, X, base, y = A_fit
    sc = A.Scaler(S, X, y - base)
    St, Xt = sc.tensors(S, X)
    Yt = torch.tensor(((y - base) - sc.my) / sc.sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net = A.make_net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=A.LR, weight_decay=A.WD)
    rng = np.random.default_rng(seed)
    Sv, Xv, base_v, y_v = A_val
    best_mae, best_ep, ep, best_state = np.inf, 0, 0, None
    for ep in range(1, A.MAX_EPOCHS + 1):
        A._epoch(net, opt, St, Xt, Yt, rng)
        mae = float(np.mean(np.abs(y_v - np.clip(base_v + A._change(net, sc, Sv, Xv), 0, 1))))
        if mae < best_mae:
            best_mae, best_ep = mae, ep
            best_state = copy.deepcopy(net.state_dict())
        elif ep - best_ep >= A.PATIENCE:
            break
    stopped = ep
    net.load_state_dict(best_state)
    cur_fit, cur_val = A._change(net, sc, S, X), A._change(net, sc, Sv, Xv)
    # stage 2: same seed, same starting weights, all training rows, E* epochs
    S2, X2, base2, y2 = A_all
    sc2 = A.Scaler(S2, X2, y2 - base2)
    St2, Xt2 = sc2.tensors(S2, X2)
    Yt2 = torch.tensor(((y2 - base2) - sc2.my) / sc2.sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net2 = A.make_net(kind)
    opt2 = torch.optim.Adam(net2.parameters(), lr=A.LR, weight_decay=A.WD)
    rng2 = np.random.default_rng(seed)
    for _ in range(best_ep):
        A._epoch(net2, opt2, St2, Xt2, Yt2, rng2)
    S_te, X_te = A_test[0], A_test[1]
    return (A._change(net2, sc2, S_te, X_te), {"best_epoch": best_ep, "stopped_epoch": stopped, "val_mae_pp": 100 * best_mae},
            cur_fit, cur_val)


def fold_id(name, h, b, k, tag):
    return f"{tag}_{name}_h{h:02d}_b{b}{'_' + k if k else ''}"


def jobs(pairs):
    """the four chart re-runs first (addition 1's networks, dp_pct, 7 days, rolling), then every declared fold."""
    out = []
    for name, h, b, k, train, test in A.folds(pairs):
        if name == "rolling" and h == 7:
            out.append(("chart", name, h, b, k, train, test, ["dp_pct"], ["LSTM", "RNN"]))
    for name, h, b, k, train, test in A.folds(pairs):
        out.append(("declared", name, h, b, k, train, test, list(L.TARGETS), ["hybrid LSTM", "hybrid RNN"]))
    return out


def run(part, n_parts, only=None):
    t0 = time.time()
    d, led = A.load()
    days, pairs, feats = A.tables(d, led)
    seqs = A.sequences(d, led)
    first_day = d.groupby("train").day_index.min().to_dict()
    PARTS.mkdir(exist_ok=True)
    J = jobs(pairs)
    mine = [j for i, j in enumerate(J) if i % n_parts == part] if only is None else [J[only]]
    print(f"part {part}/{n_parts}: {len(mine)} of {len(J)} jobs [{time.time() - t0:.0f} s]", flush=True)
    for tag, name, h, b, k, train, test, targets, methods in mine:
        fid = fold_id(name, h, b, k, tag)
        if (PARTS / f"{fid}.csv").exists():
            continue
        t1 = time.time()
        fit, val, v_start = A.validation_split(train, h)
        frames = {"fit": fit, "val": val, "all": train, "test": test}
        In = {nm: A.net_inputs(df, seqs, first_day) for nm, df in frames.items()}
        out = test[["train", "t"]].assign(test=name, h=h, block=b, held_out=k)
        seed_rows, logs, curves = [], [], []
        for c in targets:
            law = {nm: L.simple(df, c, h)["fouling law"].astype(float) for nm, df in frames.items()}
            now = {nm: df[c].to_numpy(float) for nm, df in frames.items()}
            yy = {nm: df[f"y_{c}"].to_numpy(float) for nm, df in frames.items()}
            out[f"now|{c}"], out[f"y|{c}"], out[f"law|{c}"] = now["test"], yy["test"], law["test"]
            for m in methods:
                kind, basis = KINDS[m]
                base = law if basis == "law" else now
                Ac = {nm: (*In[nm], base[nm], yy[nm]) for nm in frames}
                ch_te, ch_fit, ch_val = [], [], []
                for s in L.SEEDS:
                    ch, info, cf, cv = fit_seed(kind, s, Ac["fit"], Ac["val"], Ac["all"], Ac["test"])
                    ch_te.append(ch); ch_fit.append(cf); ch_val.append(cv)
                    seed_rows.append(test[["train", "t"]].assign(test=name, h=h, block=b, held_out=k, target=c, method=m,
                                                                 seed=s, change=ch))
                    logs.append({"job": fid, "test": name, "h": h, "block": b, "held_out": k, "target": c, "method": m,
                                 "seed": s, "n_train": len(train), "n_fit": len(fit), "n_val": len(val),
                                 "val_start_day": v_start, **info})
                out[f"{m}|{c}"] = np.clip(base["test"] + np.mean(ch_te, axis=0), 0, 1)
                if name == "rolling":
                    p_fit = np.clip(base["fit"] + np.mean(ch_fit, axis=0), 0, 1)
                    p_val = np.clip(base["val"] + np.mean(ch_val, axis=0), 0, 1)
                    q90 = float(np.quantile(np.abs(yy["val"] - p_val), 0.9))
                    for part_name, df, p in (("fit", fit, p_fit), ("val", val, p_val)):
                        curves.append(df[["train", "t"]].assign(h=h, block=b, target=c, method=m, part=part_name,
                                                                 y=yy[part_name], pred=p, q90=q90, val_start_day=v_start))
        out.to_csv(PARTS / f"{fid}.tmp", index=False)
        pd.concat(seed_rows).to_csv(PARTS / f"{fid}_seeds.csv", index=False)
        pd.DataFrame(logs).to_csv(PARTS / f"{fid}_log.csv", index=False)
        if curves:
            pd.concat(curves).to_csv(PARTS / f"{fid}_curves.csv", index=False)
        (PARTS / f"{fid}.tmp").rename(PARTS / f"{fid}.csv")
        print(f"{fid}: train {len(train)} (fit {len(fit)}, validation {len(val)} from day {v_start}), test {len(test)}; "
              f"{time.time() - t1:.0f} s [{time.time() - t0:.0f} s]", flush=True)
    print(f"part {part} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        run(0, 1, only=int(sys.argv[2]))
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
