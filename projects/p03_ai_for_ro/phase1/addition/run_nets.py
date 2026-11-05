"""Phase 3 addition - the LSTM (method 8) and the plain RNN (method 9), every fold of both tests.

    python3 -B run_nets.py [part] [n_parts]     (folds are split round-robin over parts; parts can run side by side)
    python3 -B run_nets.py collect              (joins the parts: pred_nets_rolling.csv, pred_nets_unseen_train.csv,
                                                 pred_nets_seeds.csv, nets_log.csv)

Per fold, per indicator, per network, per seed (0-4): stage 1 fits on the training rows whose target day is before the
validation slice and stops early on the slice; stage 2 refits on all the fold's training rows for E* epochs and forecasts
the test rows. Forecast = clip(today + mean of the five seeds' changes, 0, 1). Settings: declaration_addendum.md, p3add.py.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import p3add as A
L = A.L
PARTS = A.HERE / "nets_parts"
KINDS = ("LSTM", "RNN")


def fold_id(name, h, b, k):
    return f"{name}_h{h:02d}_b{b}{'_' + k if k else ''}"


def run(part, n_parts):
    t0 = time.time()
    d, led = A.load()
    days, pairs, feats = A.tables(d, led)
    seqs = A.sequences(d, led)
    first_day = d.groupby("train").day_index.min().to_dict()
    PARTS.mkdir(exist_ok=True)
    all_folds = list(A.folds(pairs))
    mine = [f for i, f in enumerate(all_folds) if i % n_parts == part]
    print(f"part {part}/{n_parts}: {len(mine)} of {len(all_folds)} folds [{time.time() - t0:.0f} s]", flush=True)
    for name, h, b, k, train, test in mine:
        fid = fold_id(name, h, b, k)
        if (PARTS / f"{fid}.csv").exists():
            continue
        t1 = time.time()
        fit, val, v_start = A.validation_split(train, h)
        In = {nm: A.net_inputs(df, seqs, first_day) for nm, df in (("fit", fit), ("val", val), ("all", train), ("test", test))}
        frames = {"fit": fit, "val": val, "all": train, "test": test}
        out = test[["train", "t"]].assign(test=name, h=h, block=b, held_out=k)
        seed_rows, logs = [], []
        for c in L.TARGETS:
            Ac = {nm: (*In[nm], frames[nm][c].to_numpy(float), frames[nm][f"y_{c}"].to_numpy(float)) for nm in frames}
            out[f"now|{c}"], out[f"y|{c}"] = Ac["test"][2], Ac["test"][3]
            for kind in KINDS:
                changes = []
                for s in L.SEEDS:
                    ch, info = A.fit_predict_seed(kind, s, Ac["fit"], Ac["val"], Ac["all"], Ac["test"])
                    changes.append(ch)
                    seed_rows.append(test[["train", "t"]].assign(test=name, h=h, block=b, held_out=k, target=c, method=kind,
                                                                 seed=s, change=ch))
                    logs.append({"fold": fid, "test": name, "h": h, "block": b, "held_out": k, "target": c, "method": kind,
                                 "seed": s, "n_train": len(train), "n_fit": len(fit), "n_val": len(val),
                                 "val_start_day": v_start, **info})
                out[f"{kind}|{c}"] = np.clip(Ac["test"][2] + np.mean(changes, axis=0), 0, 1)
        out.to_csv(PARTS / f"{fid}.tmp", index=False)
        pd.concat(seed_rows).to_csv(PARTS / f"{fid}_seeds.csv", index=False)
        pd.DataFrame(logs).to_csv(PARTS / f"{fid}_log.csv", index=False)
        (PARTS / f"{fid}.tmp").rename(PARTS / f"{fid}.csv")             # the fold counts as done only when all is written
        print(f"{fid}: train {len(train)} (fit {len(fit)}, validation {len(val)} from day {v_start}), test {len(test)}; "
              f"{time.time() - t1:.0f} s [{time.time() - t0:.0f} s]", flush=True)
    print(f"part {part} done in {time.time() - t0:.0f} s", flush=True)


def collect():
    d, led = A.load()
    days, pairs, feats = A.tables(d, led)
    ids = [fold_id(n, h, b, k) for n, h, b, k, _, _ in A.folds(pairs)]
    missing = [i for i in ids if not (PARTS / f"{i}.csv").exists()]
    assert not missing, f"folds not done: {missing}"
    wide = pd.concat([pd.read_csv(PARTS / f"{i}.csv", keep_default_na=False, na_values=[""]) for i in ids], ignore_index=True)
    wide["held_out"] = wide["held_out"].fillna("")
    long = []
    for c in L.TARGETS:
        part = wide[["test", "train", "t", "h", "block", "held_out"]].assign(target=c, now=wide[f"now|{c}"], y=wide[f"y|{c}"])
        for kind in KINDS:
            part[kind] = wide[f"{kind}|{c}"]
        long.append(part)
    long = pd.concat(long, ignore_index=True)
    for name in ("rolling", "unseen_train"):
        long[long.test == name].drop(columns="test").to_csv(A.HERE / f"pred_nets_{name}.csv", index=False)
    pd.concat([pd.read_csv(PARTS / f"{i}_seeds.csv") for i in ids]).to_csv(A.HERE / "pred_nets_seeds.csv", index=False)
    logs = pd.concat([pd.read_csv(PARTS / f"{i}_log.csv") for i in ids])
    logs.to_csv(A.HERE / "nets_log.csv", index=False)
    print(f"{len(ids)} folds joined; {len(long)} forecast rows; epochs E* by network:")
    print(logs.groupby("method")[["best_epoch", "stopped_epoch"]].describe().round(1).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
