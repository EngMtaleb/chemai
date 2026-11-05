"""declaration_v15.md: the LSTM and the RNN (addition 1's networks and fitting, unchanged: phase3/addition/p3add.py, through
phase 1B's poc_nets.py) on the merged dataset: record 2, dp_pct, 7 days, rolling origin, seeds 0-4."""
import sys, time
import numpy as np, pandas as pd
import poc_lib15 as Q
P, L = Q.P, Q.L
sys.argv = sys.argv[:1]
import poc_nets as N  # noqa: E402  (sets addition 1's network up for this record's inputs)
A = N.A
OUT = Q.HERE / "out"
T0 = time.time()


def main():
    t, c = Q.load()
    days, feats, pieces = Q.merged(t, c)
    seqs, first = {}, {}
    for p, (d, led, _) in pieces.items():
        for (tr, tt), s in N.sequences(d, led, P.EXTRA).items():
            seqs[(p + tr, tt)] = s
        for tr, f0 in d.groupby("train").day_index.min().items():
            first[p + tr] = int(f0)
    pairs = days[days.valid].reset_index(drop=True)
    rows, seed_rows, logs = [], [], []
    for b, train, test in Q.folds(pairs):
        fit, val, v_start = A.validation_split(train, Q.H)
        frames = {"fit": fit, "val": val, "all": train, "test": test}
        In = {nm: A.net_inputs(df, seqs, first) for nm, df in frames.items()}
        N.DIMS.update(seq=In["test"][0].shape[2], static=In["test"][1].shape[1])
        Ac = {nm: (*In[nm], frames[nm][Q.C].to_numpy(float), frames[nm][f"y_{Q.C}"].to_numpy(float)) for nm in frames}
        out = test[["train", "t"]].assign(plant=test.train.str[:2], block=b, now=Ac["test"][2], y=Ac["test"][3])
        for kind in N.KINDS:
            changes = []
            for s in L.SEEDS:
                ch, info = A.fit_predict_seed(kind, s, Ac["fit"], Ac["val"], Ac["all"], Ac["test"])
                changes.append(ch)
                seed_rows.append(test[["train", "t"]].assign(block=b, method=kind, seed=s, change=ch))
                logs.append({"block": b, "method": kind, "seed": s, "n_train": len(train), "n_fit": len(fit),
                             "n_val": len(val), "val_start_day": v_start, **N.DIMS, **info})
                print(f"block {b} {kind} seed {s} [{time.time() - T0:.0f} s]", flush=True)
            out[kind] = np.clip(Ac["test"][2] + np.mean(changes, axis=0), 0, 1)
        rows.append(out)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "pred_nets15.csv", index=False)
    pd.concat(seed_rows, ignore_index=True).to_csv(OUT / "pred_nets15_seeds.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "nets15_log.csv", index=False)
    print(f"nets done [{time.time() - T0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
