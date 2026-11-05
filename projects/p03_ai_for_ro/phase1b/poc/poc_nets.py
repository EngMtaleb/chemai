"""Phase 1B proof of concept - the LSTM and the RNN on both records, every fold of both tests (declaration_poc.md).

    python3 -B poc_nets.py <part> <n_parts>     (folds round-robin over parts; parts run side by side; a fold whose file
                                                 exists is skipped, so a stopped part continues where it stopped)
    python3 -B poc_nets.py smoke                (one seed of one network on one fold: shapes and timing only)
    python3 -B poc_nets.py collect              (joins the folds: out/pred_nets.csv, out/pred_nets_seeds.csv, out/nets_log.csv)

Addition 1's networks and fitting are imported unchanged (phase3/addition/p3add.py): the 17-day sequence, the LSTM and the
plain RNN (32 units, a 16-unit dense layer), stage 1 with early stopping on the validation slice, stage 2 on all the fold's
training rows for E* epochs, seeds 0-4, forecast = clip(today + mean of the five seeds' changes, 0, 1). What changes is only
what the simulated record holds: no count of 10-min samples (not simulated), and in record 2 every logged column carried
forward next to the four indicators in the sequence - so the input sizes are this record's.
"""
import sys
import time

import numpy as np
import pandas as pd

import poc_lib as P

sys.path.insert(0, str(P.SCRATCH / "phase3" / "addition"))
import p3add as A  # noqa: E402

L = P.L
OUT = P.HERE / "out"
PARTS = OUT / "nets_parts"
KINDS = ("LSTM", "RNN")
A.STATIC = [s for s in A.STATIC if s != "n_samples"]      # A.static_inputs reads A.STATIC when called
_MAKE = A.make_net
DIMS = {}


def _make_net(kind, n_seq=None, n_static=None):
    """addition 1's network, sized for this record's inputs (A.fit_predict_seed looks make_net up when it runs)."""
    return _MAKE(kind, DIMS["seq"], DIMS["static"])


A.make_net = _make_net


def sequences(d, led, extra):
    """A.sequences line for line, with `extra` columns carried forward next to the four indicators:
    (train, t) -> (17, 4 + len(extra) + 2)."""
    ch = L.IND + list(extra)
    out = {}
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        cips = np.sort(led[led.train == tr].day_index.values)
        days_, vals, first = g.index.to_numpy(), g[ch].to_numpy(float), int(g.index.min())
        for t in days_:
            last = cips[cips <= t].max() if (cips <= t).any() else -1
            s = np.zeros((A.LOOKBACK, len(ch) + 2), dtype=np.float32)
            for j, tau in enumerate(range(int(t) - A.LOOKBACK + 1, int(t) + 1)):
                pos = np.searchsorted(days_, tau, side="right") - 1
                s[j, :len(ch)] = vals[max(pos, 0)]
                s[j, len(ch)] = float(pos >= 0 and days_[pos] == tau)
                s[j, len(ch) + 1] = float(tau > last and tau >= first)
            out[(tr, int(t))] = s
    return out


def plants():
    t, c = P.load()
    for p in P.PLANTS:
        d, led, _ = P.plant(t, c, p)
        days = P.build_days(d, led)
        pairs = days[days.valid].reset_index(drop=True)
        first = d.groupby("train").day_index.min().to_dict()
        seqs = {"record 1": sequences(d, led, []), "record 2": sequences(d, led, P.EXTRA)}
        yield p, pairs, first, seqs


def fold_id(p, rec, name, h, b, k):
    return f"{p}_r{rec[-1]}_{name}_h{h:02d}_b{b}{'_' + k if k else ''}"


def all_folds():
    for p, pairs, first, seqs in plants():
        for rec in ("record 1", "record 2"):
            for name, h, b, k, train, test in P.folds(pairs):
                yield fold_id(p, rec, name, h, b, k), p, rec, name, h, b, k, train, test, first, seqs[rec]


def run_fold(fid, p, rec, name, h, b, k, train, test, first, seqs, seeds=L.SEEDS, kinds=KINDS, targets=L.TARGETS):
    fit, val, v_start = A.validation_split(train, h)
    frames = {"fit": fit, "val": val, "all": train, "test": test}
    In = {nm: A.net_inputs(df, seqs, first) for nm, df in frames.items()}
    DIMS.update(seq=In["test"][0].shape[2], static=In["test"][1].shape[1])
    out = test[["train", "t"]].assign(plant=p, record=rec, test=name, h=h, block=b, held_out=k)
    seed_rows, logs = [], []
    for c in targets:
        Ac = {nm: (*In[nm], frames[nm][c].to_numpy(float), frames[nm][f"y_{c}"].to_numpy(float)) for nm in frames}
        out[f"now|{c}"], out[f"y|{c}"] = Ac["test"][2], Ac["test"][3]
        for kind in kinds:
            changes = []
            for s in seeds:
                ch, info = A.fit_predict_seed(kind, s, Ac["fit"], Ac["val"], Ac["all"], Ac["test"])
                changes.append(ch)
                seed_rows.append(test[["train", "t"]].assign(plant=p, record=rec, test=name, h=h, block=b, held_out=k,
                                                             target=c, method=kind, seed=s, change=ch))
                logs.append({"fold": fid, "plant": p, "record": rec, "test": name, "h": h, "block": b, "held_out": k,
                             "target": c, "method": kind, "seed": s, "n_train": len(train), "n_fit": len(fit),
                             "n_val": len(val), "val_start_day": v_start, "n_seq": DIMS["seq"],
                             "n_static": DIMS["static"], **info})
            out[f"{kind}|{c}"] = np.clip(Ac["test"][2] + np.mean(changes, axis=0), 0, 1)
    return out, seed_rows, logs


def run(part, n_parts):
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    folds = list(all_folds())
    mine = [f for i, f in enumerate(folds) if i % n_parts == part]
    todo = [f for f in mine if not (PARTS / f"{f[0]}.csv").exists()]
    print(f"part {part}/{n_parts}: {len(mine)} of {len(folds)} folds, {len(todo)} to do [{time.time() - t0:.0f} s]", flush=True)
    for f in todo:
        fid = f[0]
        t1 = time.time()
        out, seed_rows, logs = run_fold(*f)
        out.to_csv(PARTS / f"{fid}.tmp", index=False)
        pd.concat(seed_rows).to_csv(PARTS / f"{fid}_seeds.csv", index=False)
        pd.DataFrame(logs).to_csv(PARTS / f"{fid}_log.csv", index=False)
        (PARTS / f"{fid}.tmp").rename(PARTS / f"{fid}.csv")
        print(f"{fid}: train {len(f[7])}, test {len(f[8])}; {time.time() - t1:.0f} s [{time.time() - t0:.0f} s]", flush=True)
    print(f"part {part} done in {time.time() - t0:.0f} s", flush=True)


def smoke():
    t0 = time.time()
    for f in all_folds():
        if f[0].endswith("rolling_h07_b3") and f[2] == "record 2":
            out, _, logs = run_fold(*f, seeds=(0,), kinds=("LSTM",), targets=("dp_pct",))
            print(f[0], "inputs", DIMS, logs[0], f"{time.time() - t0:.0f} s")
            return


def collect():
    ids = [f"{fid}" for fid, *_ in all_folds()]
    missing = [i for i in ids if not (PARTS / f"{i}.csv").exists()]
    assert not missing, f"{len(missing)} folds not done: {missing[:5]}"
    wide = pd.concat([pd.read_csv(PARTS / f"{i}.csv", keep_default_na=False, na_values=[""]) for i in ids], ignore_index=True)
    long = []
    for c in L.TARGETS:
        part = wide[["plant", "record", "test", "train", "t", "h", "block", "held_out"]].assign(
            target=c, now=wide[f"now|{c}"], y=wide[f"y|{c}"])
        for kind in KINDS:
            part[kind] = wide[f"{kind}|{c}"]
        long.append(part)
    pd.concat(long, ignore_index=True).to_csv(OUT / "pred_nets.csv", index=False)
    pd.concat([pd.read_csv(PARTS / f"{i}_seeds.csv") for i in ids]).to_csv(OUT / "pred_nets_seeds.csv", index=False)
    logs = pd.concat([pd.read_csv(PARTS / f"{i}_log.csv") for i in ids])
    logs.to_csv(OUT / "nets_log.csv", index=False)
    print(f"{len(ids)} folds joined; epochs by network:")
    print(logs.groupby(["record", "method"])[["best_epoch", "stopped_epoch"]].median().to_string())


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    elif sys.argv[1] == "smoke":
        smoke()
    else:
        run(int(sys.argv[1]), int(sys.argv[2]))
