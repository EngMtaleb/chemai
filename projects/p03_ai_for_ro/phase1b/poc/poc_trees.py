"""Phase 1B proof of concept - no change, the fouling law, XGBoost and the hybrid, on both records and both tests;
the negative control; which columns XGBoost used (declaration_poc.md).

    python3 -B poc_trees.py <out folder> [smoke | join | P1 P2 P3]

Per plant, phase 1's folds and p3lib.forecasts unchanged (XGBoost: phase 1's settings, five seeds). Every forecast saved.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import poc_lib as P

L = P.L
T0 = time.time()


def say(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]", flush=True)


def forecasts_for(days, feats, rec, p, keep, smoke=False):
    pairs = days[days.valid].reset_index(drop=True)
    out = []
    for c in L.TARGETS:
        for test, h, b, k, train, te in P.folds(pairs):
            if not keep(test, h):
                continue
            (f,) = L.forecasts(train, [te], c, h, feats)
            base = te[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"})
            out.append(base.assign(plant=p, record=rec, test=test, target=c, h=h, block=b, held_out=k,
                                   **{m: f[m] for m in P.METHODS}))
            if smoke:
                say(f"smoke {p} {rec} {c} {test} h={h} b={b}: train {len(train)}, test {len(te)}, {len(feats)} inputs")
                return out
    return out


def importance(days, feats, p):
    """XGBoost on record 2, rolling test, ΔP 7 days ahead: the test rows' MAE when one group's (or one column's) inputs are
    permuted together across the rows, five repeats."""
    pairs = days[days.valid].reset_index(drop=True)
    groups = {g: [i for i, f in enumerate(feats) if P.group_of(f) == g] for g in ("causes", "operation", "results", "no link")}
    cols = {c: [i for i, f in enumerate(feats) if f == c or f.startswith(c + "_")] for c in P.EXTRA}
    rows, rows_c = [], []
    for test, h, b, k, train, te in P.folds(pairs):
        if test != "rolling" or h != 7:
            continue
        models = L.fit_xgb(train[feats].to_numpy(float), (train["y_dp_pct"] - train["dp_pct"]).to_numpy(float))
        Xt, nowt, yt = te[feats].to_numpy(float), te["dp_pct"].to_numpy(float), te["y_dp_pct"].to_numpy(float)
        mae0 = float(np.mean(np.abs(yt - np.clip(nowt + L.predict_xgb(models, Xt), 0, 1))))
        for kind, sets, sink in (("group", groups, rows), ("column", cols, rows_c)):
            for name, idx in sets.items():
                for r in range(5):
                    perm = np.random.default_rng([P.PLANTS[p], 6, b, r]).permutation(len(Xt))
                    Xp = Xt.copy()
                    Xp[:, idx] = Xt[perm][:, idx]
                    mae = float(np.mean(np.abs(yt - np.clip(nowt + L.predict_xgb(models, Xp), 0, 1))))
                    sink.append({"plant": p, "block": b, kind: name, "repeat": r, "n": len(yt), "mae_pp": 100 * mae0,
                                 "mae_permuted_pp": 100 * mae})
    return rows, rows_c


def main():
    """resumable: each piece (plant x record, the control, importance) is written to its own file when done; a piece whose
    file exists is skipped, so a stopped run continues where it stopped. `join` writes the joined files."""
    out = Path(sys.argv[1])
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    parts = out / "trees_parts"
    parts.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "join":
        return join(out, parts)
    t, c = P.load()
    only = [a for a in sys.argv[2:] if a in P.PLANTS]          # optional: run only these plants (each run stays under 2 h)
    for p in P.PLANTS:
        if only and p not in only:
            continue
        names = [parts / f"{p}_record1.csv", parts / f"{p}_record2.csv", parts / f"{p}_control.csv",
                 parts / f"{p}_importance_groups.csv"]
        if all(n.exists() for n in names) and not smoke:
            continue
        d, led, _ = P.plant(t, c, p)
        days = P.build_days(d, led)
        fs = P.feature_sets(days)
        say(f"{p}: {int(days.valid.sum())} pairs; record 1 {len(fs['record 1'])} inputs, record 2 {len(fs['record 2'])}")
        for (rec, feats), name in zip(fs.items(), names[:2]):
            if name.exists() and not smoke:
                continue
            rows = forecasts_for(days, feats, rec, p, lambda test, h: True, smoke)
            if smoke:
                return
            pd.concat(rows, ignore_index=True).to_csv(name.with_suffix(".tmp"), index=False)
            name.with_suffix(".tmp").rename(name)
            say(f"{p} {rec} done")
        if not names[2].exists():                 # the negative control: rolling test, 7 and 14 days, both targets
            keep = (lambda test, h: test == "rolling" and h in (7, 14))
            minus = [f for f in fs["record 2"] if f not in P.extra_feats(P.INSTRUMENTS)]
            rows = forecasts_for(days, minus, "record 2 without SDI and turbidity", p, keep)
            ds, leds, k = P.plant(t, c, p, shift_instruments=True)
            days_s = P.build_days(ds, leds)
            assert (days_s.valid == days.valid).all() and (days_s.t == days.t).all()
            rows += forecasts_for(days_s, fs["record 2"], "record 2, SDI and turbidity shifted", p, keep)
            pd.concat(rows, ignore_index=True).assign(shift_days=k).to_csv(names[2].with_suffix(".tmp"), index=False)
            names[2].with_suffix(".tmp").rename(names[2])
            say(f"{p} control done (shift {k} days)")
        if not names[3].exists():
            g, cc = importance(days, fs["record 2"], p)
            pd.DataFrame(cc).to_csv(parts / f"{p}_importance_columns.csv", index=False)
            pd.DataFrame(g).to_csv(names[3].with_suffix(".tmp"), index=False)
            names[3].with_suffix(".tmp").rename(names[3])
            say(f"{p} importance done")
    say("all pieces done")


def join(out, parts):
    ps = list(P.PLANTS)
    pd.concat([pd.read_csv(parts / f"{p}_record{r}.csv", keep_default_na=False, na_values=[""]) for p in ps for r in (1, 2)],
              ignore_index=True).to_csv(out / "pred_trees.csv", index=False)
    pd.concat([pd.read_csv(parts / f"{p}_control.csv", keep_default_na=False, na_values=[""]) for p in ps],
              ignore_index=True).to_csv(out / "pred_control.csv", index=False)
    for kind in ("groups", "columns"):
        pd.concat([pd.read_csv(parts / f"{p}_importance_{kind}.csv") for p in ps],
                  ignore_index=True).to_csv(out / f"importance_{kind}.csv", index=False)
    say("joined")


if __name__ == "__main__":
    main()
