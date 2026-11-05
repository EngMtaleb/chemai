"""Phase 1 addition 2 - join the parts, then score the hybrid networks exactly as declared, next to the eleven saved
methods (phase 1's six, addition 1's four, and no change), with p3lib.scores / p3lib.verdict unchanged.

    python3 -B score2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "addition"))
import p3add as A  # noqa: E402
import run_hnets as R  # noqa: E402
L = A.L
KEY = ["target", "h", "block", "train", "t"]
NEW2 = ["hybrid LSTM", "hybrid RNN"]
ALL = L.METHODS + A.NEW + NEW2


def collect():
    d, led = A.load()
    days, pairs, feats = A.tables(d, led)
    ids = [R.fold_id(n, h, b, k, "declared") for n, h, b, k, _, _ in A.folds(pairs)]
    missing = [i for i in ids if not (R.PARTS / f"{i}.csv").exists()]
    assert not missing, f"folds not done: {missing}"
    wide = pd.concat([pd.read_csv(R.PARTS / f"{i}.csv", keep_default_na=False, na_values=[""]) for i in ids], ignore_index=True)
    wide["held_out"] = wide["held_out"].fillna("")
    long = []
    for c in L.TARGETS:
        part = wide[["test", "train", "t", "h", "block", "held_out"]].assign(target=c, now=wide[f"now|{c}"], y=wide[f"y|{c}"],
                                                                             law=wide[f"law|{c}"])
        for m in NEW2:
            part[m] = wide[f"{m}|{c}"]
        long.append(part)
    long = pd.concat(long, ignore_index=True)
    for name in ("rolling", "unseen_train"):
        long[long.test == name].drop(columns="test").to_csv(HERE / f"pred_hnets_{name}.csv", index=False)
    logs = pd.concat([pd.read_csv(R.PARTS / f"{i}_log.csv") for i in ids])
    logs.to_csv(HERE / "hnets_log.csv", index=False)
    print(f"{len(ids)} declared folds joined; {len(long)} forecast rows")
    print(logs.groupby("method")[["best_epoch", "stopped_epoch"]].describe().round(1).to_string())


def joined(test_name):
    sv = pd.read_csv(A.OUT3 / f"pred_{test_name}.csv")
    pr = pd.read_csv(A.HERE / "pred_prophet.csv")
    nn = pd.read_csv(A.HERE / f"pred_nets_{test_name}.csv", keep_default_na=False, na_values=[""])
    hn = pd.read_csv(HERE / f"pred_hnets_{test_name}.csv", keep_default_na=False, na_values=[""])
    m = sv.merge(pr, on=["train", "t", "target", "h"], how="left", validate="1:1")
    m = m.merge(nn, on=KEY, how="left", suffixes=("", "_nets"), validate="1:1")
    m = m.merge(hn, on=KEY, how="left", suffixes=("", "_hn"), validate="1:1")
    assert len(m) == len(sv) and not m[A.NEW + NEW2].isna().any().any(), "a test pair lacks a forecast"
    for col in ("now", "y"):
        assert np.array_equal(m[col].to_numpy(), m[f"{col}_nets"].to_numpy()) and np.array_equal(m[col].to_numpy(), m[f"{col}_hn"].to_numpy())
    return m


def forecast_scores(p):
    rows = []
    for (c, h), g in p.groupby(["target", "h"]):
        pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in ALL})
        blocks = {b: L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in ALL}) for b, gb in g.groupby("block")}
        for m in ALL:
            bs = [blocks[b][m]["skill_pct"] for b in sorted(blocks)]
            src = "phase 1 (saved)" if m in L.METHODS else ("addition 1 (saved)" if m in A.NEW else "addition 2")
            rows.append({"target": c, "h": h, "method": m, "source": src, "n": len(g), **pooled[m],
                         **{f"skill block {b + 1}": s for b, s in enumerate(bs)}, "blocks positive": int(sum(s > 0 for s in bs)),
                         "beats no change": L.verdict(pooled[m]["skill_pct"], bs) if m != "no change" else None})
    return pd.DataFrame(rows)


def main():
    collect()
    out = {}
    for test_name in ("rolling", "unseen_train"):
        fs = forecast_scores(joined(test_name))
        old = pd.read_csv(A.HERE / f"scores_{test_name}_addition.csv").set_index(["target", "h", "method"]).sort_index()
        a = fs[~fs.method.isin(NEW2)].set_index(["target", "h", "method"]).sort_index()
        cols = ["MAE_pp", "skill_pct", "R2_change"] + [f"skill block {i + 1}" for i in range(4)]
        diff = float(np.nanmax(np.abs(a[cols].to_numpy(float) - old.loc[a.index, cols].to_numpy(float))))
        print(f"[{test_name}] the eleven saved methods re-scored: largest difference from addition 1's score file {diff:.1e}")
        assert diff < 1e-9
        fs.to_csv(HERE / f"scores_{test_name}_addition2.csv", index=False)
        new = fs[fs.method.isin(NEW2)]
        print(new.pivot_table(index=["target", "method"], columns="h", values="skill_pct").round(1).to_string())
        print("blocks positive:\n" + new.pivot_table(index=["target", "method"], columns="h", values="blocks positive").to_string())
        passes = new[new["beats no change"] == True][["target", "h", "method"]].values.tolist()
        print("passes among the hybrid networks:", passes or "none")
        out[test_name] = {"passes": passes}
    (HERE / "score2_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
