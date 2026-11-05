"""Phase 3 addition, step 2 - before any new method runs: does this harness give phase 3's saved numbers?

    python3 -B reproduce.py          (writes reproduce_check.txt next to this file)

(a) the test pairs and the no-change MAE of every indicator and horizon, both tests;
(b) XGBoost and hybrid re-run with phase 3's own code for dP 7 days ahead (rolling origin, all blocks; unseen train, all
    blocks and held-out trains) and flux 14 days ahead (rolling origin, all blocks), against the saved forecasts and skills.
"""
import time

import numpy as np
import pandas as pd

import p3add as A
L = A.L

t0 = time.time()
lines = []
say = lambda s: (print(s, flush=True), lines.append(s))
d, led = A.load()
days, pairs, feats = A.tables(d, led)
say(f"pairs per horizon: {pairs.groupby('h').size().to_dict()} | {len(feats)} XGBoost inputs")
saved = {"rolling": pd.read_csv(A.OUT3 / "pred_rolling.csv"), "unseen_train": pd.read_csv(A.OUT3 / "pred_unseen_train.csv")}
saved_scores = {"rolling": pd.read_csv(A.OUT3 / "scores_rolling.csv"), "unseen_train": pd.read_csv(A.OUT3 / "scores_unseen_train.csv")}
ok = True

# (a) the same test pairs, the same truth and today's value, the same no-change MAE
for test_name, sv in saved.items():
    rows = []
    for name, h, b, k, train, test in A.folds(pairs):
        if name != test_name:
            continue
        for c in L.TARGETS:
            rows.append(test[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"}).assign(target=c, h=h, block=b))
    mine = pd.concat(rows, ignore_index=True)
    key = ["target", "h", "block", "train", "t"]
    m = mine.merge(sv[key + ["now", "y"]], on=key, how="outer", suffixes=("", "_saved"), indicator=True)
    same_pairs = bool((m._merge == "both").all()) and len(mine) == len(sv)
    dmax = float(np.nanmax(np.abs(np.r_[m.now - m.now_saved, m.y - m.y_saved])))
    say(f"[{test_name}] test pairs: harness {len(mine)}, saved {len(sv)}, identical sets: {same_pairs}; "
        f"largest difference in today's value or truth: {dmax:.1e}")
    ok &= same_pairs and dmax == 0.0
    for (c, h), g in mine.groupby(["target", "h"]):
        mae = 100 * np.mean(np.abs(g.y - g.now))
        ref = saved_scores[test_name].query("target == @c and h == @h and method == 'no change'").MAE_pp.item()
        good = abs(mae - ref) < 1e-9
        ok &= good
        say(f"  no change MAE {c} h={h}: harness {mae:.6f} pp, saved {ref:.6f} pp, agree to 1e-9: {good}")

# (b) XGBoost and hybrid re-run with phase 3's own functions
cells = [("rolling", "dp_pct", 7), ("unseen_train", "dp_pct", 7), ("rolling", "flux_pct", 14)]
for test_name, c, h_want in cells:
    parts = []
    for name, h, b, k, train, test in A.folds(pairs):
        if name != test_name or h != h_want:
            continue
        (f,) = L.forecasts(train, [test], c, h, feats)
        print(f"    re-ran {test_name} {c} h={h} block {b}{' held-out ' + k if k else ''} [{time.time() - t0:.0f} s]", flush=True)
        parts.append(test[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"})
                     .assign(block=b, XGBoost=f["XGBoost"], hybrid=f["hybrid"]))
    mine = pd.concat(parts, ignore_index=True)
    sv = saved[test_name].query("target == @c and h == @h_want")
    m = mine.merge(sv, on=["train", "t", "block"], suffixes=("", "_saved"))
    assert len(m) == len(mine) == len(sv)
    dx = float(np.max(np.abs(m.XGBoost - m.XGBoost_saved)))
    dh = float(np.max(np.abs(m.hybrid - m.hybrid_saved)))
    preds = {"no change": m.now.to_numpy(), "XGBoost": m.XGBoost.to_numpy(), "hybrid": m.hybrid.to_numpy()}
    pooled = L.scores(m.y.to_numpy(), m.now.to_numpy(), preds)
    blocks = {b: L.scores(g.y.to_numpy(), g.now.to_numpy(), {k2: g[k2].to_numpy() for k2 in preds}) for b, g in m.groupby("block")}
    say(f"[{test_name}] {c} h={h_want}: {len(m)} forecasts; largest difference from the saved forecasts: "
        f"XGBoost {dx:.1e}, hybrid {dh:.1e}")
    ok &= dx < 1e-6 and dh < 1e-6
    for meth in ("XGBoost", "hybrid"):
        ref = saved_scores[test_name].query("target == @c and h == @h_want and method == @meth").iloc[0]
        bs = [blocks[b][meth]["skill_pct"] for b in sorted(blocks)]
        refb = [ref[f"skill block {i + 1}"] for i in range(4)]
        good = abs(pooled[meth]["skill_pct"] - ref.skill_pct) < 0.01 and max(abs(x - y) for x, y in zip(bs, refb)) < 0.01
        ok &= good
        say(f"  {meth}: skill harness {pooled[meth]['skill_pct']:+.3f} %, saved {ref.skill_pct:+.3f} %; blocks harness "
            f"{[round(x, 2) for x in bs]}, saved {[round(x, 2) for x in refb]}; verdict harness "
            f"{L.verdict(pooled[meth]['skill_pct'], bs)}, saved {ref['beats no change']}; agree to 0.01: {good}")

say(f"REPRODUCTION {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f} s")
(A.HERE / "reproduce_check.txt").write_text("\n".join(lines) + "\n")
