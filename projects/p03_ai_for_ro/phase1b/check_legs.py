"""Phase 3, a check found later: the ledger lists 18 campaigns, but the daily table's own cleaning days include
5 second legs, 1-2 days after a campaign day (B 526, 545; C 345, 525, 547 - all before day 550). Phase 3 cut pairs at
the ledger days only, so a few TRAINING pairs span a second leg; no test pair can (test days start at 550).
This re-runs the forecasts with the legs added to the cleaning days and compares the verdicts and skills.

    python check_legs.py <phase3 folder> <out csv>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P3 = Path(sys.argv[1]); sys.path.insert(0, str(P3))
import p3lib as L  # noqa: E402

t0 = time.time()
d = pd.read_csv(P3 / "data/derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
led = pd.read_csv(P3 / "data/cip_campaign_ledger.csv")
extra = []
for t, g in d.groupby("train"):
    legs = {int(round(v)) for v in pd.concat([(g.day_index - g.days_since_cip).dropna(), (g.day_index + g.days_to_next_cip).dropna()])}
    extra += [{"train": t, "episode": "second leg", "day_index": x, "stage": "leg"} for x in sorted(legs - set(led[led.train == t].day_index))]
print("second legs added:", [(e["train"], e["day_index"]) for e in extra])
led2 = pd.concat([led, pd.DataFrame(extra)], ignore_index=True)

def run(ledger):
    days = L.build_days(d, ledger); pairs = days[days.valid].reset_index(drop=True); feats = L.features(days)
    ro, lo = [], []
    for c in L.TARGETS:
        for h in L.HORIZONS:
            P = pairs[pairs.h == h]
            for b, (start, _) in enumerate(L.BLOCKS):
                train = P[P.t + h < start]; test = P[P.block == b]
                (f,) = L.forecasts(train, [test], c, h, feats)
                ro.append(test[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"}).assign(target=c, h=h, block=b, **f))
                for k in "ABC":
                    tr_k, te_k = train[train.train != k], test[test.train == k]
                    if len(te_k):
                        (fk,) = L.forecasts(tr_k, [te_k], c, h, feats)
                        lo.append(te_k[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"}).assign(target=c, h=h, block=b, **fk))
    return pd.concat(ro, ignore_index=True), pd.concat(lo, ignore_index=True), len(pairs)

def verdicts(p, test):
    rows = []
    for (c, h), g in p.groupby(["target", "h"]):
        pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in L.METHODS})
        blocks = [L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in L.METHODS}) for _, gb in g.groupby("block")]
        for m in L.METHODS[1:]:
            bs = [bl[m]["skill_pct"] for bl in blocks]
            rows.append({"test": test, "target": c, "h": h, "method": m, "skill_pct": pooled[m]["skill_pct"], "beats": L.verdict(pooled[m]["skill_pct"], bs)})
    return pd.DataFrame(rows)

out = []
for name, ledger in (("ledger only (as declared)", led), ("ledger + second legs", led2)):
    ro, lo, n = run(ledger)
    print(f"{name}: {n} pairs in all [{time.time() - t0:.0f} s]", flush=True)
    out += [verdicts(ro, "rolling origin").assign(cut=name), verdicts(lo, "unseen train").assign(cut=name)]
res = pd.concat(out)
res.to_csv(sys.argv[2], index=False)
a = res[res.cut.str.startswith("ledger only")].set_index(["test", "target", "h", "method"])
b = res[res.cut.str.startswith("ledger +")].set_index(["test", "target", "h", "method"])
print("verdicts that change:", int((a.beats != b.beats).sum()), "| largest change of skill, points:", round(float((a.skill_pct - b.skill_pct).abs().max()), 1))
print("passes with the legs:", b[b.beats].index.tolist())
print(f"done in {time.time() - t0:.0f} s")
