"""Phase 3b, step 1 - the gate's real target, run once as declared: the real plant, phase 3's rolling origin, ΔP history only,
the table's own cleaning legs, one XGBoost seed.

    python real_level0.py <phase3 folder> <out folder>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P3 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); sys.path.insert(0, str(P3))
import p3lib as L  # noqa: E402

FEATS_DP = ["dp_pct", "dp_pct_lag1", "dp_pct_lag3", "dp_pct_lag7", "dp_pct_lag14", "dp_pct_slope14", "dp_pct_state",
            "dp_pct_law", "days_since_cip"]
L.fit_xgb.__defaults__ = ((0,),)                 # one seed, as in step 4 (the default is bound when p3lib is imported:
assert len(L.fit_xgb(np.zeros((10, 2)), np.arange(10.0))) == 1   # setting L.SEEDS afterwards does not reach it)

t0 = time.time()
d = pd.read_csv(P3 / "data/derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
legs = []
for t, g in d.groupby("train"):
    days = {int(round(v)) for v in pd.concat([(g.day_index - g.days_since_cip).dropna(), (g.day_index + g.days_to_next_cip).dropna()])}
    legs += [{"train": t, "day_index": x} for x in sorted(days)]
legs = pd.DataFrame(legs)
days = L.build_days(d, legs)
pairs = days[days.valid].reset_index(drop=True)
rows = []
c = "dp_pct"
for h in L.HORIZONS:
    P = pairs[pairs.h == h]
    for b, (start, _) in enumerate(L.BLOCKS):
        train, test = P[P.t + h < start].sort_values(["t", "train"]), P[P.block == b]      # phase 3's row order
        (f,) = L.forecasts(train, [test], c, h, FEATS_DP)
        rows.append(test[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"}).assign(h=h, block=b, **f))
ro = pd.concat(rows, ignore_index=True)
res = []
for h, g in ro.groupby("h"):
    pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in L.METHODS})
    blocks = [L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in L.METHODS}) for _, gb in g.groupby("block")]
    for m in L.METHODS:
        bs = [bl[m]["skill_pct"] for bl in blocks]
        res.append({"h": h, "method": m, "n": len(g), "MAE_pp": pooled[m]["MAE_pp"], "skill_pct": pooled[m]["skill_pct"],
                    **{f"skill block {i + 1}": s for i, s in enumerate(bs)},
                    "beats no change": L.verdict(pooled[m]["skill_pct"], bs) if m != "no change" else None})
res = pd.DataFrame(res)
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "real_level0.csv", index=False)
(OUT / "real_level0.json").write_text(json.dumps({"inputs": FEATS_DP, "pairs": int(len(pairs)), "seconds": round(time.time() - t0),
                                                  "verdicts": res[res.method != "no change"][["h", "method", "beats no change"]].values.tolist()},
                                                 indent=1, default=str))
pd.set_option("display.width", 200)
print(res.pivot(index="method", columns="h", values="skill_pct").loc[L.METHODS].round(1))
print("no change MAE (pp):", res[res.method == "no change"].set_index("h").MAE_pp.round(2).to_dict())
print("beats no change:", res[res["beats no change"] == True][["h", "method"]].values.tolist() or "none")
print(f"done in {time.time() - t0:.0f} s")
