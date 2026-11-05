"""Prophet's chart data (declaration_addendum2.md, "Also saved for the deck"): dp_pct, 7 days ahead, rolling test.

    <python with prophet> -B prophet_chart.py

With addition 1's forecast() unchanged: forecasts at the origins of block 1's training pairs (the first three quarters
and the last quarter, by day, as phase 1 splits them for its ranges), and at the last quarter of every block's
training pairs, for a 90 % range computed afterwards: the 90th percentile of the absolute errors there.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "addition"))
import run_prophet as RP  # noqa: E402  (addition 1's Prophet code, unchanged)
L = RP.L

t0 = time.time()
d = pd.read_csv(RP.DATA / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
led = pd.read_csv(RP.DATA / "cip_campaign_ledger.csv")
days = L.build_days(d, led)
pairs = days[days.valid].reset_index(drop=True)
c, h = "dp_pct", 7
P = pairs[pairs.h == h]
cache = {}
FAILED = []


def fc(tr, t):
    if (tr, t) not in cache:
        g = d[(d.train == tr) & (d.day_index <= t)]
        cips = np.sort(led[led.train == tr].day_index.values)
        try:
            cache[(tr, t)] = float(RP.forecast(g.day_index.to_numpy(), g[c].to_numpy(float), cips, int(t), [h], False)[0])
        except Exception as e:                       # addition 1's declared fallback: no change, counted
            cache[(tr, t)] = float(g[c].iloc[-1]); FAILED.append({"train": tr, "t": int(t), "error": repr(e)[:200]})
    return cache[(tr, t)]


rows, q90 = [], {}
for b, (start, _) in enumerate(L.BLOCKS):
    train = P[P.t + h < start].sort_values(["t", "train"])
    cal_start = train.t.iloc[int(round(0.75 * len(train)))]      # phase 1's corrected split (final notebook): a whole day
    calib, fit = train[train.t >= cal_start], train[train.t + h < cal_start]
    errs = []
    for r in calib.itertuples():
        p = fc(r.train, int(r.t)); errs.append(abs(getattr(r, f"y_{c}") - p))
        if b == 0:
            rows.append({"part": "val", "train": r.train, "t": int(r.t), "y": float(getattr(r, f"y_{c}")), "pred": p, "now": float(getattr(r, c))})
    q90[b] = float(np.quantile(errs, 0.9))
    if b == 0:
        for r in fit.itertuples():
            rows.append({"part": "fit", "train": r.train, "t": int(r.t), "y": float(getattr(r, f"y_{c}")), "pred": fc(r.train, int(r.t)), "now": float(getattr(r, c))})
    print(f"block {b}: train {len(train)}, last quarter {len(calib)}, q90 {q90[b]:.4f}; fits so far {len(cache)} [{time.time() - t0:.0f} s]", flush=True)
pd.DataFrame(rows).to_csv(HERE / "prophet_chart_curves.csv", index=False)
(HERE / "prophet_chart_q90.json").write_text(json.dumps({"q90_by_block": q90, "fits": len(cache), "failed_fits_no_change": len(FAILED), "failures": FAILED, "seconds": round(time.time() - t0, 1)}))
print("done", len(cache), "fits", round(time.time() - t0), "s")
