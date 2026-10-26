"""Part C rows: every constant-setpoint CCRO batch that runs 25 minutes or more - its features at
minute 20 and its live cost after minute 20 (points every 2 % of V20). Development: datasets 3-4;
test: dataset 6 (the batches Project 5 scored its rule and the square-root law on)."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import read_dataset
from chemai.features.ccro_batches import COLUMNS, batch_curve
from chemai.features.ccro_forecast import early_and_points

HERE = Path(__file__).parent
P5 = HERE.parent / "p05_when_to_stop" / "results"
t0 = time.time()
cfg = CCROConfig()
batches = pd.read_parquet(P5 / "ccro_batches.parquet")
feats, points = [], []
for name in ("dataset3", "dataset4", "dataset6"):
    df = read_dataset(name, columns=COLUMNS)
    b_ds = batches[(batches.dataset == name) & batches.constant_setpoints].set_index(["exp", "file"])
    for (exp, file), g in df.groupby(["exp", "file"], observed=True, sort=False):
        if (exp, file) not in b_ds.index:
            continue
        row = b_ds.loc[(exp, file)]
        if not np.isfinite(row["brine_l"]) or not np.isfinite(row["prior_drain_kwh"]):
            continue
        out = early_and_points(batch_curve(g, cfg), row, cfg)
        if out is None:
            continue
        f, p = out
        bid = f"{name}|{exp}|{file}"
        f.update(batch=bid, dataset=name, exp=exp, file=file)
        feats.append(f)
        if len(p):
            p["batch"] = bid
            points.append(p)
F = pd.DataFrame(feats)
P = pd.concat(points, ignore_index=True)
F.to_parquet(HERE / "results" / "ccro_feats.parquet")
P.to_parquet(HERE / "results" / "ccro_points.parquet")
print(F.groupby("dataset").size().to_dict(), "points", len(P), round(time.time() - t0), "s")
print(F.groupby("dataset").end_r.describe().round(2))
