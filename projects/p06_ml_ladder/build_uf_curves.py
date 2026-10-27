"""Every UF filtration segment's binned ln R curve (the rows of Project 4's segments.parquet,
same cutting), for Part B. Output: results/uf_curves.parquet (long form)."""
from pathlib import Path
import time

import pandas as pd

from chemai.config import UFPilotConfig, UFSegmentConfig
from chemai.data.uf_pilot import read_run, run_root
from chemai.features.uf_segments import segment_curves

HERE = Path(__file__).parent
P4 = HERE.parent / "p04_tide_fouling" / "results"

t0 = time.time()
cfg, scfg = UFPilotConfig(), UFSegmentConfig()
runs = pd.read_parquet(P4 / "runs.parquet")
root = run_root(cfg=cfg)
parts = []
for r in runs.itertuples():
    c = segment_curves(read_run(root / r.split / r.file), r.file, scfg)
    if not c.empty:
        parts.append(c)
curves = pd.concat(parts, ignore_index=True)
curves.to_parquet(HERE / "results" / "uf_curves.parquet")
seg = pd.read_parquet(P4 / "segments.parquet")
n = curves.groupby(["run", "t_start"]).ngroups
print("curves", n, "segments", len(seg), "rows", len(curves), round(time.time() - t0), "s")
