"""Part B rows: one per UF filtration segment, features at minute 5 and the climb to minute 11."""
from pathlib import Path
import time

import pandas as pd

from chemai.features.uf_forecast import build_rows

HERE = Path(__file__).parent
t0 = time.time()
seg = pd.read_parquet(HERE.parent / "p04_tide_fouling" / "results" / "segments.parquet")
cur = pd.read_parquet(HERE / "results" / "uf_curves.parquet")
rows = build_rows(seg, cur)
rows.to_parquet(HERE / "results" / "uf_rows.parquet")
print(rows.shape, "with target", int(rows["y"].notna().sum()), round(time.time() - t0), "s")
