"""Step 2 - cut every included run into filtration segments and measure fouling.

Output: results/segments.parquet, one row per segment. Quality figures only
(counts, spreads, how much each run contributes); nothing tidal is computed.

    python projects/p04_tide_fouling/step2_segments.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import UFPilotConfig, UFSegmentConfig
from chemai.data.uf_pilot import read_run, run_root, to_standard_time
from chemai.features.uf_segments import segment_table

OUT = Path(__file__).parent / "results"


def main():
    cfg, scfg = UFPilotConfig(), UFSegmentConfig()
    runs = pd.read_parquet(OUT / "runs.parquet")
    root = run_root(cfg=cfg)
    parts = []
    for r in runs.itertuples():
        df = read_run(root / r.split / r.file)
        seg = segment_table(df, r.file, scfg)
        if seg.empty:
            continue
        seg["split"], seg["episode"] = r.split, r.episode
        seg["turbidity_reads"] = r.turbidity_reads
        seg["t_std"] = to_standard_time(pd.DatetimeIndex(seg.t_mid), cfg)
        parts.append(seg)
        print(f"{r.file:14s} {len(seg):5d} segments  median {seg.fouling_pct_h.median():6.2f} %/h", flush=True)
    s = pd.concat(parts, ignore_index=True)
    s.to_parquet(OUT / "segments.parquet")

    per_run = s.groupby("run").size()
    q = {
        "segments": int(len(s)),
        "runs_with_segments": int(s.run.nunique()),
        "runs_without_segments": sorted(set(runs.file) - set(s.run)),
        "segments_per_run": {"min": int(per_run.min()), "median": float(per_run.median()),
                             "max": int(per_run.max())},
        "minutes": s.minutes.describe().round(1).to_dict(),
        "fouling_pct_h": s.fouling_pct_h.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict(),
        "share_negative": round(float((s.fouling_pct_h < 0).mean()), 3),
        "median_se_over_abs_slope": round(float(np.median(s.fouling_se_pct_h / s.fouling_pct_h.abs())), 3),
        "by_split": s.split.value_counts().to_dict(),
        "by_n_online": s.n_online.value_counts().to_dict(),
        "feed_turbidity_segments": int((s.turbidity_reads == "feed").sum()),
        "episodes": int(s.episode.nunique()),
    }
    (OUT / "step2_segments.json").write_text(json.dumps(q, indent=2, default=str))
    print(json.dumps(q, indent=2, default=str))


if __name__ == "__main__":
    main()
