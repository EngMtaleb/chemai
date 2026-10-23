"""Step 1 - audit every file before any modelling.

Which files carry a clock, whether the clock can be trusted, how long each run
is, which runs form one continuous episode, and what the turbidity sensor was
actually reading. Nothing tidal is computed here.

    python projects/p04_tide_fouling/step1_audit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import UFPilotConfig
from chemai.data.uf_pilot import assign_episodes, audit_files, read_run

OUT = Path(__file__).parent / "results"


def turbidity_position(path: str, feed_min: float) -> tuple[str, float]:
    """'none' (no signal), 'feed' (median >= feed_min) or 'filtrate'."""
    tu = read_run(path)["turbidity"].to_numpy(float)
    tu = tu[np.isfinite(tu)]
    if len(tu) == 0 or np.mean(tu < -1) > 0.5:
        return "none", float("nan")
    med = float(np.median(tu))
    return ("feed" if med >= feed_min else "filtrate"), med


def daily_temperature_peak_hour(path: str) -> float:
    """Clock check: hour of the mean daily temperature maximum (runs >= 48 h)."""
    d = read_run(path)["temp_c"].resample("1h").mean().dropna()
    dev = d - d.rolling(24, center=True, min_periods=12).mean()
    prof = dev.groupby(dev.index.hour).mean()
    return float(prof.idxmax())


def main():
    from chemai.config import UFSegmentConfig
    cfg, scfg = UFPilotConfig(), UFSegmentConfig()
    a = audit_files(cfg=cfg)
    inc = a[a.included].copy()
    inc["episode"] = assign_episodes(inc, cfg)
    pos = [turbidity_position(p, scfg.feed_turbidity_min_ntu) for p in inc.path]
    inc["turbidity_reads"] = [p[0] for p in pos]
    inc["turbidity_median"] = [p[1] for p in pos]
    peaks = {r.file: daily_temperature_peak_hour(r.path) for r in inc.itertuples() if r.hours >= 48}

    # does the published test folder continue a training run without a break?
    eps = inc.groupby("episode").split.agg(lambda s: sorted(set(s)))
    mixed = [int(e) for e, s in eps.items() if len(s) > 1]

    summary = {
        "files": int(len(a)),
        "by_layout": a.kind.value_counts().to_dict(),
        "with_clock": int(a.has_clock.sum()),
        "malformed_lines": a[a.bad_lines > 0][["split", "file", "bad_lines"]].to_dict("records"),
        "clock_not_trusted": a[a.has_clock & ~a.clock_trusted][["split", "file", "clock_offset_days"]]
            .round(2).to_dict("records"),
        "trusted_and_long_enough": int(len(inc)),
        "by_split": inc.split.value_counts().to_dict(),
        "hours_total": round(float(inc.hours.sum()), 1),
        "hours_by_split": inc.groupby("split").hours.sum().round(1).to_dict(),
        "episodes": int(inc.episode.nunique()),
        "episodes_mixing_training_and_test": mixed,
        "first_and_last": [str(inc.t_start.min()), str(inc.t_end.max())],
        "turbidity_reads": inc.turbidity_reads.value_counts().to_dict(),
        "feed_turbidity_hours": round(float(inc.loc[inc.turbidity_reads == "feed", "hours"].sum()), 1),
        "temperature_peak_hour": {"runs": len(peaks),
                                  "histogram": pd.Series(peaks).value_counts().sort_index().to_dict()},
    }
    OUT.mkdir(exist_ok=True)
    keep = ["split", "file", "kind", "t_start", "t_end", "hours", "dt_median_s", "clock_offset_days",
            "episode", "turbidity_reads", "turbidity_median"]
    inc[keep].to_parquet(OUT / "runs.parquet")
    (OUT / "step1_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
