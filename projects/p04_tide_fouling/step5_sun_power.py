"""Step 5 - how big a daily pattern and a restart effect could this record see?

Run before the sun tests are declared; the data's real alignment with the time
of day and the real early-hours coefficient are never used.

    python projects/p04_tide_fouling/step5_sun_power.py
"""
import json
from pathlib import Path

import pandas as pd

from chemai.evaluation.restart_test import RestartConfig, power as restart_power, window
from chemai.evaluation.tide_test import DAY_HARMONICS, TideTestConfig, power_curve, prepare

OUT = Path(__file__).parent / "results"


def restart_segments(seg: pd.DataFrame, cfg: RestartConfig) -> pd.DataFrame:
    ep = pd.read_parquet(OUT / "episodes.parquet")
    s = seg.merge(ep, on="episode", how="left")
    s = s[s.idle_before_h >= cfg.min_gap_hours]
    return window(s, "episode_start", cfg)


def main():
    seg = pd.read_parquet(OUT / "segments.parquet")
    cfg, rcfg = TideTestConfig(), RestartConfig()
    prep = prepare(seg, "fouling_pct_h", cfg, tod=False, temperature=False)
    out = {
        "H2_daily_wave": {"n": int(len(prep.y)),
                          "detection_rate_at_p_le_0.01": power_curve(
                              prep, (0.05, 0.10, 0.20, 0.40), n_sim=200, cfg=cfg,
                              periods=DAY_HARMONICS, wave_period=24.0, shift="hours")},
    }
    print(json.dumps(out), flush=True)
    w = restart_segments(seg, rcfg)
    out["H3_restart_step"] = {"n": int(len(w)), "episodes": int(w.episode.nunique()),
                              "detection_rate_at_p_le_0.01": restart_power(w, cfg=rcfg)}
    print(json.dumps(out["H3_restart_step"]), flush=True)
    (OUT / "step5_sun_power.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
