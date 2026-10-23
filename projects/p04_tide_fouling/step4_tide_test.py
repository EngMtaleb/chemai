"""Step 4 - the declared tide test, exactly as DECLARED_BEFORE_RUN.md fixes it.

    python projects/p04_tide_fouling/step4_tide_test.py

Writes results/step4_tide_test.json with every number and the verdict each
declared rule gives. Nothing here is tuned: thresholds and choices come from
the declaration and from chemai/evaluation/tide_test.py's TideTestConfig.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import UFPilotConfig
from chemai.data.uf_pilot import to_standard_time
from chemai.evaluation.tide_test import (
    CONSTITUENTS, PLACEBO, TideTestConfig, confirm_out_of_sample, prepare, tide_test,
)

OUT = Path(__file__).parent / "results"
SPLIT_DATE = pd.Timestamp("2014-01-01")


def run(seg, col, cfg, periods=None, **kw):
    prep = prepare(seg, col, cfg, **kw)
    return tide_test(prep, periods or CONSTITUENTS, cfg, response=col).as_dict()


def main():
    cfg = TideTestConfig()
    seg = pd.read_parquet(OUT / "segments.parquet")
    feed = seg[seg.turbidity_reads == "feed"]
    res = {"config": cfg.__dict__}

    # ---- declared tests
    res["H1_fouling"] = run(seg, "fouling_pct_h", cfg)
    res["C1_pump_speed"] = run(seg, "pump_speed", cfg)
    res["C2_mf_inlet_pressure"] = run(seg, "p_mf_in", cfg)
    res["S1_feed_turbidity"] = run(feed, "turbidity", cfg)
    res["placebo_fouling"] = run(seg, "fouling_pct_h", cfg, periods=PLACEBO)

    early = seg[seg.t_std < SPLIT_DATE]
    late = seg[seg.t_std >= SPLIT_DATE]
    res["confirm_out_of_time"] = confirm_out_of_sample(
        prepare(early, "fouling_pct_h", cfg), prepare(late, "fouling_pct_h", cfg), CONSTITUENTS, cfg)

    # ---- reported, not decisive
    res["official_split"] = confirm_out_of_sample(
        prepare(seg[seg.split == "training"], "fouling_pct_h", cfg),
        prepare(seg[seg.split == "test"], "fouling_pct_h", cfg), CONSTITUENTS, cfg)
    res["early_only"] = run(early, "fouling_pct_h", cfg)
    res["late_only"] = run(late, "fouling_pct_h", cfg)
    no_dst = seg.copy()
    no_dst["t_std"] = to_standard_time(pd.DatetimeIndex(no_dst.t_mid), UFPilotConfig(remove_daylight_saving=False))
    res["sens_no_daylight_saving"] = run(no_dst, "fouling_pct_h", cfg)
    res["sens_no_covariates"] = run(seg, "fouling_pct_h", cfg, covariates=False)
    res["sens_no_winsorising"] = run(seg, "fouling_pct_h", replace(cfg, winsor=(0.0, 1.0)))

    # ---- the declared verdicts
    h1, conf, plac = res["H1_fouling"], res["confirm_out_of_time"], res["placebo_fouling"]
    v = {
        "H1": h1["p_value"] <= cfg.alpha,
        "confirmation": conf["p_value"] <= cfg.alpha_confirm and conf["oos_variance_share"] > 0,
        "calibration": plac["p_value"] > cfg.alpha,
        "relevance": h1["relative_range"] >= cfg.relevance and h1["range_5_95"] > h1["range_null_q95"],
        "C1": res["C1_pump_speed"]["p_value"] <= cfg.alpha,
        "C2": res["C2_mf_inlet_pressure"]["p_value"] <= cfg.alpha,
        "S1": res["S1_feed_turbidity"]["p_value"] <= cfg.alpha,
    }
    if not v["calibration"]:
        outcome = 4
    elif v["H1"] and v["confirmation"]:
        outcome = 1
    elif v["H1"]:
        outcome = 2
    else:
        outcome = 3
    res["verdicts"] = v
    res["declared_outcome"] = outcome
    (OUT / "step4_tide_test.json").write_text(json.dumps(res, indent=2, default=float))

    def line(name, r):
        return (f"{name:26s} n={r['n']:6d} blocks={r['n_blocks']:3d} partialR2={r['partial_r2']:.5f} "
                f"(null q95 {r['null_partial_r2_q95']:.5f}) p={r['p_value']:.3f} "
                f"range={r['range_5_95']:.3f} vs null q95 {r['range_null_q95']:.3f} "
                f"rel={r['relative_range']:.3f}  M2 amp={r['amplitudes'].get('M2', float('nan')):.3f}")
    for k in ("H1_fouling", "C1_pump_speed", "C2_mf_inlet_pressure", "S1_feed_turbidity", "placebo_fouling",
              "early_only", "late_only", "sens_no_daylight_saving", "sens_no_covariates", "sens_no_winsorising"):
        print(line(k, res[k]))
    for k in ("confirm_out_of_time", "official_split"):
        r = res[k]
        print(f"{k:26s} oos share={r['oos_variance_share']:.5f} p={r['p_value']:.3f} null q95={r['null_q95']:.5f}")
    print("verdicts", v, "outcome", outcome)


if __name__ == "__main__":
    main()
