"""Step 6 - the sun tests, exactly as DECLARED_BEFORE_RUN_2.md fixes them.

    python projects/p04_tide_fouling/step6_sun_tests.py

Writes results/step6_sun_tests.json with every number and each declared verdict.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.evaluation.restart_test import RestartConfig, restart_test, window
from chemai.evaluation.tide_test import (
    DAY_HARMONICS, PLACEBO, TideTestConfig, confirm_out_of_sample, harmonic_block, prepare,
    rotation_stack, tide_design, tide_test,
)

OUT = Path(__file__).parent / "results"
SPLIT = pd.Timestamp("2014-01-01")
SOLAR = (9.0, 15.0)


def daily_profile(beta) -> np.ndarray:
    """The fitted daily component at each minute of the day (standard clock)."""
    tau = np.arange(0, 24, 1 / 60)
    x, _ = harmonic_block(tau, DAY_HARMONICS)
    return x @ np.asarray(beta)


def solar_contrast(beta) -> float:
    tau = np.arange(0, 24, 1 / 60)
    h = daily_profile(beta)
    sun = (tau >= SOLAR[0]) & (tau < SOLAR[1])
    return float(h[sun].mean() - h[~sun].mean())


def contrast_interval(prep, n_boot=999, seed=7):
    """Bootstrap over shift blocks of the final regression (base model not refitted)."""
    d = tide_design(prep, DAY_HARMONICS)
    k = len(d.blocks)
    width = d.width
    t_res = d.zcat.reshape(len(prep.y), k, width).sum(axis=1)        # identity rotation = real clock
    rng = np.random.default_rng(seed)
    idx_by_block = {b: np.flatnonzero(prep.block == b) for b in d.blocks}
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(d.blocks, len(d.blocks), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        beta = np.linalg.lstsq(t_res[rows], prep.y_res[rows], rcond=None)[0]
        out[i] = solar_contrast(beta)
    return out


def main():
    cfg, rcfg = TideTestConfig(), RestartConfig()
    seg = pd.read_parquet(OUT / "segments.parquet")
    res = {}

    # ---------------- H2: the sun's clock
    sun = dict(tod=False, temperature=False)
    prep = prepare(seg, "fouling_pct_h", cfg, **sun)
    h2 = tide_test(prep, DAY_HARMONICS, cfg, "fouling_pct_h", shift="hours").as_dict()
    boot = contrast_interval(prep)
    med = float(np.median(prep.y))
    c = solar_contrast(h2["beta"])
    h2["solar_contrast"] = c
    h2["solar_contrast_relative"] = c / med
    h2["solar_contrast_ci95"] = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
    h2["profile_by_hour"] = [float(v) for v in daily_profile(h2["beta"])[::60]]
    res["H2_fouling_daily"] = h2
    res["C3_temperature_daily"] = tide_test(prepare(seg, "temp_c", cfg, **sun), DAY_HARMONICS, cfg,
                                            "temp_c", shift="hours").as_dict()
    res["H2_placebo"] = tide_test(prep, PLACEBO, cfg, "fouling_pct_h", shift="hours").as_dict()
    early, late = seg[seg.t_std < SPLIT], seg[seg.t_std >= SPLIT]
    res["H2_confirm_out_of_time"] = confirm_out_of_sample(
        prepare(early, "fouling_pct_h", cfg, **sun), prepare(late, "fouling_pct_h", cfg, **sun),
        DAY_HARMONICS, cfg, shift="hours")
    with_t = tide_test(prepare(seg, "fouling_pct_h", cfg, tod=False, temperature=True), DAY_HARMONICS, cfg,
                       "fouling_pct_h", shift="hours").as_dict()
    with_t["solar_contrast_relative"] = solar_contrast(with_t["beta"]) / med
    res["H2_sens_with_temperature"] = with_t

    # ---------------- H3: the morning restart
    ep = pd.read_parquet(OUT / "episodes.parquet")
    s = seg.merge(ep, on="episode", how="left")
    restarts = s[s.idle_before_h >= rcfg.min_gap_hours]
    w = window(restarts, "episode_start", rcfg)
    res["H3_restart"] = restart_test(w, cfg=rcfg).as_dict()
    span = (s.groupby("episode").t_start.max() - s.groupby("episode").episode_start.first()).dt.total_seconds() / 3600
    long_eps = span[span >= rcfg.placebo_start_hours + rcfg.window_hours].index
    wp = window(s[s.episode.isin(long_eps)], "episode_start", rcfg, offset_h=rcfg.placebo_start_hours)
    res["H3_placebo_24h"] = restart_test(wp, cfg=rcfg).as_dict()
    res["H3_2012_13"] = restart_test(w[w.episode_start < SPLIT], cfg=rcfg).as_dict()
    res["H3_2014_15"] = restart_test(w[w.episode_start >= SPLIT], cfg=rcfg).as_dict()
    res["H3_idle_2_12h"] = restart_test(w[w.idle_before_h < 12], cfg=rcfg).as_dict()
    res["H3_idle_over_12h"] = restart_test(w[w.idle_before_h >= 12], cfg=rcfg).as_dict()

    # ---------------- declared verdicts
    h2, c3, pl, cf = res["H2_fouling_daily"], res["C3_temperature_daily"], res["H2_placebo"], res["H2_confirm_out_of_time"]
    h3, h3p, a, b = res["H3_restart"], res["H3_placebo_24h"], res["H3_2012_13"], res["H3_2014_15"]
    v = {
        "H2": h2["p_value"] <= cfg.alpha,
        "H2_confirmation": cf["p_value"] <= cfg.alpha_confirm and cf["oos_variance_share"] > 0,
        "H2_positive_control": c3["p_value"] <= cfg.alpha,
        "H2_placebo_clean": pl["p_value"] > cfg.alpha,
        "H2_relevant": abs(h2["solar_contrast_relative"]) >= 0.10,
        "H3": h3["p_value"] <= rcfg.alpha,
        "H3_placebo_clean": h3p["p_value"] > rcfg.alpha,
        "H3_replicates": (np.sign(a["effect"]) == np.sign(b["effect"]) and a["p_value"] <= rcfg.alpha_confirm
                          and b["p_value"] <= rcfg.alpha_confirm),
        "H3_relevant": abs(h3["relative_effect"]) >= rcfg.relevance,
    }
    res["verdicts"] = {k: bool(x) for k, x in v.items()}
    (OUT / "step6_sun_tests.json").write_text(json.dumps(res, indent=2, default=float))

    for k in ("H2_fouling_daily", "C3_temperature_daily", "H2_placebo", "H2_sens_with_temperature"):
        r = res[k]
        print(f"{k:26s} n={r['n']:6d} blocks={r['n_blocks']} pR2={r['partial_r2']:.5f} "
              f"(null q95 {r['null_partial_r2_q95']:.5f}) p={r['p_value']:.3f}")
    print(f"solar contrast {h2['solar_contrast']:.3f} %/h = {h2['solar_contrast_relative']:+.3f} of median "
          f"(CI {h2['solar_contrast_ci95'][0]:.3f}..{h2['solar_contrast_ci95'][1]:.3f})")
    print("profile by hour:", np.round(h2["profile_by_hour"], 2))
    print(f"confirm out of time: share={cf['oos_variance_share']:.5f} p={cf['p_value']:.3f}")
    for k in ("H3_restart", "H3_placebo_24h", "H3_2012_13", "H3_2014_15", "H3_idle_2_12h", "H3_idle_over_12h"):
        r = res[k]
        print(f"{k:18s} n={r['n_segments']:5d} eps={r['n_episodes']:3d} effect={r['effect']:+.3f} %/h "
              f"({r['relative_effect']:+.3f} of median) se={r['se']:.3f} p={r['p_value']:.4f}")
    print("verdicts", res["verdicts"])


if __name__ == "__main__":
    main()
