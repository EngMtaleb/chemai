"""Project 2 - Week 6: archive compression.

Run:  python projects/p02_control_loops/week6_compression.py

  1. How compressed the real archives already are.
  2. What a slower logging rate costs a known-stiction loop.
  3. What a deadband costs - and why it is worse.
  4. The limit, in samples per oscillation cycle.

Teaching version: notebook p02_week6_compression.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from chemai.config import DATA_DIR, ControlLoopConfig, DataQualityConfig
from chemai.data import (compress, downsample, load_isdb, load_sacac, sample_loop_spec,
                         samples_per_cycle, simulate_loop)
from chemai.features import (cycles_in_record, oscillation_regularity, prepare_for_shape,
                             shape_index, shape_signal)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week6")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week6_compression.json"

RATES = (1, 2, 5, 10, 20)
DEADBANDS = (0.25, 0.5, 1.0, 2.0)
N_LOOPS = 12


def diagnose(sp, pv, op, cfg: ControlLoopConfig, loop_type: str = "F") -> str:
    """The adopted baseline, exactly as the weekly report applies it."""
    e = detrend(np.asarray(sp, float) - np.asarray(pv, float))
    period, regularity = oscillation_regularity(e)
    if not np.isfinite(period) or cycles_in_record(len(e), period) < cfg.min_cycles:
        return "insufficient data"
    signal, _ = shape_signal(loop_type, pv, sp, op)
    shape = shape_index(prepare_for_shape(signal, period, cfg.shape_harmonics))
    if regularity <= cfg.regularity_threshold or not np.isfinite(shape):
        return "undetermined"
    return "stiction" if shape > cfg.shape_triangular else "tuning"


def how_compressed_is_the_archive() -> dict:
    """Repeated samples and logging rates across every real loop we hold."""
    rows = []
    def note(name, pv, op, ts):
        if np.isfinite(pv).all():
            rows.append({"name": name, "ts": ts, "n": len(pv),
                         "pv_repeat": float(np.mean(np.diff(pv) == 0))})
    for r in load_sacac(DATA_DIR / "sacac"):
        note(r.name, r.data.pv.to_numpy(), r.data.op.to_numpy(), r.ts)
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if not x.in_sacac:
            note(x.key, x.data.pv.to_numpy(), x.data.op.to_numpy(), x.ts)
    df = pd.DataFrame(rows)
    quantiles = df.pv_repeat.quantile([.5, .75, .9, .95]).round(2).to_dict()
    log.info("%d real loops - repeated PV samples by quantile: %s", len(df), quantiles)
    log.info("worst: %s", df.nlargest(3, "pv_repeat")[["name", "pv_repeat"]].to_dict("records"))
    log.info("logging intervals present, s: %s", sorted(df.ts.dropna().unique()))
    return {"n_loops": int(len(df)), "pv_repeat_quantiles": quantiles,
            "worst": df.nlargest(3, "pv_repeat")[["name", "pv_repeat"]].round(2).to_dict("records"),
            "logging_intervals_s": sorted(df.ts.dropna().unique().tolist())}


def main() -> dict:
    cfg, dq = ControlLoopConfig(), DataQualityConfig()
    archive = how_compressed_is_the_archive()

    loops = [simulate_loop(sample_loop_spec("F", "stiction", np.random.default_rng(200 + i), cfg),
                           6000, seed=i, cfg=cfg) for i in range(N_LOOPS)]
    period = float(np.median([oscillation_regularity(detrend((d.sp - d.pv).to_numpy()))[0]
                              for d in loops]))
    log.info("%d simulated sticky loops, median period %.0f samples", N_LOOPS, period)

    rate_rows = []
    for k in RATES:
        calls = [diagnose(*[downsample(d, k)[c].to_numpy() for c in ("sp", "pv", "op")], cfg)
                 for d in loops]
        counts = pd.Series(calls).value_counts().to_dict()
        rate_rows.append({"every": k, "samples_per_cycle": round(samples_per_cycle(period, k), 1),
                          "stiction_found": counts.get("stiction", 0),
                          "read_as_tuning": counts.get("tuning", 0),
                          "undetermined": counts.get("undetermined", 0)
                          + counts.get("insufficient data", 0)})
    rates = pd.DataFrame(rate_rows)
    log.info("logging rate\n%s", rates.to_string(index=False))

    band_rows = []
    for band in DEADBANDS:
        calls = []
        for d in loops:
            squeezed = compress(d, deadband_std=band)
            calls.append(diagnose(d.sp.to_numpy(), squeezed.pv.to_numpy(),
                                  squeezed.op.to_numpy(), cfg))
        counts = pd.Series(calls).value_counts().to_dict()
        band_rows.append({"deadband_x_std": band, "stiction_found": counts.get("stiction", 0),
                          "read_as_tuning": counts.get("tuning", 0),
                          "undetermined": counts.get("undetermined", 0)
                          + counts.get("insufficient data", 0)})
    bands = pd.DataFrame(band_rows)
    log.info("deadband\n%s", bands.to_string(index=False))

    safe = rates[rates.stiction_found >= N_LOOPS // 2]
    limit = float(safe.samples_per_cycle.min()) if len(safe) else float("nan")
    log.info("the diagnosis survives down to about %.0f samples per cycle; "
             "the adopted guard is %.0f", limit, dq.min_samples_per_cycle)
    log.info("WORST CASE: a deadband of 1 x std leaves %d of %d loops read as TUNING - "
             "a confident wrong answer, not an uncertain one",
             int(bands.loc[bands.deadband_x_std == 1.0, "read_as_tuning"].iloc[0]), N_LOOPS)

    results = {"archive": archive, "n_simulated": N_LOOPS, "median_period_samples": round(period, 1),
               "by_rate": rates.to_dict("records"), "by_deadband": bands.to_dict("records"),
               "survives_to_samples_per_cycle": limit,
               "adopted_guard": dq.min_samples_per_cycle,
               "recommendation": "log at least 10 samples per expected oscillation cycle; "
                                 "check the historian deadband before the logging rate"}
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
