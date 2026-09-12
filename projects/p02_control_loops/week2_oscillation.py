"""Project 2 - Week 2, part 2: is the loop oscillating at all?

Run:  python projects/p02_control_loops/week2_oscillation.py

Produces the evidence for the decision recorded in VALIDATION.md section 15:
the regularity index is a CONFIDENCE SCORE, not a gate before diagnosis.
Teaching version: notebook p02_week2_oscillation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import load_isdb, load_sacac
from chemai.features import assess, cycles_in_record, oscillation_regularity

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week2osc")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week2_osc.json"

DB_RUNS = [f"stiction-P-oilgas-DB-{i}-baccidicapaci-2018" for i in range(1, 8)]


def spectral_period(x: np.ndarray) -> float:
    """Dominant period in samples, from the power spectrum - independent of the
    autocorrelation, so it can say WHY a loop scores zero regularity."""
    f, p = welch(x, nperseg=min(512, max(16, len(x) // 4)))
    return float(1 / f[1:][p[1:].argmax()])


def labelled_loops():
    """(source, name, label, sp, pv, op) for every real loop with a published label."""
    rows = []
    for r in load_sacac(DATA_DIR / "sacac"):
        if r.label is None:
            continue
        d = r.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        rows.append(("SACAC", r.name, r.label, sp, d.pv.to_numpy(), d.op.to_numpy()))
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.label is None or x.in_sacac:
            continue
        d = x.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        label = x.label if x.tier == "stated" else f"{x.label} ({x.tier})"
        rows.append(("ISDB", x.key, label, sp, d.pv.to_numpy(), d.op.to_numpy()))
    return rows


def main() -> dict:
    cfg = ControlLoopConfig()
    rows = []
    for src, name, label, sp, pv, op in labelled_loops():
        if not np.isfinite(pv).all():
            continue
        period, r = oscillation_regularity(detrend(pv - sp))
        spec = spectral_period(detrend(pv - sp))
        rows.append(dict(source=src, name=name, label=label, n=len(pv),
                         period=period, regularity=r, spectral_period=spec,
                         cycles=cycles_in_record(len(pv), spec),
                         quality=assess(sp, pv, op).flags))
    df = pd.DataFrame(rows)
    log.info("%d labelled real loops", len(df))

    by_label = df.groupby(df.label.str.replace(" (likely)", "", regex=False)).regularity.agg(
        n="size", above_1=lambda x: round(float((x > 1).mean()), 2), median=lambda x: round(x.median(), 2))
    log.info("regularity by label\n%s", by_label.to_string())

    # 1. the classes straddle the published threshold
    missed = df[(df.label.str.startswith("stiction")) & (df.regularity <= 1)]
    false_pos = df[(df.label.str.startswith(("no_stiction", "no_oscillation"))) & (df.regularity > 1)]
    log.info("labelled stiction below the threshold: %d of %d",
             len(missed), int(df.label.str.startswith("stiction").sum()))
    log.info("\n%s", missed[["name", "n", "regularity", "spectral_period", "cycles"]].to_string(index=False))
    log.info("labelled 'no stiction' above the threshold: %s", false_pos.name.tolist())

    # 2. a zero means one of two different things
    zeros = df[df.regularity == 0]
    short = zeros[zeros.cycles < cfg.min_cycles]
    quiet = zeros[zeros.cycles >= cfg.min_cycles]
    log.info("zero regularity, record too short (< %.0f cycles): %s", cfg.min_cycles,
             {r["name"]: round(r["cycles"], 1) for _, r in short.iterrows()})
    log.info("zero regularity with a long enough record - genuinely not oscillating: %s",
             sorted(set(quiet.label)))

    # 3. seven runs of ONE valve - no gap across the threshold
    runs = df[df.name.isin(DB_RUNS)].sort_values("regularity")
    log.info("seven runs of one sticky valve: %s",
             [(r["name"][-24:-18], round(r["regularity"], 2)) for _, r in runs.iterrows()])

    results = {
        "n_labelled": int(len(df)),
        "regularity_threshold": cfg.regularity_threshold,
        "min_cycles": cfg.min_cycles,
        "by_label": by_label.to_dict("index"),
        "stiction_below_threshold": missed.name.tolist(),
        "no_stiction_above_threshold": false_pos.name.tolist(),
        "zero_regularity_too_short": {r["name"]: round(r["cycles"], 1) for _, r in short.iterrows()},
        "zero_regularity_not_oscillating": quiet.name.tolist(),
        "db_runs": {r["name"]: round(r["regularity"], 2) for _, r in runs.iterrows()},
        "decision": "regularity is a confidence score carried into diagnosis, never a gate",
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
