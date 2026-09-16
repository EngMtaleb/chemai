"""Project 2 - Week 3: the weekly loop report.

Run:  python projects/p02_control_loops/week3_report.py

Two runs:
  1. Eastman - 30 loops from one plant, recorded together, with a published
     root cause (LC2, tag 22) and a schematic to grade locations from.
  2. Every labelled single loop in SACAC and ISDB, with no location table -
     the state a plant starts in before an engineer fills one.

Teaching version: notebook p02_week3_report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import detrend

from chemai.config import DATA_DIR, ControlLoopConfig, ReportConfig
from chemai.data import load_isdb, load_sacac
from chemai.evaluation import build_report
from chemai.features import (assess, cycles_in_record, oscillation_regularity,
                             prepare_for_shape, shape_index, shape_signal)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week3")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week3_report.json"

# Location weights for Eastman, graded from the published schematic
# (Thornhill, Cox & Paulonis, 2003). Tag 22 is LC2, the documented root cause;
# tags 23-28 and 30 are column 3, which carries the product. Everything else
# keeps the default of 1 because the schematic does not say more - a declared
# default, not a guess (VALIDATION.md 18.4).
EASTMAN_LOCATIONS = {22: 2, **{t: 3 for t in (23, 24, 25, 26, 27, 28, 30)}}
EASTMAN_TS = 20.0


def measure(sp, pv, op, loop_type: str, ts: float, cfg: ControlLoopConfig) -> dict:
    """Period, regularity and shape for one loop, by the rules of sections 15-16."""
    e = detrend(np.asarray(sp, float) - np.asarray(pv, float))
    period, regularity = oscillation_regularity(e)
    cycles = cycles_in_record(len(e), period)
    shape = np.nan
    if np.isfinite(period) and cycles >= cfg.min_cycles:
        signal, _ = shape_signal(loop_type, pv, sp, op)
        shape = shape_index(prepare_for_shape(signal, period, cfg.shape_harmonics))
    return {"period_min": period * ts / 60 if np.isfinite(period) else np.nan,
            "regularity": regularity, "cycles": cycles, "shape": shape,
            "severity": float(np.std(e))}


def eastman_report(cfg: ControlLoopConfig, rcfg: ReportConfig) -> dict:
    path = next((DATA_DIR / "sacac").rglob("EastmanDatasetFromNFThornhill_Data.mat"))
    m = sio.loadmat(path)
    err, op = m["errmat"], m["opmat"]
    rows = []
    for j in range(err.shape[1]):
        e = err[:, j]
        if not np.isfinite(e).all():
            continue
        # the published data is mean-centred and normalised, so PV = SP - err
        f = measure(np.zeros(len(e)), -e, op[:, j], "L", EASTMAN_TS, cfg)
        rows.append({"loop": j + 1, **f})
    df = pd.DataFrame(rows)

    report = build_report(df, EASTMAN_LOCATIONS, cfg=rcfg, loop_cfg=cfg)
    log.info("EASTMAN - %s", report.note)
    log.info("\n%s", report.top(6).to_string(index=False))

    top = report.table.iloc[0]
    log.info("root cause published by Thornhill et al. (2003): tag 22 (LC2), sticking valve")
    log.info("report puts tag %s first, priority %.2f (second: %.2f)",
             top.loop, top.priority, report.table.iloc[1].priority)
    return {"first": int(top.loop), "diagnosis": top.diagnosis,
            "priority": float(top.priority),
            "second_priority": float(report.table.iloc[1].priority),
            "n_undetermined": int((report.table.diagnosis == "undetermined").sum()),
            "n_loops": int(len(report.table)), "note": report.note}


def archive_report(cfg: ControlLoopConfig, rcfg: ReportConfig) -> dict:
    """Every labelled single loop, as a plant would see them with no location table."""
    rows = []
    for r in load_sacac(DATA_DIR / "sacac"):
        if r.label is None:
            continue
        d = r.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        pv, op = d.pv.to_numpy(), d.op.to_numpy()
        if not np.isfinite(pv).all():
            continue
        rows.append({"loop": r.name, "label": r.label, "flags": assess(sp, pv, op).flags,
                     **measure(sp, pv, op, r.loop_type, r.ts or 1.0, cfg)})
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.label is None or x.in_sacac:
            continue
        d = x.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        pv, op = d.pv.to_numpy(), d.op.to_numpy()
        if not np.isfinite(pv).all():
            continue
        rows.append({"loop": x.key, "label": x.label, "flags": assess(sp, pv, op).flags,
                     **measure(sp, pv, op, x.loop_type, x.ts or 1.0, cfg)})
    df = pd.DataFrame(rows)
    # Severity is the standard deviation of the control error, in the loop's own
    # units. These loops come from different plants, and several are normalised -
    # comparing their standard deviations would rank by unit, not by cost. Severity
    # is only comparable inside one plant (VALIDATION.md 18.5).
    df = df.drop(columns="severity")

    report = build_report(df, cfg=rcfg, loop_cfg=cfg)
    log.info("ARCHIVE - %s (severity dropped: loops are in different units)", report.note)
    log.info("\n%s", report.top(rcfg.top_n).to_string(index=False))

    table = report.table
    agreement = pd.crosstab(table.label, table.diagnosis)
    log.info("published label against report diagnosis\n%s", agreement.to_string())
    owners = table.head(rcfg.top_n).owner.value_counts().to_dict()
    log.info("owners in the top %d: %s", rcfg.top_n, owners)
    return {"n_loops": int(len(table)), "note": report.note,
            "top": table.head(rcfg.top_n)[["loop", "label", "diagnosis", "priority",
                                           "owner"]].to_dict("records"),
            "owners_in_top": owners,
            "diagnosis_counts": table.diagnosis.value_counts().to_dict()}


def main() -> dict:
    cfg, rcfg = ControlLoopConfig(), ReportConfig()
    results = {"fault_weights": dict(rcfg.fault_weights),
               "eastman": eastman_report(cfg, rcfg),
               "archive": archive_report(cfg, rcfg)}
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
