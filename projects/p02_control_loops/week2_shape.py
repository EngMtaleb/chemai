"""Project 2 - Week 2, part 3: triangular or sinusoidal?

Run:  python projects/p02_control_loops/week2_shape.py

Establishes the BASELINE the Week 2 classifier must beat: the shape index
alone, at a threshold of 0.6, on every labelled real loop with enough cycles.
Teaching version: notebook p02_week2_shape.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import load_isdb, load_sacac
from chemai.features import (cycles_in_record, oscillation_regularity,
                             prepare_for_shape, shape_index, shape_signal)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week2shape")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week2_shape.json"


def labelled_loops():
    for r in load_sacac(DATA_DIR / "sacac"):
        if r.label:
            d = r.data
            sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
            yield "SACAC", r.name, r.label, r.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy()
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.label and not x.in_sacac:
            d = x.data
            sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
            label = x.label if x.tier == "stated" else f"{x.label} ({x.tier})"
            yield "ISDB", x.key, label, x.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy()


def main() -> dict:
    cfg = ControlLoopConfig()
    rows, skipped = [], []
    for src, name, label, ltype, sp, pv, op in labelled_loops():
        if not np.isfinite(pv).all():
            continue
        period, r = oscillation_regularity(detrend(pv - sp))
        cycles = cycles_in_record(len(pv), period)
        if cycles < cfg.min_cycles:
            skipped.append({"name": name, "cycles": round(cycles, 1)})
            continue
        signal, which = shape_signal(ltype, pv, sp, op)
        rows.append(dict(source=src, name=name, label=label, loop_type=ltype, signal=which,
                         period=round(period, 1), regularity=round(r, 2), cycles=round(cycles, 1),
                         shape=round(shape_index(prepare_for_shape(signal, period,
                                                                   cfg.shape_harmonics)), 2)))
    df = pd.DataFrame(rows)
    log.info("%d labelled loops with at least %.0f cycles (%d skipped)",
             len(df), cfg.min_cycles, len(skipped))
    log.info("shape index by label\n%s", df.groupby("label")["shape"].agg(
        n="size", median="median", low="min", high="max").round(2).to_string())

    # the baseline: shape index alone, on loops that actually oscillate
    osc = df[df.regularity > cfg.regularity_threshold].copy()
    osc["is_stiction"] = osc.label.str.startswith("stiction")
    scores = {}
    for thr in (0.5, 0.55, 0.6, 0.65, 0.7):
        tp = int(((osc["shape"] > thr) & osc.is_stiction).sum())
        fp = int(((osc["shape"] > thr) & ~osc.is_stiction).sum())
        fn = int(((osc["shape"] <= thr) & osc.is_stiction).sum())
        scores[thr] = {"caught": tp, "of": tp + fn, "false_alarms": fp,
                       "precision": round(tp / max(tp + fp, 1), 2),
                       "recall": round(tp / max(tp + fn, 1), 2)}
        log.info("threshold %.2f: caught %d of %d, %d false alarms (precision %.2f)",
                 thr, tp, tp + fn, fp, scores[thr]["precision"])

    thr = cfg.shape_triangular
    missed = osc[osc.is_stiction & (osc["shape"] <= thr)]
    false_pos = osc[~osc.is_stiction & (osc["shape"] > thr)]
    log.info("stiction read as sinusoidal:\n%s",
             missed[["name", "loop_type", "signal", "shape", "regularity"]].to_string(index=False))
    log.info("not stiction but read as triangular:\n%s",
             false_pos[["name", "label", "loop_type", "signal", "shape"]].to_string(index=False))

    results = {"n_measured": int(len(df)), "n_skipped_few_cycles": len(skipped),
               "skipped": skipped, "n_oscillating": int(len(osc)),
               "baseline_by_threshold": scores, "adopted_threshold": thr,
               "baseline": scores[thr], "missed": missed.name.tolist(),
               "false_alarms": false_pos.name.tolist(),
               "by_label": df.groupby("label")["shape"].median().round(2).to_dict()}
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
