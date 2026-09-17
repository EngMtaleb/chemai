"""The whole diagnosis in one call - data in, report out.

Six weeks of work sit behind two functions:

    analyse_loop    one loop  -> quality flags, indices, diagnosis, action, owner
    analyse_plant   many loops -> the weekly report, plus propagation and cascades

Order matters, and it is the order the weeks were built in:

  1. data quality first (Week 2) - a frozen sensor or a loop in manual is not a
     control diagnosis, and a coarse archive cannot carry a shape at all;
  2. indices second (Weeks 1-2) - Harris is deliberately NOT part of the
     diagnosis: it ranks, it does not diagnose, and it needs a dead time that a
     plant rarely knows (VALIDATION.md 11.2, 17.2);
  3. the adopted baseline third (Weeks 2-3) - a regular oscillation with a
     triangular shape is stiction; everything else is undetermined;
  4. the plant view last (Weeks 4-5) - loops sharing one oscillation are ONE
     event, so their victims get no work orders.

Nothing here invents a number the data cannot support. Where the plant has not
supplied location weights, the report says so instead of guessing them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from scipy.signal import detrend

from chemai.config import ControlLoopConfig, DataQualityConfig, ReportConfig
from chemai.data import samples_per_cycle
from chemai.evaluation.loop_report import build_report, confidence, confidence_band, diagnose
from chemai.evaluation.loop_report import action_for
from chemai.features import (assess, cycles_in_record, find_cascades, find_oscillation_cluster,
                             oscillation_regularity, prepare_for_shape, rank_source_candidates,
                             shape_index, shape_signal)

log = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Everything known about one loop, and what to do about it."""
    loop: str
    loop_type: str
    ts: float
    n: int
    period_s: float
    regularity: float
    cycles: float
    samples_per_cycle: float
    shape: float
    shape_signal: str
    diagnosis: str
    confidence: float
    confidence_band: str
    action: str
    owner: str
    quality_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        period = ("" if not np.isfinite(self.period_s) else
                  f"period {self.period_s:.0f} s" if self.period_s < 120 else
                  f"period {self.period_s / 60:.0f} min")
        d["evidence"] = " · ".join(filter(None, [
            period,
            f"regularity {self.regularity:.2f}",
            f"shape {self.shape:.2f}" if np.isfinite(self.shape) else "no shape",
        ]))
        return d


def analyse_loop(sp, pv, op, loop: str = "loop", loop_type: str = "F", ts: float = 1.0,
                 cfg: ControlLoopConfig | None = None, dq: DataQualityConfig | None = None,
                 rcfg: ReportConfig | None = None) -> LoopResult:
    """One loop, from raw samples to an action and an owner."""
    cfg, dq, rcfg = cfg or ControlLoopConfig(), dq or DataQualityConfig(), rcfg or ReportConfig()
    sp, pv, op = (np.asarray(v, dtype=float) for v in (sp, pv, op))
    if not (len(sp) == len(pv) == len(op)):
        raise ValueError("sp, pv and op must be the same length")
    if len(pv) < 10:
        raise ValueError("a record of fewer than 10 samples cannot be analysed")

    error = detrend(sp - pv)
    period, regularity = oscillation_regularity(error)
    cycles = cycles_in_record(len(pv), period)
    per_cycle = samples_per_cycle(period)

    quality = assess(sp, pv, op, dq, period=period)

    signal, which = shape_signal(loop_type, pv, sp, op)
    shape = float("nan")
    if np.isfinite(period) and cycles >= cfg.min_cycles and quality.ok_for_pv_shape:
        if which == "OP" and not quality.ok_for_op_shape:
            pass                                     # OP is unusable; no shape is reported
        else:
            shape = shape_index(prepare_for_shape(signal, period, cfg.shape_harmonics))

    call = diagnose(regularity, shape, quality.flags, cfg)
    conf = confidence(regularity, rcfg)
    band = confidence_band(conf, rcfg)
    action, owner = action_for(call, band)

    notes = list(quality.notes)
    if np.isfinite(period) and cycles < cfg.min_cycles:
        notes.append(f"only {cycles:.1f} cycles in the record - too short to judge (section 15.1)")
    return LoopResult(
        loop=loop, loop_type=loop_type, ts=ts, n=len(pv),
        period_s=period * ts if np.isfinite(period) else float("nan"),
        regularity=round(regularity, 2), cycles=round(cycles, 1),
        samples_per_cycle=round(per_cycle, 1), shape=round(shape, 2) if np.isfinite(shape) else shape,
        shape_signal=which, diagnosis=call, confidence=round(conf, 2), confidence_band=band,
        action=action, owner=owner, quality_flags=quality.flags, notes=notes)


def analyse_plant(loops: dict[str, dict], ts: float = 1.0,
                  location_weights: dict | None = None, previous: pd.DataFrame | None = None,
                  cfg: ControlLoopConfig | None = None, dq: DataQualityConfig | None = None,
                  rcfg: ReportConfig | None = None) -> dict:
    """Every loop in a unit, plus the connections between them.

    `loops` maps a name to {'sp': [...], 'pv': [...], 'op': [...], 'loop_type': 'F'}.
    Loops recorded at the same time can also be tested for shared oscillation
    (Week 5) and cascade structure (Week 4); with a single loop those are skipped.
    """
    cfg, dq, rcfg = cfg or ControlLoopConfig(), dq or DataQualityConfig(), rcfg or ReportConfig()
    results = [analyse_loop(d["sp"], d["pv"], d["op"], name, d.get("loop_type", "F"), ts,
                            cfg, dq, rcfg) for name, d in loops.items()]
    table = pd.DataFrame([r.as_row() for r in results])

    names = list(loops)
    lengths = {len(loops[n]["pv"]) for n in names}
    simultaneous = len(names) > 1 and len(lengths) == 1

    event = cascades = None
    if simultaneous:
        errors = np.column_stack([np.asarray(loops[n]["sp"], float)
                                  - np.asarray(loops[n]["pv"], float) for n in names])
        event = find_oscillation_cluster(errors, names, fs=1 / ts)
        setpoints = np.column_stack([np.asarray(loops[n]["sp"], float) for n in names])
        outputs = np.column_stack([np.asarray(loops[n]["op"], float) for n in names])
        cascades = find_cascades(setpoints, outputs, names)

    report = build_report(table[["loop", "regularity", "shape", "period_min"]]
                          if "period_min" in table else
                          table.assign(period_min=table.period_s / 60)
                          [["loop", "regularity", "shape", "period_min"]],
                          location_weights, previous, rcfg, cfg)

    # one event, not many faults: victims are not given their own work orders
    if event is not None and len(event.members) > 1:
        victims = set(event.victims())
        mask = report.table.loop.isin(victims)
        report.table.loc[mask, "action"] = (
            "Likely victim of the propagation event - no work order until the source is repaired")
        report.table.loc[mask, "priority"] = (report.table.loc[mask, "priority"] / 2).round(3)
        report.table.sort_values("priority", ascending=False, inplace=True)
        report.table.reset_index(drop=True, inplace=True)

    log.info("analysed %d loops; propagation event: %s; cascade candidates: %d",
             len(results), "yes" if event else "no", 0 if cascades is None else len(cascades))
    return {"loops": results, "table": table, "report": report,
            "propagation": None if event is None else {
                "period_s": event.period, "members": event.members,
                "candidates": rank_source_candidates(event).to_dict("records"),
                "note": event.note},
            "cascades": None if cascades is None else cascades.to_dict("records"),
            "simultaneous": simultaneous}
