"""The weekly loop report - the output an engineer actually reads.

A plant has hundreds of loops and one question: which ten do I fix this week,
and who fixes them? Ranking by how hard a loop oscillates answers the wrong
question - a surge drum swinging by 8 % is doing its job, while a quality loop
swinging by 1 % burns energy every hour.

    priority = severity x location weight x fault weight x confidence

Severity and confidence are measured from the data. The LOCATION WEIGHT comes
from the plant: an engineer grades each loop 1-3 once. Without that table the
report still runs, ranking by severity alone and saying so at the top - no
weights are invented (VALIDATION.md 18).

Every factor is kept as its own column. An engineer who cannot see why a loop
rose or fell will not trust the ranking, and a black box is not used twice.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from chemai.config import ACTIONS, ControlLoopConfig, ReportConfig

log = logging.getLogger(__name__)

REPORT_COLUMNS = ["loop", "diagnosis", "evidence", "confidence_band", "status",
                  "priority", "action", "owner"]


def diagnose(regularity: float, shape: float, flags=(), cfg: ControlLoopConfig | None = None) -> str:
    """The adopted baseline (VALIDATION.md 16.2, 17.3), with the data-quality
    flags taking precedence: a frozen sensor or a loop in manual is not a
    control diagnosis at all, and saying 'stiction' about it would be wrong."""
    cfg = cfg or ControlLoopConfig()
    flags = set(flags)
    if "frozen_sensor" in flags:
        return "frozen_sensor"
    if {"op_constant", "op_inactive"} & flags:
        return "manual"
    if "saturated" in flags:
        return "saturation"
    if not np.isfinite(regularity) or regularity <= cfg.regularity_threshold:
        return "undetermined"
    if not np.isfinite(shape):
        return "undetermined"
    return "stiction" if shape > cfg.shape_triangular else "tuning"


def confidence(regularity: float, cfg: ReportConfig | None = None) -> float:
    """0 to 1, from the oscillation regularity. Confidence changes the ACTION,
    never whether a loop appears: a big problem diagnosed with medium confidence
    must still reach the engineer (VALIDATION.md 18.2)."""
    cfg = cfg or ReportConfig()
    if not np.isfinite(regularity):
        return 0.0
    return float(min(max(regularity, 0.0), cfg.confidence_full) / cfg.confidence_full)


def confidence_band(value: float, cfg: ReportConfig | None = None) -> str:
    cfg = cfg or ReportConfig()
    if value >= cfg.confidence_high:
        return "high"
    return "medium" if value >= cfg.confidence_medium else "low"


def action_for(diagnosis: str, band: str) -> tuple[str, str]:
    """(what to do, who owns it). A confident stiction diagnosis opens a work
    order; a weaker one asks for a field check first."""
    action, owner = ACTIONS.get(diagnosis, ACTIONS["undetermined"])
    if diagnosis == "stiction" and band != "high":
        return "Field check before raising a work order: " + action.lower(), owner
    return action, owner


@dataclass
class WeeklyReport:
    table: pd.DataFrame
    note: str
    ranked_by_severity_only: bool

    def top(self, n: int | None = None) -> pd.DataFrame:
        return self.table.head(n or len(self.table))[REPORT_COLUMNS]


def build_report(loops: pd.DataFrame, location_weights: dict | None = None,
                 previous: pd.DataFrame | None = None,
                 cfg: ReportConfig | None = None,
                 loop_cfg: ControlLoopConfig | None = None) -> WeeklyReport:
    """Rank loops for one week.

    `loops` needs: loop, regularity, shape; optionally severity, period_min,
    flags. `location_weights` maps loop name -> 1, 2 or 3. `previous` is last
    week's table, used only to mark a loop confirmed (section 18.3).
    """
    cfg = cfg or ReportConfig()
    loop_cfg = loop_cfg or ControlLoopConfig()
    df = loops.copy()
    # note: df.flags is a pandas attribute, so the column is always taken by name
    quality_flags = (df["flags"] if "flags" in df else
                     pd.Series([[]] * len(df), index=df.index))

    df["diagnosis"] = [diagnose(r, s, f, loop_cfg)
                       for r, s, f in zip(df.regularity, df["shape"], quality_flags)]
    df["confidence"] = [confidence(r, cfg) for r in df["regularity"]]
    df["confidence_band"] = [confidence_band(c, cfg) for c in df.confidence]

    severity_known = ("severity" in df and df["severity"].notna().any()
                      and df["severity"].std() > 0)
    if not severity_known:
        df["severity"] = 1.0

    weights_known = bool(location_weights)
    df["location"] = (df["loop"].map(location_weights).fillna(cfg.default_location_weight)
                      if weights_known else cfg.default_location_weight)
    df["fault_weight"] = [cfg.fault_weight(d) for d in df.diagnosis]
    df["priority"] = (df["severity"] * df["location"] * df["fault_weight"]
                      * df["confidence"]).round(3)

    seen_before = set(previous["loop"]) if previous is not None and len(previous) else set()
    df["status"] = np.where(df["loop"].isin(seen_before), "confirmed", "new")

    actions = [action_for(d, b) for d, b in zip(df["diagnosis"], df["confidence_band"])]
    df["action"] = [a for a, _ in actions]
    df["owner"] = [o for _, o in actions]
    # a loop only earns a work order once it has come back (section 18.3)
    first_time = (df["status"] == "new") & (df["diagnosis"] != "frozen_sensor")
    df.loc[first_time, "action"] = "Watch - first appearance, not yet confirmed"

    df["evidence"] = [
        " · ".join(filter(None, [
            f"period {p:.0f} min" if "period_min" in df and np.isfinite(p) else "",
            f"regularity {r:.2f}" if np.isfinite(r) else "",
            f"shape {s:.2f}" if np.isfinite(s) else "no shape (too few cycles)"]))
        for p, r, s in zip(df.get("period_min", pd.Series(np.nan, index=df.index)),
                           df["regularity"], df["shape"])]

    notes = []
    if not weights_known:
        notes.append("Ranked WITHOUT plant location weights - severity and diagnosis only")
    if not severity_known:
        notes.append("Severity unavailable (normalised or constant data) - excluded from the ranking")
    note = " | ".join(notes) or "Ranked with plant location weights"
    log.info("report: %d loops, %s", len(df), note)
    return WeeklyReport(df.sort_values("priority", ascending=False).reset_index(drop=True),
                        note, not weights_known)
