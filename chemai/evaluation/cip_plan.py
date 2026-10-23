"""The weekly CIP plan: which train, what to do, by when, why, and who.

A forecast date is not a decision. What a membrane plant needs on Monday is a
short table: one row per train, an action, a window, the evidence, a
confidence, and an owner. The rules below are deliberately few and are applied
in a fixed order, so anyone can check why a train landed where it did:

  1. the manufacturer's pressure-drop limit overrides everything;
  2. too little data since the last cleaning is said plainly, not guessed;
  3. at or above the review level: decide this week;
  4. within a small margin of the level, or a forecast crossing inside the
     planning window: plan the cleaning. The margin was added after the
     external test, where it beat the forecast on real data (VALIDATION.md 12);
  5. rising, but further out: watch, re-forecast next week;
  6. flat or falling: there is no hydraulic reason to clean.

And one check that runs alongside: a cleaning that barely moved the pressure
drop is a procedure problem (chemistry, pH, temperature, flow, soak time), not a
scheduling one - cleaning the same train again sooner will not fix it.

Two trains are never planned into the same week if that would leave fewer than
`min_trains_online` running.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from chemai.config import ROConfig
from chemai.models.fouling_forecast import Forecast, forecast_cycle

OWNERS = {
    "limit": "operations",
    "review": "process engineering",
    "plan": "operations planning",
    "watch": "process engineering",
    "none": "-",
    "data": "instrumentation",
    "wait": "-",
}


@dataclass
class TrainRecord:
    """One train's current cycle, as the plan reads it."""
    name: str
    day: list[float]                  # days since the last cleaning ended
    dp_norm: list[float]              # DP_norm over start-up (1.15 = +15 %)
    censored: list[bool] | None = None
    dp_vessel_bar: float | None = None    # raw dP across the worst vessel now, bar
    dp_element_bar: float | None = None   # raw dP across the lead element now, bar
    # the last cleaning: DP_norm just before it, just after it, and just after the
    # cleaning before that - enough to judge whether it worked
    pre_cleaning: float | None = None
    post_cleaning: float | None = None
    previous_post_cleaning: float | None = None


@dataclass
class PlanRow:
    train: str
    action: str
    category: str                 # limit | review | plan | watch | none | data | wait
    owner: str
    week: int | None              # planned week from now (1 = this week)
    level_now: float | None
    rate_per_week: float | None
    days_p10: float | None
    days_p50: float | None
    days_p90: float | None
    confidence: str               # high | medium | low | -
    evidence: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def cleaning_recovery(pre: float | None, post: float | None, previous_post: float | None) -> float | None:
    """Share of the rise since the previous cleaning that the last cleaning removed."""
    if pre is None or post is None or previous_post is None:
        return None
    rise = pre - previous_post
    if rise <= 0.02:                  # nothing to recover: no judgement possible
        return None
    return float((pre - post) / rise)


def _confidence(f: Forecast, cfg: ROConfig) -> str:
    if not f.in_envelope:
        return "low"
    if f.extrapolation_ratio is None or f.extrapolation_ratio <= 1.0:
        return "high"
    return "medium" if f.extrapolation_ratio <= cfg.extrapolation_ratio_max else "low"


def plan_train(tr: TrainRecord, cfg: ROConfig | None = None, review_level: float | None = None) -> PlanRow:
    cfg = cfg or ROConfig()
    level = cfg.review_level if review_level is None else review_level
    flags: list[str] = []
    rec = cleaning_recovery(tr.pre_cleaning, tr.post_cleaning, tr.previous_post_cleaning)
    if rec is not None and rec < cfg.min_recovery_share:
        flags.append(f"last cleaning removed only {rec:.0%} of the rise: check the CIP procedure "
                     "(chemistry, pH, temperature, flow, soak) and consider a lead-element autopsy")
    if tr.post_cleaning is not None and tr.post_cleaning >= level:
        flags.append(f"started this cycle already at {tr.post_cleaning:.2f}, at or above the review "
                     "level: the cleaning did not restore the membrane")

    f = forecast_cycle(tr.day, tr.dp_norm, level, censored=tr.censored, cfg=cfg)
    flags += f.warnings
    row = dict(train=tr.name, week=None, level_now=f.level_now, rate_per_week=f.rate_per_week,
               days_p10=f.days_lo, days_p50=f.days_to_level, days_p90=f.days_hi, flags=flags)

    over_vessel = tr.dp_vessel_bar is not None and tr.dp_vessel_bar >= cfg.dp_max_vessel_bar
    over_element = tr.dp_element_bar is not None and tr.dp_element_bar >= cfg.dp_max_element_bar
    if over_vessel or over_element:
        what = (f"vessel dP {tr.dp_vessel_bar:.2f} bar (limit {cfg.dp_max_vessel_bar})" if over_vessel
                else f"element dP {tr.dp_element_bar:.2f} bar (limit {cfg.dp_max_element_bar})")
        return PlanRow(action="Clean now: manufacturer pressure-drop limit reached",
                       category="limit", owner=OWNERS["limit"], confidence="high", evidence=what,
                       **{**row, "week": 1})

    if f.status == "insufficient":
        last = max(tr.day) if len(tr.day) else 0
        if last < cfg.min_span_days + cfg.skip_after_cip_days:
            return PlanRow(action="Too soon after the cleaning to judge: re-forecast next week",
                           category="wait", owner=OWNERS["wait"], confidence="-",
                           evidence=f"{len(tr.day)} days recorded since the cleaning", **row)
        return PlanRow(action="Forecast withheld: too few recorded days - check logging and transmitters",
                       category="data", owner=OWNERS["data"], confidence="-",
                       evidence=f"{f.n_points} usable points over {f.span_days:.0f} days", **row)

    conf = _confidence(f, cfg)
    rate = f"{f.rate_per_week * 100:+.2f} %/week" if f.rate_per_week is not None else ""
    if f.status == "at_or_above":
        return PlanRow(action=f"Decide this week: DP_norm {f.level_now:.2f} is at or above "
                              f"the review level {level:.2f}",
                       category="review", owner=OWNERS["review"], confidence=conf,
                       evidence=f"level {f.level_now:.3f}, trend {rate}", **{**row, "week": 1})
    near = _recent_median(tr, cfg)
    if f.status == "crossing" and f.days_to_level is not None \
            and f.days_to_level <= cfg.plan_weeks * 7:
        week = int(max(1, np.ceil(max(f.days_lo or 0.0, 1.0) / 7.0)))
        return PlanRow(action=f"Plan a CIP: review level likely in {f.days_to_level:.0f} days "
                              f"(P10-P90 {_fmt(f.days_lo)}-{_fmt(f.days_hi)})",
                       category="plan", owner=OWNERS["plan"], confidence=conf,
                       evidence=f"level {f.level_now:.3f}, trend {rate}", **{**row, "week": week})
    if near is not None and near >= level - cfg.plan_margin:
        return PlanRow(action=f"Plan a CIP: the 7-day median {near:.3f} is within {cfg.plan_margin:.2f} "
                              f"of the review level - day-to-day swings will reach it",
                       category="plan", owner=OWNERS["plan"], confidence=conf,
                       evidence=f"level {f.level_now:.3f}, trend {rate}",
                       **{**row, "week": min(cfg.plan_weeks, 2)})
    if f.status == "crossing":
        return PlanRow(action=f"Watch: review level in about {f.days_to_level:.0f} days at the current "
                              f"rate; re-forecast next week",
                       category="watch", owner=OWNERS["watch"], confidence=conf,
                       evidence=f"level {f.level_now:.3f}, trend {rate}", **row)
    if f.status == "falling":
        return PlanRow(action="No hydraulic reason to clean: the pressure drop is falling",
                       category="none", owner=OWNERS["none"], confidence=conf,
                       evidence=f"level {f.level_now:.3f}, trend {rate}", **row)
    return PlanRow(action=f"No cleaning needed for pressure drop within {cfg.horizon_max_days} days "
                          "at the current rate",
                   category="none", owner=OWNERS["none"], confidence=conf,
                   evidence=f"level {f.level_now:.3f}, trend {rate}", **row)


def _recent_median(tr: TrainRecord, cfg: ROConfig) -> float | None:
    """Median of the values recorded in the last `smooth_days` days."""
    if not len(tr.day):
        return None
    d = np.asarray(tr.day, float)
    x = np.asarray(tr.dp_norm, float)
    m = d > d.max() - cfg.smooth_days
    return float(np.median(x[m])) if m.sum() >= 3 else None


def _fmt(v: float | None) -> str:
    return "?" if v is None else f"{v:.0f}"


ORDER = {"limit": 0, "review": 1, "plan": 2, "data": 3, "watch": 4, "wait": 5, "none": 6}


@dataclass
class WeeklyPlan:
    rows: list[PlanRow]
    review_level: float
    notes: list[str]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.rows])


def build_plan(trains: list[TrainRecord], cfg: ROConfig | None = None,
               review_level: float | None = None, trains_total: int | None = None) -> WeeklyPlan:
    """One row per train, most urgent first, with cleanings staggered so that at
    least `min_trains_online` trains keep running."""
    cfg = cfg or ROConfig()
    level = cfg.review_level if review_level is None else review_level
    rows = [plan_train(t, cfg, level) for t in trains]
    rows.sort(key=lambda r: (ORDER[r.category], r.days_p50 if r.days_p50 is not None else 1e9))
    n_total = trains_total or len(trains)
    per_week = max(1, n_total - cfg.min_trains_online)
    taken: dict[int, int] = {}
    notes = [f"Review level {level:.2f} = DP_norm over start-up; the vendor rule of +15 % since "
             "start-up is 1.15. Replace it with the plant's own level.",
             "Pressure drop is the only fouling signal used here. Salt passage and permeate flow "
             "need their own review."]
    for r in rows:
        if r.week is None:
            continue
        w = r.week
        while taken.get(w, 0) >= per_week and r.category != "limit":
            w += 1
        if w != r.week:
            r.flags.append(f"moved from week {r.week} to week {w}: keeps {cfg.min_trains_online} "
                           "trains online")
            r.week = w
        taken[w] = taken.get(w, 0) + 1
    if sum(1 for r in rows if r.category == "limit") > per_week:
        notes.append("More trains are at the manufacturer limit than can be cleaned while keeping "
                     "supply: escalate - capacity, not scheduling, is the constraint.")
    return WeeklyPlan(rows=rows, review_level=level, notes=notes)
