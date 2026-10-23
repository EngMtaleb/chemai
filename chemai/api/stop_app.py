"""FastAPI service for Project 5 - when to stop.

Design notes worth reading before changing anything:

  * Nothing is fitted at startup. The rule is a few lines of arithmetic on the
    caller's own log; the worked examples are a module of numbers
    (chemai/api/stop_examples.py, generated from the public records).

  * /batch is the rule tested in projects/p05_when_to_stop (declared, then run
    once on 153 held-out batches): stop when the live cost of the next cubic
    metre reaches the average cost so far. It assumes the batch holds its
    setpoints; with a moving flux the live cost moves for other reasons.

  * /interval and /law are the same rule for slow cycles (days between
    cleanings): the square-root law and the cost a cleaning would have to have
    for the interval actually run to be the best one. Prices are the caller's.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from chemai import __version__
from chemai.api.stop_examples import EXAMPLES
from chemai.api.stop_page import PAGE
from chemai.api.stop_schemas import (
    BatchRequest, BatchResponse, IntervalRequest, IntervalResponse, LawResponse, StopHealthResponse,
)
from chemai.models.stopping import average_cost, batch_decision, cumulative, interval_economics, rolling_median, sqrt_law

log = logging.getLogger(__name__)

# The declared results (projects/p05_when_to_stop/VALIDATION.md 4, 6, 7).
RESULTS = {
    "batch_rule_test": {
        "batches": 153, "record": "closed-circuit RO testbed, Bayesian-optimisation runs (held out)",
        "median_above_best_pct": {"b=1": 0.069, "b=10": 0.019}, "p90_above_best_pct": {"b=1": 0.112, "b=10": 0.061},
        "optimiser_ended_while_cost_still_falling_pct": {"BO (test)": 90, "ESC (development)": 100},
        "square_root_law_forecast": "within 4-7 % for b <= 3; 11 % short at b = 10 (declared limit 10 %: failed)"},
    "ro_cleaning": {"cycles": 15, "energy_paid_for_a_cleaning": "0 of 15 (mid case); at most 6 of 15 in any of 216 cases",
                    "no_energy_rise": "9 of 15 cycles",
                    "longer_cycles_left_more_pressure_drop": {"spearman": 0.63, "p": 0.016}},
    "uf_backwash": {"runs": 104, "backwash_share_of_feed": 0.287,
                    "energy_and_water_alone_would_filter_per_backwash": "15.7x more (median; 6.9x in the corner case)"},
    "uf_chemical_clean": {"intervals": 2, "t_star_over_actual": [1.07, 1.28]},
}

LIMITS = [
    "The batch rule was tested on a one-element NaCl testbed whose pumps run at ~12 % marginal efficiency; "
    "its energy is ten times a full-scale plant's. The rule carries over; the testbed's numbers do not.",
    "The rule assumes the flux and recirculation setpoints are held for the batch. A controller that moves "
    "them changes the live cost for reasons that have nothing to do with when to stop.",
    "Prices are yours: b is brine in kWh per m3 of brine, and a cleaning's cost is per m3/day of capacity "
    "in your money unit. The project's results were run on stated grids, not on any plant's prices.",
    "For slow cycles the rule counts only the cost you give it. Energy alone rarely justifies an RO "
    "cleaning; pressure-drop limits, salt passage and fouling that hardens are costs it does not see.",
    "A deployed prototype: no historian connection, no authentication, no monitoring.",
]

SOURCES = {
    "ccro_testbed": "doi:10.7481/1963162 (CC BY 4.0)",
    "ro_plant_deposit": "Su, Meng & Chang 2026, Membranes 16(8) 257; data doi:10.5281/zenodo.21055948 (CC BY 4.0)",
    "uf_pilot": "doi:10.5068/D1310B, also Zenodo 4630769 (CC0)",
}


def _clean(a) -> list:
    return [None if x is None or not math.isfinite(x) else round(float(x), 6) for x in a]


def create_stop_app() -> FastAPI:
    app = FastAPI(
        title="When to Stop",
        description=(
            "One rule for four decisions in a desalination plant: end a closed-circuit RO batch, backwash a "
            "filter, clean it chemically, clean an RO train. Stop when the live cost of the next cubic metre "
            "reaches the average cost of the cycle so far. Tested on 153 held-out batches of a public RO "
            "testbed: a median 0.02-0.07 % above each batch's best end.\n\n/batch runs it on your own log."),
        version=__version__,
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page():
        return PAGE

    @app.get("/health", response_model=StopHealthResponse, tags=["ops"])
    def health():
        return StopHealthResponse(status="ok", version=__version__, results=RESULTS, limits=LIMITS, sources=SOURCES)

    @app.get("/examples", tags=["examples"])
    def examples():
        return EXAMPLES

    @app.post("/batch", response_model=BatchResponse, tags=["rule"])
    def batch(req: BatchRequest):
        fixed = req.drain_kwh + req.b * req.brine_l / 1000.0
        r = batch_decision(req.t, req.power_kw, req.permeate_lpm, fixed, req.window_s, req.min_minutes,
                           conductivity=req.conductivity)
        if r["status"] == "insufficient":
            return BatchResponse(status="insufficient", fixed_kwh=fixed,
                                 verdict=f"too little filtration after minute {req.min_minutes:g} to judge",
                                 notes=["the live cost needs permeate flow above a tenth of its median"])
        notes = []
        if r["status"] == "stop":
            s = r["stop"]
            verdict = (f"stop at minute {s['minutes']:.1f} ({s['permeate_l']:.0f} L of permeate): the live cost "
                       f"{s['live_kwh_m3']:.2f} kWh/m3 reached the average {s['average_kwh_m3']:.2f}; "
                       f"{r['regret_pct']:.2f} % above the best end in hindsight")
        else:
            verdict = ("run longer: when the log ends the live cost is still below the average, so the "
                       "average was still falling")
        if r["best_is_at_end"]:
            notes.append("the best end in hindsight is the log's last minute: the true best lies beyond it")
        if r["law"] is not None and r["law"]["slope_kwh_m3_per_m3"] <= 0:
            notes.append("the live cost did not rise in the fitting window: the square-root law has no stop")
        notes.append(f"fixed cost {fixed:.4f} kWh = drain {req.drain_kwh:g} kWh + b {req.b:g} x brine {req.brine_l:g} L")
        law = r["law"]
        if law is not None and not math.isfinite(law["v_star_l"]):
            law = None
        return BatchResponse(status=r["status"], verdict=verdict, stop=r["stop"], best_in_hindsight=r["best"],
                             end=r["end"], regret_pct=r["regret_pct"], end_regret_pct=r["end_regret_pct"],
                             fixed_kwh=fixed, law=law, curve=r["curve"], notes=notes)

    @app.post("/interval", response_model=IntervalResponse, tags=["rule"])
    def interval(req: IntervalRequest):
        order = np.argsort(req.day)
        day = np.asarray(req.day, float)[order]
        cost = np.asarray(req.live_cost, float)[order]
        e = interval_economics(day, cost, req.fixed_cost, smooth=req.smooth_days, min_t=req.min_days)
        rose = math.isfinite(e.slope) and e.slope > 0
        if e.rule_time is not None:
            verdict = (f"clean at day {e.rule_time:g}: the live cost reached the average cost of the cycle, "
                       f"cleaning included")
        elif rose:
            verdict = (f"waiting paid: the live cost never reached the average within {e.length:g} days; "
                       f"the square-root law puts the best interval at {e.t_star:.0f} days")
        else:
            verdict = "the live cost never rose: on this cost alone there is no reason to clean"
        m = rolling_median(day * 86400.0, cost, req.smooth_days * 86400.0, min_periods=1)
        avg = average_cost(req.fixed_cost, cumulative(day, cost), day)
        return IntervalResponse(
            verdict=verdict, length_days=e.length, slope_per_day=e.slope if math.isfinite(e.slope) else None,
            t_star_days=e.t_star if rose else None, rule_day=e.rule_time,
            implied_fixed_cost=e.implied_fixed if rose else None, average_cost_at_end=e.j_actual,
            curve={"day": _clean(day), "live": _clean(m), "average": _clean(avg)})

    @app.get("/law", response_model=LawResponse, tags=["rule"])
    def law(fixed: float = Query(gt=0, description="Cost of ending a cycle"),
            slope: float = Query(description="How fast the live cost grows, per unit of output or time"),
            m0: float = Query(default=0.0, description="Live cost at the start of the cycle")):
        v, j = sqrt_law(fixed, slope, m0)
        if not math.isfinite(v):
            return LawResponse(fixed=fixed, slope=slope, m0=m0, best_length=None, best_average_cost=None,
                               note="a live cost that does not grow never pays for a stop")
        return LawResponse(fixed=fixed, slope=slope, m0=m0, best_length=v, best_average_cost=j,
                           note="best length sqrt(2 F / s); twice the fixed cost -> sqrt(2) times longer")

    return app
