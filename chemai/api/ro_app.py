"""FastAPI service for RO fouling and cleaning (Project 3).

Design notes worth reading before changing anything:

  * There is NO trained model to load. The forecast is a robust straight line
    through the current cycle, chosen on simulated development plants. The one
    alternative tested - the same line through the gap-law transform - gained
    one to two days, there and on the real plant, and was not adopted
    (VALIDATION.md sections 7 and 9). The service is stateless and starts
    instantly, so it cannot stop Projects 1 and 2 from booting on the same
    deployment.

  * The real question is /plan: one row per train with an action, a week, the
    evidence and an owner. /forecast is the same arithmetic for one train.

  * /normalise turns logged flows, pressures and temperatures into DP_norm. By
    default it measures the plant's own pressure-drop flow exponent from
    day-to-day flow changes; on simulated plants a wrong exponent cost more
    forecast accuracy than any model choice did (VALIDATION.md section 6).

  * The limits are served, not hidden: /health lists them and the demo page
    prints them.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from chemai import __version__
from chemai.api.ro_examples import build_example, example_index
from chemai.api.ro_page import PAGE
from chemai.api.ro_schemas import (ForecastRequest, ForecastResponse, NormaliseRequest,
                                   NormaliseResponse, NormalisedRow, PlanRequest, PlanResponse,
                                   PlanRowModel, ROHealthResponse)
from chemai.config import ROConfig
from chemai.evaluation.cip_plan import TrainRecord, build_plan
from chemai.features.ro_physics import fit_flow_exponent, normalise
from chemai.models.fouling_forecast import forecast_cycle, prepare_window

log = logging.getLogger(__name__)

LIMITS = [
    "On the one real plant it was tested on (three trains, eighteen cleanings), the date forecast "
    "did not beat a simple rule. 'The 7-day median is within 0.03 of the level' caught 43 % of "
    "the crossings inside 28 days at 93 % precision; the forecast caught 21 % at 72 %. The weekly "
    "plan therefore uses both - a rule adopted after that test (VALIDATION.md section 12).",
    "Level forecasts 14 days ahead were no better than assuming the level stays where it is "
    "(mean error 0.034 for both).",
    "The P10-P90 band is rough: it held 78 % of outcomes on simulated test plants, 89 % on the "
    "real plant and 69 % on a quieter simulated variant.",
    "A date is flagged when it looks further ahead than twice the days it has seen, or when most "
    "of its window sits on the baseline: on the real plant flagged dates missed by a median of "
    "59 days, the rest by 12.5.",
    "Reads the hydraulic signal only - normalised feed-channel pressure drop. Scaling of the tail "
    "elements and membrane damage show in permeate flow and salt passage first; review those "
    "separately. An upset (a bloom, a pretreatment failure) changes the rate overnight and no "
    "line sees it coming.",
    "Normalisation decides the answer: a wrong flow exponent turns a production change into "
    "fouling. Measure the plant's own exponent (/normalise does) before trusting a date.",
    "One published plant is not a validation, and no engineer has acted on one of its dates. "
    "A deployed prototype: no historian connection, no authentication, no monitoring.",
]


def create_ro_app(cfg: ROConfig | None = None) -> FastAPI:
    cfg = cfg or ROConfig()
    app = FastAPI(
        title="RO Fouling and Cleaning",
        description=(
            "When will each RO train reach its review level for a cleaning (CIP)? A forecast "
            "from the normalised feed-channel pressure drop, with a P10-P90 interval, an "
            "extrapolation flag, and a weekly plan with an action and an owner per train.\n\n"
            "Every date carries its interval; a record that cannot carry a forecast is answered "
            "with 'insufficient', not guessed."),
        version=__version__,
    )
    app.state.cfg = cfg

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=ROHealthResponse, tags=["ops"])
    def health() -> ROHealthResponse:
        return ROHealthResponse(
            status="ok", version=__version__, law=cfg.law, review_level_default=cfg.review_level,
            interval=f"P{int(round((1 - cfg.interval) / 2 * 100))}-P{int(round((1 + cfg.interval) / 2 * 100))}",
            limits=LIMITS,
            data="External test: Su, Meng & Chang (2026) Membranes 16(8) 257, Zenodo 21055948 (CC-BY-4.0)")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return PAGE

    @app.get("/examples", tags=["demo"])
    def examples() -> list[dict]:
        """Simulated cycles whose cause is known."""
        return example_index()

    @app.get("/examples/{key}", tags=["demo"])
    def example(key: str) -> dict:
        return build_example(key)

    @app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(req: ForecastRequest) -> ForecastResponse:
        c = req.cycle
        f = forecast_cycle(c.day, c.dp_norm, req.review_level, req.law, c.censored, cfg)
        t, _, _ = prepare_window(c.day, c.dp_norm, cfg, c.censored)
        log.info("%s -> %s (%s days)", c.name, f.status, f.days_to_level)
        return ForecastResponse(train=c.name, fit_window=[float(t[0]), float(t[-1])] if len(t) else [],
                                **f.to_dict())

    @app.post("/plan", response_model=PlanResponse, tags=["forecast"])
    def plan(req: PlanRequest) -> PlanResponse:
        trains = [TrainRecord(name=t.name, day=t.day, dp_norm=t.dp_norm, censored=t.censored,
                              dp_vessel_bar=t.dp_vessel_bar, dp_element_bar=t.dp_element_bar,
                              pre_cleaning=t.pre_cleaning, post_cleaning=t.post_cleaning,
                              previous_post_cleaning=t.previous_post_cleaning) for t in req.trains]
        p = build_plan(trains, cfg, req.review_level, req.trains_total)
        return PlanResponse(review_level=p.review_level, notes=p.notes,
                            rows=[PlanRowModel(**r.to_dict()) for r in p.rows])

    @app.post("/normalise", response_model=NormaliseResponse, tags=["normalisation"])
    def normalise_records(req: NormaliseRequest) -> NormaliseResponse:
        df = pd.DataFrame([r.model_dump() for r in req.records]).sort_values("day")
        if df["day"].duplicated().any():
            raise ValueError("one record per day (or per hour): a day is repeated")
        has_dp = df["dp"].notna().all()
        if not has_dp:
            df = df.drop(columns=["dp"])
        notes = []
        if req.flow_exponent == "fit":
            fit = fit_flow_exponent(df)
            n = fit["n"] if fit["ok"] else cfg.dp_flow_exponent
            if not fit["ok"]:
                notes.append(fit["note"])
        else:
            fit = {"ok": False, "note": "exponent given by the caller"}
            n = float(req.flow_exponent)
        m = float(np.clip(2.0 - n, 0.0, 0.5))
        if not has_dp:
            notes.append("dP taken as feed minus concentrate pressure from two transmitters: their "
                         "independent drift lands in DP_norm. A differential transmitter is better.")
        out = normalise(df, baseline_rows=req.baseline_rows, flow_exponent=n, visc_exponent=m, cfg=cfg)
        rows = [NormalisedRow(day=float(d), dp_norm=round(float(a), 4), npf_ratio=round(float(b), 4),
                              nsp_ratio=round(float(c), 4), ndp=round(float(e), 3))
                for d, a, b, c, e in zip(df["day"], out["dp_norm"], out["npf_ratio"],
                                         out["nsp_ratio"], out["ndp"])]
        return NormaliseResponse(flow_exponent=round(float(n), 3), viscosity_exponent=round(m, 3),
                                 exponent_fit=fit, rows=rows, notes=notes)

    return app


ro_app = create_ro_app()
