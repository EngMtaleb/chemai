"""FastAPI service for Project 4 - desalination on the sun, and the moon.

Design notes worth reading before changing anything:

  * Nothing is fitted at startup. The design table is a module of numbers
    (chemai/api/sun_designs.py, generated from the step 7 and step 8 results);
    the two tests run on the caller's data in about a second. The service is
    stateless and cannot stop the other projects from booting.

  * /test is the declared method pointed at anyone's log: is a plant signal
    locked to the tide (or to the time of day)? The placebo periods run beside
    every answer, so a record that fools the null says so.

  * /design reads the solar-and-tidal design sweep: production capacity, tank,
    PV and tidal turbine in, renewable share and grid energy out. A synthetic
    year, no site, no prices - said on the page and in /health.
"""
from __future__ import annotations

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from chemai import __version__
from chemai.api.sun_designs import DESIGNS
from chemai.api.sun_page import PAGE
from chemai.api.sun_schemas import DesignResponse, SeriesTestRequest, SeriesTestResponse, SunHealthResponse
from chemai.evaluation.tide_test import series_test
from chemai.models.solar_plan import SolarPlanConfig

log = logging.getLogger(__name__)

# The declared results (projects/p04_tide_fouling/VALIDATION.md 4 and 7).
RESULTS = {
    "record": "seawater UF pilot, 2012-2015: 107 runs, 5,106 h, 14,998 filtration segments",
    "tide_fouling": {"p": 0.244, "answer": "no tide-locked fouling; a 10 % swing would have been found 99.5 % of the time"},
    "daylight_hours": {"p": 0.746, "contrast": -0.008, "ci95": [-0.025, 0.012],
                       "answer": "09-15 h foul like the rest of the day"},
    "restart": {"p": 0.382, "effect": 0.031, "ci95": [-0.039, 0.100],
                "answer": "the first 3 h after a restart foul like the rest"},
    "solar_design_90pc": {"capacity": 3.0, "storage_h": 16, "pv_kwp": 0.7, "renewable": 0.93,
                          "grid_kwh_m3": 0.23, "from_kwh_m3": 3.26},
    "with_tidal_0_10kW": {"capacity": 2.0, "storage_h": 8, "pv_kwp": 0.9, "renewable": 0.91},
}

LIMITS = [
    "One pilot at one intake. Its pump hydraulics suggest the feed came through storage that damps "
    "the tide; an intake drawing straight from the sea could behave differently - /test is how to find out.",
    "The tests cover the ultrafiltration pre-treatment only. Reverse osmosis under daily stop-start "
    "depends on a flush at each stop (literature), not on anything measured here.",
    "The design sweep is a synthetic year with no site and no prices: capacity, tanks, PV and turbines "
    "are traded in physical units. A real design needs the site's weather and costs.",
    "Running faster instead of adding modules raises fouling per cubic metre for any flux exponent "
    "above 1; this record cannot pin the exponent, so that cost is a range.",
    "A tidal turbine needs a strong current at the site; most sites have none.",
    "A deployed prototype: no historian connection, no authentication, no monitoring.",
]

_TABLE = {tuple(r[:5]): r for r in DESIGNS}


def create_sun_app() -> FastAPI:
    app = FastAPI(
        title="Desalination on the Sun - and the Moon",
        description=(
            "Two things a solar-run seawater plant needs to know. Does the pre-treatment membrane foul "
            "differently by the hour, after a restart, or with the tide? (No - over 5,106 hours of a "
            "public pilot.) And how much capacity, storage and PV - or a tidal turbine - buys a given "
            "share of renewable energy.\n\n/test runs the declared tide or time-of-day test on your own log."),
        version=__version__,
    )
    sec = SolarPlanConfig().sec

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=SunHealthResponse, tags=["ops"])
    def health() -> SunHealthResponse:
        return SunHealthResponse(status="ok", version=__version__, results=RESULTS, limits=LIMITS,
                                 data="UF pilot record: Zenodo 4630769 / Dryad doi:10.5068/D1310B (CC0)")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return PAGE

    @app.get("/designs", tags=["design"])
    def designs() -> dict:
        """Every design in the sweep, for the page's selectors."""
        return {"columns": ["weather", "capacity", "storage_h", "pv_kwp", "tidal_kw", "renewable_fraction",
                            "grid_kwh_m3", "curtailed_share", "run_hours_per_day", "rule_fraction"],
                "rows": DESIGNS, "sec_kwh_m3": sec}

    @app.get("/design", response_model=DesignResponse, tags=["design"])
    def design(weather: str = "sunny", capacity: float = 3.0, storage_h: float = 16, pv_kwp: float = 0.7,
               tidal_kw: float = 0.0) -> DesignResponse:
        key = (weather, float(capacity), float(storage_h), float(pv_kwp), float(tidal_kw))
        row = _TABLE.get(key)
        if row is None:
            raise HTTPException(status_code=404, detail="not in the sweep - GET /designs lists every design")
        w, c, s, p, k, ren, grid, curt, hours, rule = row
        return DesignResponse(
            weather=w, capacity=c, storage_h=s, pv_kwp=p, tidal_kw=k, renewable_fraction=ren,
            grid_kwh_m3=grid, curtailed_share=curt, run_hours_per_day=hours, rule_fraction=rule,
            per_1000_m3_day={"pv_kwp": round(p * 1000), "tidal_kw": round(k * 1000),
                             "tank_m3": round(s / 24 * 1000), "plant_m3_day": round(c * 1000),
                             "grid_mwh_year": round(grid * 365, 1),
                             "grid_mwh_year_without": round(sec * 365, 1)})

    @app.post("/test", response_model=SeriesTestResponse, tags=["test"])
    def test(req: SeriesTestRequest) -> SeriesTestResponse:
        # A stamp with an offset keeps its own wall-clock time: the test wants local time.
        df = pd.DataFrame({"t": [r.t.replace(tzinfo=None) for r in req.records],
                           "value": [r.value for r in req.records],
                           "run": [r.run for r in req.records]})
        if df["t"].duplicated().any():
            raise ValueError("timestamps repeat: one value per time")
        out = series_test(df, req.kind, req.n_surrogates, req.block_days)
        log.info("%s test on %d points -> %s", req.kind, out["n"], out.get("verdict", out["status"]))
        return SeriesTestResponse(**out)

    return app


sun_app = create_sun_app()
