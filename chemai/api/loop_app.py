"""FastAPI service for control-loop diagnosis (Project 2).

Design notes worth reading before changing anything:

  * There is NO model to load. The diagnosis is the adopted baseline - a
    regular oscillation with a triangular waveform is stiction - which beat a
    trained classifier on plant data (VALIDATION.md 17.3). So the service is
    stateless and starts instantly.

  * No endpoint returns a bare verdict. Every diagnosis carries its evidence,
    its confidence band, the action and the owner. An engineer who cannot see
    why a loop was called stiction should not act on it.

  * /analyse/plant is the real endpoint. One loop at a time cannot see that ten
    "faults" are one oscillation with nine victims, which is the finding that
    changes what the plant does on Monday.

  * The limits are served, not hidden: /health lists them, and every response
    that rests on missing plant knowledge says so in `ranking_note`.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chemai import __version__
from chemai.api.loop_schemas import (CascadePair, LoopDiagnosis, LoopHealthResponse, LoopRequest,
                                     PlantRequest, PlantResponse, PropagationSummary)
from chemai.config import ControlLoopConfig, DataQualityConfig, ReportConfig
from chemai.pipeline import analyse_loop, analyse_plant

log = logging.getLogger(__name__)

LIMITS = [
    "Stiction below the load noise is not detectable from OP and PV at all - "
    "about half of it is lost (VALIDATION.md 17.1). Log the valve position.",
    "Needs at least 10 oscillation cycles in the record, and at least 10 samples "
    "per cycle. A coarse historian turns stiction into 'tuning' - a wrong answer, "
    "not an uncertain one (21.2).",
    "Cascade pairs and propagation sources are CANDIDATES for an engineer to "
    "confirm, never verdicts (19.1, 20.3).",
    "Ranking needs plant location weights. Without them the response says so.",
    "Diagnoses stiction and valve saturation only. Other actuator faults have no "
    "labelled data to test against.",
]


def _to_diagnosis(result) -> LoopDiagnosis:
    row = result.as_row()
    return LoopDiagnosis(
        loop=result.loop, loop_type=result.loop_type, diagnosis=result.diagnosis,
        confidence=result.confidence, confidence_band=result.confidence_band,
        action=result.action, owner=result.owner, evidence=row["evidence"],
        period_s=None if result.period_s != result.period_s else round(result.period_s, 1),
        regularity=result.regularity,
        shape=None if result.shape != result.shape else result.shape,
        shape_signal=result.shape_signal, samples_per_cycle=result.samples_per_cycle,
        quality_flags=result.quality_flags, notes=result.notes)


def create_loop_app(cfg: ControlLoopConfig | None = None) -> FastAPI:
    cfg = cfg or ControlLoopConfig()
    dq, rcfg = DataQualityConfig(), ReportConfig()

    app = FastAPI(
        title="Control Loop Diagnosis",
        description=(
            "Diagnoses control loops from SP, PV and OP: sticking valve, tuning, "
            "saturation, frozen sensor, loop in manual.\n\n"
            "Every answer carries its evidence, a confidence band, an action and an "
            "owner. Loops recorded together are also checked for one shared "
            "oscillation - ten faults are often one fault and nine victims."
        ),
        version=__version__,
    )
    app.state.cfg = cfg

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """A record too short or inconsistent is a client error, not a server one."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=LoopHealthResponse, tags=["ops"])
    def health() -> LoopHealthResponse:
        return LoopHealthResponse(
            status="ok", version=__version__,
            diagnoses=["stiction", "tuning", "saturation", "frozen_sensor", "manual",
                       "undetermined"],
            limits=LIMITS)

    @app.post("/analyse/loop", response_model=LoopDiagnosis, tags=["diagnosis"])
    def analyse_one(req: LoopRequest) -> LoopDiagnosis:
        d = req.loop
        result = analyse_loop(d.sp, d.pv, d.op, d.name, d.loop_type, req.ts, cfg, dq, rcfg)
        log.info("%s -> %s (%s confidence)", d.name, result.diagnosis, result.confidence_band)
        return _to_diagnosis(result)

    @app.post("/analyse/plant", response_model=PlantResponse, tags=["diagnosis"])
    def analyse_many(req: PlantRequest) -> PlantResponse:
        names = [d.name for d in req.loops]
        if len(set(names)) != len(names):
            raise ValueError("loop names must be unique")
        loops = {d.name: {"sp": d.sp, "pv": d.pv, "op": d.op, "loop_type": d.loop_type}
                 for d in req.loops}
        out = analyse_plant(loops, req.ts, req.location_weights, None, cfg, dq, rcfg)
        report = out["report"]
        propagation = None
        if out["propagation"]:
            p = out["propagation"]
            propagation = PropagationSummary(
                period_s=round(p["period_s"], 1), members=p["members"],
                source_candidate=p["candidates"][0]["loop"], note=p["note"])
            log.info("propagation: %d loops at %.0f s, candidate %s",
                     len(p["members"]), p["period_s"], propagation.source_candidate)
        return PlantResponse(
            loops=[_to_diagnosis(r) for r in out["loops"]],
            ranking=report.top(rcfg.top_n).to_dict("records"),
            ranking_note=report.note,
            propagation=propagation,
            cascades=[CascadePair(**c) for c in (out["cascades"] or [])],
            simultaneous=out["simultaneous"])

    return app


loop_app = create_loop_app()
