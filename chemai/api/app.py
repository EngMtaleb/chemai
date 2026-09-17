"""FastAPI service for the soft sensor.

Design notes worth reading before changing anything:

  * The model is fitted ONCE at startup, not per request. Fitting takes
    seconds (500 bootstrap resamples); doing it per call would make the
    service unusable.

  * Every response carries the interval and the envelope flag. There is
    deliberately no endpoint that returns a bare number - a caller who
    only sees a value will trust it unconditionally.

  * /health reports whether the model is fitted. An orchestrator uses this
    to decide whether the container is ready to receive traffic.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import pickle

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chemai import __version__
from chemai.config import SoftSensorConfig
from chemai.data import load_distillation_tower, add_periods, split_by_period
from chemai.models import SoftSensor
from chemai.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from chemai.api.loop_app import create_loop_app

log = logging.getLogger(__name__)

# Module-level state, populated at startup.
_state: dict = {"model": None, "rows": None}


MODEL_PATH = Path(os.getenv("CHEMAI_MODEL_PATH", "models/soft_sensor.pkl"))


def _fit_model(cfg: SoftSensorConfig) -> tuple[SoftSensor, int]:
    """Load a serialised model if one exists; otherwise fit from data.

    Deployments ship the fitted model, not the data. The model is a set of
    coefficients and a covariance matrix - it is not the raw records, and it
    is what a served endpoint actually needs. Fitting at startup remains the
    path for local development, where the data is present.
    """
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        log.info("loaded serialised model from %s", MODEL_PATH)
        return model, -1

    df = add_periods(load_distillation_tower(cfg=cfg), cfg)
    train, _ = split_by_period(df, cfg)
    return SoftSensor(cfg).fit(train), len(train)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fit at startup, so the first request is not slow.

    If the data is missing the service still starts, but /health reports
    model_fitted=false and /predict returns 503. Starting degraded and
    saying so is better than crashing on boot with no explanation.
    """
    cfg: SoftSensorConfig = app.state.cfg
    try:
        model, n = _fit_model(cfg)
        _state["model"], _state["rows"] = model, n
        log.info("model fitted on %d rows at startup", n)
    except FileNotFoundError as exc:
        log.error("model NOT fitted: %s", exc)
    yield
    _state["model"] = None


def create_app(cfg: SoftSensorConfig | None = None) -> FastAPI:
    cfg = cfg or SoftSensorConfig()

    app = FastAPI(
        title="Distillation Soft Sensor",
        description=(
            "Estimates laboratory vapour pressure from two process measurements.\n\n"
            "Every response includes a confidence interval and an operating-envelope "
            "flag. When `in_envelope` is false the estimate is extrapolation and "
            "must not be relied on."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Physical validation failures are client errors, not server errors."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        m = _state["model"]
        return HealthResponse(
            status="ok" if m else "degraded",
            model_fitted=m is not None,
            version=__version__,
            features=[cfg.temperature_col, cfg.pressure_col],
            training_rows=_state["rows"],
            residual_sigma=round(m.residual_sigma_, 3) if m else None,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(req: PredictionRequest) -> PredictionResponse:
        m = _state["model"]
        if m is None:
            raise HTTPException(
                status_code=503,
                detail="model not fitted - training data unavailable at startup",
            )
        p = m.predict_one(req.temperature, req.pressure)
        if not p.in_envelope:
            log.warning(
                "extrapolation requested: T=%.1f P=%.1f distance=%.2f",
                req.temperature, req.pressure, p.distance,
            )
        return PredictionResponse(
            vapour_pressure=p.value,
            interval_lower=p.lower,
            interval_upper=p.upper,
            in_envelope=p.in_envelope,
            distance=p.distance,
            warning=p.warning,
        )

    # Project 2 rides along on the same deployment, under /loops. It holds no
    # model and no state, so it costs nothing at startup and cannot make this
    # service fail to boot.
    app.mount("/loops", create_loop_app())

    return app


app = create_app()