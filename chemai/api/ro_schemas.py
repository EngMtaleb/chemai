"""Request and response shapes for the RO fouling service (Project 3).

Two rules carried from the earlier projects:

  * A record that cannot carry a forecast is answered with "insufficient", not
    guessed - and a request that is malformed (mismatched lengths, negative
    days, DP_norm given in percent) is refused with 422 before it reaches the
    model.
  * Every forecast carries its interval, its extrapolation flag and its
    warnings. There is no field that returns a bare date.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_POINTS = 2000
MAX_TRAINS = 20
MAX_RECORDS = 5000


class CycleData(BaseModel):
    """The current cleaning cycle of one train."""

    name: str = Field("Train A", max_length=80)
    day: list[float] = Field(..., description="Days since the last cleaning ended, one value per day")
    dp_norm: list[float] = Field(..., description="Normalised pressure drop over its start-up value "
                                                  "(1.15 = 15 % above start-up)")
    censored: list[bool] | None = Field(None, description="True where the value is only known to be "
                                                          "at or below the start-up baseline")

    @model_validator(mode="after")
    def consistent(self):
        n = len(self.day)
        if len(self.dp_norm) != n or (self.censored is not None and len(self.censored) != n):
            raise ValueError("day, dp_norm (and censored) must have the same length")
        if n > MAX_POINTS:
            raise ValueError(f"at most {MAX_POINTS} days per cycle")
        if any(d < 0 for d in self.day):
            raise ValueError("days are counted from the end of the last cleaning: no negative days")
        if len(set(self.day)) != n:
            raise ValueError("one value per day: a day is repeated")
        if any(not (0.3 < x < 5.0) for x in self.dp_norm):
            raise ValueError("dp_norm is a ratio to start-up (about 1.0-2.0). A value like 15 or "
                             "115 is a percentage: send 1.15")
        return self


class ForecastRequest(BaseModel):
    cycle: CycleData
    review_level: float = Field(1.15, gt=1.0, lt=3.0,
                                description="DP_norm at which an engineer reviews the train for a "
                                            "cleaning. 1.15 is the vendor rule of +15 % since start-up")
    law: Literal["linear", "blockage"] | None = Field(
        None, description="Override the shipped law (linear). 'blockage' fits DP_norm^(-1/n), "
                          "straight when the foulant layer grows steadily")


class ForecastResponse(BaseModel):
    train: str
    status: str = Field(..., description="at_or_above | crossing | not_in_sight | falling | insufficient")
    law: str
    review_level: float
    level_now: float | None
    rate_per_week: float | None = Field(None, description="DP_norm units per week (0.01 = 1 %/week)")
    rate_lo: float | None
    rate_hi: float | None
    days_to_level: float | None = Field(None, description="P50, days from the last observation")
    days_lo: float | None = Field(None, description="P10")
    days_hi: float | None = Field(None, description="P90; null when beyond the 180-day horizon")
    extrapolation_ratio: float | None
    in_envelope: bool
    n_points: int
    span_days: float
    warnings: list[str]
    fit_window: list[float] = Field(..., description="First and last day used by the fit")


class TrainData(CycleData):
    dp_vessel_bar: float | None = Field(None, ge=0, le=20, description="Raw dP across the worst "
                                                                       "vessel now, bar")
    dp_element_bar: float | None = Field(None, ge=0, le=10, description="Raw dP across the lead "
                                                                        "element now, bar")
    pre_cleaning: float | None = Field(None, gt=0.3, lt=5, description="DP_norm just before the last cleaning")
    post_cleaning: float | None = Field(None, gt=0.3, lt=5, description="DP_norm just after it")
    previous_post_cleaning: float | None = Field(None, gt=0.3, lt=5,
                                                 description="DP_norm just after the cleaning before")


class PlanRequest(BaseModel):
    trains: list[TrainData] = Field(..., min_length=1, max_length=MAX_TRAINS)
    review_level: float = Field(1.15, gt=1.0, lt=3.0)
    trains_total: int | None = Field(None, ge=1, le=MAX_TRAINS,
                                     description="Trains installed, if more than those sent")

    @model_validator(mode="after")
    def unique_names(self):
        names = [t.name for t in self.trains]
        if len(set(names)) != len(names):
            raise ValueError("train names must be unique")
        return self


class PlanRowModel(BaseModel):
    train: str
    action: str
    category: str
    owner: str
    week: int | None
    level_now: float | None
    rate_per_week: float | None
    days_p10: float | None
    days_p50: float | None
    days_p90: float | None
    confidence: str
    evidence: str
    flags: list[str]


class PlanResponse(BaseModel):
    review_level: float
    rows: list[PlanRowModel]
    notes: list[str]


class RawRecord(BaseModel):
    """One day (or hour) of operation, as logged. Flows m3/h, pressures bar, TDS g/L."""
    day: float
    feed_flow: float = Field(..., gt=0)
    conc_flow: float = Field(..., gt=0)
    perm_flow: float = Field(..., gt=0)
    feed_pressure: float = Field(..., gt=0, lt=120)
    conc_pressure: float = Field(..., gt=0, lt=120)
    perm_pressure: float = Field(..., ge=0, lt=20)
    temperature_c: float = Field(..., ge=-5, le=60, description="Degrees Celsius - not kelvin")
    feed_tds: float = Field(..., gt=0, lt=100)
    perm_tds: float = Field(..., ge=0, lt=10)
    dp: float | None = Field(None, gt=0, lt=20, description="dP from a differential transmitter, "
                                                            "if fitted - better than two pressures")


class NormaliseRequest(BaseModel):
    records: list[RawRecord] = Field(..., min_length=14, max_length=MAX_RECORDS)
    baseline_rows: int = Field(14, ge=3, le=200, description="Rows at start-up that define 1.00")
    flow_exponent: float | Literal["fit"] = Field(
        "fit", description="'fit' measures the plant's own dP flow exponent from day-to-day flow "
                           "changes (falls back to 1.75); or give a number")

    @field_validator("flow_exponent")
    @classmethod
    def sane_exponent(cls, v):
        if isinstance(v, (int, float)) and not 0.8 <= v <= 2.2:
            raise ValueError("flow_exponent must be between 0.8 and 2.2")
        return v


class NormalisedRow(BaseModel):
    day: float
    dp_norm: float
    npf_ratio: float
    nsp_ratio: float
    ndp: float


class NormaliseResponse(BaseModel):
    flow_exponent: float
    viscosity_exponent: float
    exponent_fit: dict
    rows: list[NormalisedRow]
    notes: list[str]


class ROHealthResponse(BaseModel):
    status: str
    version: str
    law: str
    review_level_default: float
    interval: str
    limits: list[str]
    data: str
