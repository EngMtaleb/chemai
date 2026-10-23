"""Request and response models for the when-to-stop service (Project 5)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BatchRequest(BaseModel):
    """One batch, from start-up to wherever the log ends."""
    t: list[float] = Field(min_length=30, max_length=200_000, description="Seconds since start-up, increasing")
    power_kw: list[float] = Field(description="Total pump power, kW")
    permeate_lpm: list[float] = Field(description="Permeate flow, L/min")
    conductivity: list[float] | None = Field(default=None, description="Reject conductivity, mS/cm (optional, for reporting)")
    drain_kwh: float = Field(default=0.034, ge=0, description="Energy of draining one batch's brine (the last drain)")
    brine_l: float = Field(default=16.0, ge=0, description="Brine drained per batch, L (the last drain)")
    b: float = Field(default=1.0, ge=0, le=1000, description="Price of brine in kWh per m3 of brine "
                                                              "(brine $/m3 divided by electricity $/kWh)")
    window_s: float = Field(default=120.0, ge=10, le=3600, description="Trailing median of the live cost")
    min_minutes: float = Field(default=5.0, ge=0, le=600, description="No stop before this")

    @model_validator(mode="after")
    def same_length_and_finite(self):
        n = len(self.t)
        cols = {"power_kw": self.power_kw, "permeate_lpm": self.permeate_lpm}
        if self.conductivity is not None:
            cols["conductivity"] = self.conductivity
        for name, v in cols.items():
            if len(v) != n:
                raise ValueError(f"{name} has {len(v)} values, t has {n}")
        for name, v in {"t": self.t, **cols}.items():
            if any(x != x or abs(x) == float("inf") for x in v):
                raise ValueError(f"{name} holds values that are not finite numbers")
        if any(b <= a for a, b in zip(self.t, self.t[1:])):
            raise ValueError("t must increase strictly: one row per time")
        if any(p < 0 for p in self.power_kw):
            raise ValueError("power_kw cannot be negative")
        return self


class Point(BaseModel):
    minutes: float
    permeate_l: float
    conductivity: float | None = None
    live_kwh_m3: float | None = None
    average_kwh_m3: float


class SquareRootLaw(BaseModel):
    fitted_minutes: tuple[float, float]
    m0_kwh_m3: float
    slope_kwh_m3_per_m3: float
    fixed_kwh: float
    v_star_l: float
    forecast_minutes: float | None


class BatchResponse(BaseModel):
    status: Literal["stop", "run_longer", "insufficient"]
    verdict: str
    stop: Point | None = None
    best_in_hindsight: Point | None = None
    end: Point | None = None
    regret_pct: float | None = Field(default=None, description="The rule's stop against the best end in hindsight")
    end_regret_pct: float | None = Field(default=None, description="The log's own end against the best end in hindsight")
    fixed_kwh: float
    law: SquareRootLaw | None = None
    curve: dict = Field(default_factory=dict, description="Up to 400 points for plotting")
    notes: list[str] = []


class IntervalRequest(BaseModel):
    """A slow cycle (days between cleanings): the live cost of running fouled, day by day."""
    day: list[float] = Field(min_length=5, max_length=20_000, description="Days since the last cleaning")
    live_cost: list[float] = Field(description="Extra cost of the next m3, in money per m3 (or kWh/m3)")
    fixed_cost: float = Field(gt=0, description="Cost of a cleaning per m3/day of capacity, same money unit")
    smooth_days: float = Field(default=7.0, ge=1, le=60)
    min_days: float = Field(default=7.0, ge=0, le=365)

    @model_validator(mode="after")
    def same_length(self):
        if len(self.day) != len(self.live_cost):
            raise ValueError(f"live_cost has {len(self.live_cost)} values, day has {len(self.day)}")
        if any(x != x or abs(x) == float("inf") for x in self.day + self.live_cost):
            raise ValueError("day and live_cost must be finite numbers")
        if len(set(self.day)) != len(self.day):
            raise ValueError("day repeats: one value per day")
        return self


class IntervalResponse(BaseModel):
    verdict: str
    length_days: float
    slope_per_day: float | None
    t_star_days: float | None = Field(description="Square-root law: sqrt(2 F / slope); null if the cost had no rising trend")
    rule_day: float | None = Field(description="First day the smoothed live cost reached the average cost")
    implied_fixed_cost: float | None = Field(description="What a cleaning would have to cost for this interval to be the best")
    average_cost_at_end: float
    curve: dict = Field(default_factory=dict)


class LawResponse(BaseModel):
    fixed: float
    slope: float
    m0: float
    best_length: float | None
    best_average_cost: float | None
    note: str


class StopHealthResponse(BaseModel):
    status: str
    version: str
    results: dict
    limits: list[str]
    sources: dict
