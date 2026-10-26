"""Request and response models for the sun-and-moon service (Project 4)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SeriesPoint(BaseModel):
    t: datetime = Field(description="Local standard time (no daylight saving) of the sample")
    value: float = Field(description="The plant signal: a fouling rate, a pressure, a turbidity...")
    run: str | None = Field(default=None, description="Optional run label; otherwise a gap over 6 h starts a new run")


class SeriesTestRequest(BaseModel):
    records: list[SeriesPoint] = Field(min_length=100, max_length=200_000)
    kind: Literal["tide", "clock"] = Field(
        default="tide", description="'tide': locked to the moon? 'clock': locked to the time of day?")
    n_surrogates: int = Field(default=499, ge=99, le=999)
    block_days: float = Field(default=14.0, ge=3.0, le=60.0,
                              description="A long unbroken record is cut into pieces this long for the null")

    @field_validator("records")
    @classmethod
    def finite(cls, v):
        bad = [p for p in v if p.value != p.value or abs(p.value) == float("inf")]
        if bad:
            raise ValueError(f"{len(bad)} values are not finite numbers")
        return v


class SeriesTestResponse(BaseModel):
    status: Literal["ok", "insufficient"]
    kind: str | None = None
    verdict: str | None = None
    p_value: float | None = None
    placebo_p_value: float | None = None
    partial_r2: float | None = None
    null_partial_r2_q95: float | None = None
    amplitudes: dict | None = None
    phases_deg: dict | None = None
    range_5_95: float | None = None
    range_null_q95: float | None = None
    median_y: float | None = None
    relative_range: float | None = None
    n: int
    n_blocks: int
    span_hours: float
    notes: list[str] = []


class DesignResponse(BaseModel):
    weather: str
    capacity: float
    storage_h: float
    pv_kwp: float
    tidal_kw: float
    renewable_fraction: float
    grid_kwh_m3: float
    curtailed_share: float
    run_hours_per_day: float
    rule_fraction: float | None
    per_1000_m3_day: dict


class SunHealthResponse(BaseModel):
    status: str
    version: str
    results: dict
    limits: list[str]
    data: str
