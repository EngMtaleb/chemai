"""Request and response shapes.

Validation lives here, not in the endpoint. A request that reaches the
model has already been checked for type and physical plausibility.
"""
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """One estimate request.

    Bounds are deliberately wide - they reject physically impossible input
    (negative absolute temperature, zero pressure), not unusual input.
    Deciding whether the input is *within the validated envelope* is the
    model's job, not the schema's, and it is reported separately.
    """

    temperature: float = Field(
        ..., gt=0, le=2000,
        description="Absolute temperature at the measurement point (K)",
        examples=[450.0],
    )
    pressure: float = Field(
        ..., gt=0, le=10000,
        description="Column operating pressure, same units as training data",
        examples=[228.0],
    )

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_absolute(cls, v: float) -> float:
        # A Celsius value would silently produce a wrong 1000/T.
        if v < 100:
            raise ValueError(
                "temperature looks like Celsius; an absolute scale (K) is required"
            )
        return v


class PredictionResponse(BaseModel):
    """Three fields, not one.

    A bare number invites the caller to trust it unconditionally. The
    interval says how confident the estimate is, and in_envelope says
    whether the question was inside the range the model was validated on.
    """

    vapour_pressure: float = Field(..., description="Estimated vapour pressure")
    interval_lower: float
    interval_upper: float
    in_envelope: bool = Field(
        ..., description="False means the input is outside the validated range"
    )
    distance: float = Field(..., description="Mahalanobis distance from training centre")
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_fitted: bool
    version: str
    features: list[str]
    training_rows: int | None = None
    residual_sigma: float | None = None
