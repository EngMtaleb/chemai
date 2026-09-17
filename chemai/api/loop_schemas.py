"""Request and response shapes for the loop-diagnosis service.

Validation lives here. Two rules are worth stating:

  * A record is rejected if it is too short to judge rather than analysed and
    reported with false confidence - "insufficient data" is a real answer
    (VALIDATION.md 15.1).
  * Every response carries the evidence and the confidence band, never a bare
    verdict. An engineer who cannot see why a loop was called stiction will not
    act on it, and should not.
"""
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_SAMPLES = 50_000
MAX_LOOPS = 100


class LoopData(BaseModel):
    """One loop's record: setpoint, measurement and controller output."""

    name: str = Field("loop", max_length=120, description="Tag name, e.g. FC101")
    loop_type: str = Field(
        "F", description="F flow · P pressure · L level · T temperature · Q quality. "
                         "Selects which signal carries the stiction fingerprint "
                         "(OP for fast loops, PV for level)")
    sp: list[float] = Field(..., description="Setpoint samples")
    pv: list[float] = Field(..., description="Measurement samples")
    op: list[float] = Field(..., description="Controller output samples")

    @field_validator("loop_type")
    @classmethod
    def known_loop_type(cls, v: str) -> str:
        v = v.strip().upper()[:1]
        if v not in set("FPLTQ"):
            raise ValueError("loop_type must be one of F, P, L, T, Q")
        return v

    @model_validator(mode="after")
    def same_length_and_sane(self):
        n = len(self.pv)
        if not (len(self.sp) == len(self.op) == n):
            raise ValueError("sp, pv and op must have the same number of samples")
        if n < 10:
            raise ValueError("a record of fewer than 10 samples cannot be analysed")
        if n > MAX_SAMPLES:
            raise ValueError(f"at most {MAX_SAMPLES} samples per loop")
        return self


class LoopRequest(BaseModel):
    loop: LoopData
    ts: float = Field(1.0, gt=0, le=3600, description="Logging interval, seconds")


class PlantRequest(BaseModel):
    """Several loops from one unit. Loops of equal length are treated as
    recorded together, which is what allows propagation and cascade checks."""

    loops: list[LoopData] = Field(..., min_length=1, max_length=MAX_LOOPS)
    ts: float = Field(1.0, gt=0, le=3600, description="Logging interval, seconds")
    location_weights: dict[str, int] | None = Field(
        None, description="Tag -> 1 utility, 2 energy, 3 specification or safety. "
                          "Supplied by the plant; without it the ranking says so")

    @field_validator("location_weights")
    @classmethod
    def weights_in_range(cls, v):
        if v and any(w not in (1, 2, 3) for w in v.values()):
            raise ValueError("location weights must be 1, 2 or 3")
        return v


class LoopDiagnosis(BaseModel):
    loop: str
    loop_type: str
    diagnosis: str = Field(..., description="stiction · tuning · saturation · frozen_sensor · "
                                            "manual · undetermined")
    confidence: float
    confidence_band: str
    action: str
    owner: str
    evidence: str
    period_s: float | None = None
    regularity: float
    shape: float | None = None
    shape_signal: str
    samples_per_cycle: float
    quality_flags: list[str] = []
    notes: list[str] = []


class PropagationSummary(BaseModel):
    period_s: float
    members: list[str]
    source_candidate: str
    note: str


class CascadePair(BaseModel):
    master: str
    slave: str
    correlation: float
    scale: float


class PlantResponse(BaseModel):
    """The weekly report, with the plant-level structure that changes what it says."""

    loops: list[LoopDiagnosis]
    ranking: list[dict]
    ranking_note: str = Field(..., description="States when no location weights were supplied")
    propagation: PropagationSummary | None = None
    cascades: list[CascadePair] = []
    simultaneous: bool = Field(
        ..., description="False when the records differ in length, so plant-level "
                         "checks were skipped")


class LoopHealthResponse(BaseModel):
    status: str
    version: str
    diagnoses: list[str]
    limits: list[str]
