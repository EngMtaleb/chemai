"""Configuration - one place for every constant, so nothing is buried in code."""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class SoftSensorConfig:
    """Settings for the distillation soft sensor.

    Frozen so it cannot be mutated accidentally halfway through a run -
    a silent config change between training and evaluation is a classic
    source of results that cannot be reproduced.
    """

    data_file: str = "distillation-tower.csv"
    target: str = "VapourPressure"

    # Operating campaigns. Boundaries come from the gaps in the record;
    # see docs/EDA.md section 3. Never split this data randomly.
    period_breaks: tuple[int, ...] = (163, 186)
    train_period: int = 0
    test_period: int = 2

    # Features. The inverse-temperature form is required by Antoine's
    # equation; see docs/EDA.md section 8. Raw temperature scores 0.95,
    # this form scores 0.98.
    temperature_col: str = "Temp9"
    pressure_col: str = "PressureC1"

    # Reference temperature for vapour pressure, 37.8 C = 100 F.
    t_reference_k: float = 311.0

    # Model
    ridge_alpha: float = 1.0
    random_state: int = 42

    # Uncertainty
    n_bootstrap: int = 500
    confidence: float = 0.95

    # Operating envelope. Mahalanobis distance, chi-squared threshold.
    envelope_percentile: float = 0.99

    # Columns that are exact transformations of other columns.
    # Keeping both forms corrupts feature attribution; see docs/EDA.md section 2.
    derived_columns: tuple[str, ...] = (
        "InvTemp1", "InvTemp2", "InvTemp3", "InvPressure1",
    )
