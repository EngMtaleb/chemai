"""Configuration - one place for every constant, so nothing is buried in code."""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path comes from the environment first. When the package is INSTALLED
# (in a container, for example) __file__ points into site-packages and the
# source-tree default is wrong. Configuration belongs in the environment,
# not in the module's location on disk.
DATA_DIR = Path(os.getenv("CHEMAI_DATA_DIR", str(PROJECT_ROOT / "data")))


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


# ============================================================ Project 2
# Control loop performance. Every range below is a modelling choice and is
# reported with the results; see projects/p02_control_loops/VALIDATION.md.

@dataclass(frozen=True)
class ProcessRange:
    """Parameter ranges for one loop type. Each simulated loop draws once
    from these ranges, so every loop is a different plant - that is what
    makes a split by loop meaningful.

    Units: gain in %PV per %MV (self-regulating) or %PV/s per %MV
    (integrating); tau and theta in seconds.
    """
    integrating: bool
    gain: tuple[float, float]
    tau: tuple[float, float] | None     # None for an integrating process
    theta: tuple[float, float]
    ts: float                           # historian sampling interval, s
    substeps: int = 5                   # simulation steps per sample


@dataclass(frozen=True)
class ControlLoopConfig:
    """Settings for the control loop study (Project 2)."""

    # --- processes. F and P are fast and slow self-regulating; L integrates.
    flow: ProcessRange = ProcessRange(False, (0.8, 1.5), (1.0, 5.0), (0.5, 2.0), ts=1.0)
    pressure: ProcessRange = ProcessRange(False, (0.5, 2.0), (10.0, 60.0), (1.0, 5.0), ts=1.0)
    level: ProcessRange = ProcessRange(True, (0.005, 0.02), None, (2.0, 10.0), ts=2.0)

    n_samples: int = 3000
    warmup_samples: int = 300           # discarded: start-up transient

    # --- conditions. Tuning is defined PHYSICALLY, not by an arbitrary
    # multiplier: SIMC closed-loop time constant tau_c (Skogestad, 2003)
    # or a target gain margin.
    conditions: tuple[str, ...] = (
        "healthy", "stiction", "tuning_tight", "tuning_sluggish", "external_oscillation",
    )
    healthy_tauc_over_theta: tuple[float, float] = (1.0, 2.0)
    # ...but never so fast that measurement noise shakes the valve: the
    # proportional kick Kc * sigma_meas is capped at this OP standard
    # deviation (% of travel). Without the cap, SIMC on lag-dominant and
    # integrating loops gives Kc up to ~20 and noise dithers the valve
    # continuously - see VALIDATION.md, Week 1.
    op_noise_limit: float = 0.25
    # sluggish: closed loop slower than the open loop itself
    sluggish_tauc_over_open_loop: tuple[float, float] = (3.0, 8.0)   # x (tau + theta)
    sluggish_tauc_over_theta_integrating: tuple[float, float] = (15.0, 40.0)
    # tight: close enough to instability to OSCILLATE, as the SACAC tight
    # files do. Regular oscillation needs GM below ~1.1 (Week 1 sweep).
    tight_gain_margin: tuple[float, float] = (1.03, 1.12)

    # stiction (Choudhury et al., 2005): S = deadband + stickband, J = slip jump,
    # both in % of valve travel. J > 0 is required for a limit cycle in a
    # self-regulating loop with PI control.
    stiction_s: tuple[float, float] = (1.0, 5.0)
    stiction_j_over_s: tuple[float, float] = (0.2, 1.0)

    # external oscillation entering at the process input (an upstream loop)
    ext_period_over_loop: tuple[float, float] = (10.0, 40.0)   # x (tau_c + theta)
    ext_amplitude: tuple[float, float] = (2.0, 6.0)            # % at process input

    # noise
    load_noise_std: tuple[float, float] = (0.3, 1.0)           # % at process input
    # constant load bias: the valve must settle AWAY from where it started.
    # Without it the loop begins exactly at its balance point and a sticky
    # valve can sit there forever - an artefact, not physics.
    load_offset: tuple[float, float] = (1.0, 4.0)              # |%| at process input
    load_noise_corr_time: tuple[float, float] = (5.0, 30.0)    # s
    meas_noise_std: tuple[float, float] = (0.02, 0.15)         # % of PV range

    op_nominal: float = 50.0
    sp_nominal: float = 50.0

    # --- performance and oscillation indices
    harris_ar_order: int = 30
    regularity_threshold: float = 1.0   # Thornhill et al. (2003)

    random_state: int = 42

    def process(self, loop_type: str) -> ProcessRange:
        try:
            return {"F": self.flow, "P": self.pressure, "L": self.level}[loop_type]
        except KeyError:
            raise ValueError(f"loop type must be F, P or L, got '{loop_type}'") from None


# SACAC labels: mapping from the published file descriptions, not from any
# model. See projects/p02_control_loops/VALIDATION.md sections 1 and 4.
SACAC_LABELS: dict[str, str] = {
    "stiction": "stiction",
    "tuning-F-chemical-DB": "tuning_sluggish",     # description: "sluggish tuning"
    "tuning-L-paper-horch": "tuning_tight",        # description: "tight tuning"
    "tuning-Q-paper-horch": "tuning_tight",
    "other-F-paper-horch-2003-2": "healthy",       # normal operation, SP change
    "other-L-paper-horch-2003": "healthy",         # normal operation
    "other-F-paper-horch-2003": "external_oscillation",
}

# Data-quality rules: rows kept per file. Raw files are never edited.
# tuning-L-paper-horch rows >= 846 are a copy of stiction-L-paper-horch.
SACAC_KEEP_ROWS: dict[str, tuple[int, int]] = {
    "tuning-L-paper-horch-2003": (0, 846),
}

# Published process dead times, seconds (min, max). The only real loops
# for which the Harris index can be computed without guessing the delay.
SACAC_DEAD_TIME: dict[str, tuple[float, float]] = {
    "other-F-paper-horch-2003-2": (7.0, 8.0),
    "other-F-paper-horch-2003": (3.0, 8.0),
    "other-L-paper-horch-2003": (4.0, 4.0),
}
