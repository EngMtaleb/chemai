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
    # Thornhill et al. (2003) call r > 1 a regular oscillation. Used as a
    # CONFIDENCE band in reports, never as a gate before diagnosis: on real
    # data the classes straddle it (VALIDATION.md 15). Below min_cycles the
    # index is not reported at all - the record is too short to judge.
    regularity_threshold: float = 1.0
    min_cycles: float = 10.0
    # shape index (He et al., 2007): >0.6 triangular, <0.4 sinusoidal. 0.6 on
    # 31 real oscillating loops catches 15 of 20 stiction cases with 3 false
    # alarms - the BASELINE any classifier must beat (VALIDATION.md 16).
    shape_triangular: float = 0.6
    shape_sinusoidal: float = 0.4
    shape_harmonics: int = 7

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


@dataclass(frozen=True)
class DataQualityConfig:
    """Thresholds for the data-quality checks that run before any diagnosis.

    Every value was set against 150 real loops (SACAC + ISDB): see
    projects/p02_control_loops/VALIDATION.md, section 14. Counts are in
    SAMPLES, not seconds - freezing and coarse recording happen sample by
    sample. 200 samples is 3 minutes at 1 s sampling but over an hour at 20 s.
    """
    pv_flat_run: int = 300          # above every flat run in an expert-labelled normal loop
                                    # (ISDB buildings.8, 'no oscillation', reaches 275)
    op_flat_run: int = 200
    op_levels_min: int = 50         # fewer distinct OP values = coarsely recorded OP
    pv_levels_min: int = 20         # fewer distinct PV values = quantised sensor
    saturation_fraction: float = 0.05
    saturation_min_run: int = 3     # a touch of the limit is not saturation; a stay is
    moving_sp_fraction: float = 0.5 # SP changing in most samples = likely cascade slave
    pi_r2_min: float = 0.5          # below: 'PI law not confirmed' - a note, not an exclusion
    min_segment: int = 150          # shortest regulatory segment worth analysing (5 x AR order)
    # Week 6: below ten samples per oscillation cycle the waveform shape is gone
    # and stiction comes back as 'tuning' - a wrong answer, not an uncertain one.
    min_samples_per_cycle: float = 10.0


@dataclass(frozen=True)
class ReportConfig:
    """Weekly loop report: how loops are ranked, and who is asked to act.

    priority = severity x location weight x fault weight x confidence

    Severity and confidence come from the data; the location weight comes from
    the PLANT (an engineer fills a table once); the fault weights below are
    declared constants. See projects/p02_control_loops/VALIDATION.md, 18.
    """

    # --- fault weights. Engineering judgement, NOT measured: no repair cost has
    # been recorded in any plant yet. The first site to log what fixes actually
    # cost should replace them. Stiction degrades, needs maintenance and possibly
    # a shutdown; bad tuning is an hour of a control engineer's time from the
    # control room. External oscillation is low NOT because it is harmless, but
    # because fixing that loop is the wrong action - the loop is a victim.
    fault_weights: tuple[tuple[str, float], ...] = (
        ("stiction", 1.0),
        ("saturation", 0.8),
        ("tuning", 0.6),
        ("external_oscillation", 0.4),
        ("undetermined", 0.2),
    )

    # --- location weights, written by the plant. 3 = sets specification or
    # safety, 2 = consumes energy, 1 = utility or storage. Three grades, because
    # an engineer must be able to fill the table in one sitting; five invites an
    # argument that ends with no table at all.
    default_location_weight: int = 1
    location_grades: tuple[str, ...] = ("utility or storage", "energy", "specification or safety")

    # --- confidence, from the oscillation regularity (VALIDATION.md 15.3)
    confidence_full: float = 3.0        # regularity at which confidence reaches 1
    confidence_high: float = 0.7        # above: open a work order
    confidence_medium: float = 0.4      # above: field check first

    top_n: int = 10
    consecutive_weeks_to_confirm: int = 2

    def fault_weight(self, diagnosis: str) -> float:
        return dict(self.fault_weights).get(diagnosis, 0.2)


# What the plant should do about each diagnosis, and who owns it. The reasoning
# is the physics of Week 1, not a lookup invented for the report.
ACTIONS: dict[str, tuple[str, str]] = {
    "stiction": ("Field-test the valve: step it in manual and watch the stem",
                 "maintenance"),
    "saturation": ("Valve is undersized for the load, or the load has changed",
                   "process engineering"),
    "tuning": ("Retune from the control room - no shutdown needed",
               "control engineering"),
    "external_oscillation": ("Do NOT retune this loop: find the source first",
                             "control engineering"),
    "frozen_sensor": ("URGENT: transmitter stuck while the controller kept acting",
                      "instrumentation"),
    "manual": ("Loop is not in automatic - confirm with operations",
               "operations"),
    "undetermined": ("Not enough evidence: longer record, or check the loop by hand",
                     "control engineering"),
}


# ============================================================ Project 3
# RO membrane fouling and cleaning (CIP). Every value below is either physics
# with a stated source, a vendor convention named as such, or a modelling choice
# reported with the results; see projects/p03_ro_fouling/VALIDATION.md.

@dataclass(frozen=True)
class RODepositConfig:
    """The public deposit used as the external test set.

    Data CC-BY-4.0: cite the article and the deposit wherever a number from it
    is shown. The plant owner is not named by the authors and is not named here.
    Not committed to the repository: `chemai.data.ro_deposit.download_deposit`
    fetches it and checks the published md5.
    """
    record: str = "21055948"                      # Zenodo record, deposit v1.3
    concept_doi: str = "10.5281/zenodo.21003996"
    zip_name: str = "ESWRT_reproducibility_deposit_v1.3-20260630.zip"
    md5: str = "68b0223978f0e674d51cfc4c24e95435"  # as published on the record page
    daily_file: str = "derived_daily_indicator_table.csv"
    ledger_file: str = "cip_campaign_ledger.csv"
    folder: str = "ro_deposit"
    # Cleaning legs closer than this are one campaign (acid + alkaline stages).
    # The deposit's own ledger uses the same rule and lists 18 campaigns.
    campaign_gap_days: int = 7
    # 10-min logging: a day cannot hold more than 144 samples. More means
    # duplicated or re-gridded records - flagged, never repaired.
    max_samples_per_day: int = 144
    citation: str = (
        "Su, Y.H., Meng, F.C. and Chang, P.Y. (2026) Auditable clean-in-place "
        "decision support from routine SWRO SCADA. Membranes 16(8), 257. "
        "doi:10.3390/membranes16080257 - data: doi:10.5281/zenodo.21055948 (CC-BY-4.0)")

    @property
    def url(self) -> str:
        return f"https://zenodo.org/records/{self.record}/files/{self.zip_name}?download=1"


@dataclass(frozen=True)
class ROConfig:
    """Physics, forecast and planning settings for the RO project (Project 3)."""

    # --- reference temperature for every normalised quantity
    t_ref_c: float = 25.0

    # --- temperature correction of permeability, TCF = exp(E (1/298.15 - 1/T)).
    # Vendor convention, not a law: 2640 K for water is a common one; salt
    # passage rises roughly twice as fast with temperature (~6 %/K). Vendors
    # publish different constants - use the installed membrane's.
    tcf_e_water: float = 2640.0
    tcf_e_salt: float = 5000.0

    # --- feed-channel pressure drop: dP ~ Q_avg^(2-m) * mu^m * gap^-n.
    # With a friction factor f ~ Re^-m, the flow exponent is 2 - m and the
    # viscosity exponent is m. Schock & Miquel (1987) give m = 0.3, CFD on
    # spacer channels ~0.2 (Fluids 3, 12, 2018): the physics says 1.7-1.8.
    dp_flow_exponent: float = 1.75
    dp_visc_exponent: float = 0.25
    # Vendors normalise dP with a flow ratio raised to 1.0-1.5. Kept to show
    # what that costs when production changes (VALIDATION.md section 6).
    vendor_flow_exponent: float = 1.5
    # How dP grows as the open gap closes. Slit or Schock-Miquel scaling gives
    # 3; CFD on a spacer channel gives ~2.2 (10 % gap loss -> +26 % dP).
    gap_exponent: float = 2.2

    # --- manufacturer limits. Typical 8-inch values (1.0 bar per element,
    # 3.4 bar per vessel); replace with the installed element's datasheet.
    dp_max_element_bar: float = 1.0
    dp_max_vessel_bar: float = 3.4

    # --- forecast. Chosen on simulation (development plants only) and frozen
    # before the external test; see VALIDATION.md sections 7-10.
    review_level: float = 1.15          # DP_norm: vendor "+15 % since start-up"
    law: str = "linear"                 # 'linear' | 'blockage' - see VALIDATION.md 7 and 9
    fit_window_days: int = 42           # trailing window; whole cycle if shorter
    skip_after_cip_days: int = 3        # flushing transient right after a cleaning
    min_points: int = 10
    min_span_days: int = 14
    horizon_max_days: int = 180         # beyond this a crossing is "not in sight"
    extrapolation_ratio_max: float = 2.0  # days ahead / days observed; beyond: flagged
    smooth_days: int = 7                # trailing median used for "now"
    interval: float = 0.80              # P10-P90: a planning interval
    n_draws: int = 400                  # plausible lines drawn around the fit
    interval_inflation: float = 2.0     # calibrated on simulated development plants: 80 % coverage
    random_state: int = 42

    # --- weekly plan
    plan_weeks: int = 4                 # "plan a CIP" when the P50 date falls inside this many weeks...
    # ...or when the 7-day median is already within this margin of the review level.
    # Added AFTER the external test, where the margin rule beat the forecast; checked
    # on simulated development and test plants before adoption (VALIDATION.md 12).
    plan_margin: float = 0.03
    # A cleaning that removes less than this share of the rise since the
    # previous cleaning is called ineffective - the procedure, not the
    # schedule, needs attention.
    min_recovery_share: float = 0.25
    min_trains_online: int = 2          # never schedule two trains in the same week below this


@dataclass(frozen=True)
class ROSimConfig:
    """Ranges for the simulated RO plants (Project 3).

    Each plant draws once from these ranges; its trains share the plant's
    water and fouling law and differ a little, as trains on one intake do.
    The simulator exists to provide what no public data set has: the true
    fouling state, and how it would have continued had nobody cleaned.
    """
    days: int = 770
    trains: int = 3
    vessels: int = 3
    elements: int = 8
    element_area_m2: float = 37.2             # 8-inch, 400 ft2
    flux_lmh: tuple[float, float] = (10.0, 14.0)
    recovery: tuple[float, float] = (0.40, 0.45)
    feed_tds_g_l: tuple[float, float] = (34.0, 37.0)
    temp_mean_c: tuple[float, float] = (22.0, 28.0)
    temp_amp_c: tuple[float, float] = (2.0, 6.0)
    perm_pressure_bar: float = 1.0
    a_w_lmh_bar: tuple[float, float] = (0.9, 1.3)       # water permeability, 25 C
    b_s_lmh: tuple[float, float] = (0.02, 0.05)         # salt permeability, 25 C
    conc_polarisation: tuple[float, float] = (1.05, 1.15)
    dp_clean_bar: tuple[float, float] = (0.8, 1.6)      # vessel dP, design flow, 25 C
    friction_m: tuple[float, float] = (0.19, 0.30)      # f ~ Re^-m
    gap_exponent: tuple[float, float] = (2.0, 3.0)
    lead_share: tuple[float, float] = (0.15, 0.25)      # lead element's share of clean dP
    rest_fouling_ratio: tuple[float, float] = (0.05, 0.40)
    # fouling laws for the lead element's blocked gap fraction
    laws: tuple[str, ...] = ("linear", "biofilm", "asymptotic")
    linear_rate: tuple[float, float] = (0.0003, 0.0020)  # gap fraction per day at 25 C
    biofilm_k: tuple[float, float] = (0.008, 0.030)      # 1/day, exponential phase
    biofilm_seed: tuple[float, float] = (0.01, 0.04)
    biofilm_cap: tuple[float, float] = (0.45, 0.70)
    asym_level: tuple[float, float] = (0.05, 0.25)
    asym_tau_days: tuple[float, float] = (30.0, 120.0)
    q10: tuple[float, float] = (1.5, 2.5)               # rate x q10 per +10 K
    rate_noise: float = 0.25                            # day-to-day lognormal sigma
    upsets_per_year: float = 3.0                        # blooms, pretreatment upsets
    upset_size: tuple[float, float] = (0.01, 0.04)      # extra gap fraction per upset
    permeability_loss: tuple[float, float] = (0.1, 0.5) # A / (1 + a*phi)
    compaction: tuple[float, float] = (0.03, 0.12)
    # cleaning
    cip_efficiency: tuple[float, float] = (0.55, 0.95)  # share of the removable layer
    irreversible_share: tuple[float, float] = (0.02, 0.10)
    relax_share: tuple[float, float] = (0.0, 0.5)       # loosened, flushed out over days
    relax_days: tuple[float, float] = (5.0, 20.0)
    two_leg_probability: float = 0.4
    # the operator, whose decisions are endogenous to what the screen shows
    operator_level: tuple[float, float] = (1.15, 1.35)
    operator_delay_days: tuple[int, int] = (0, 14)
    max_days_without_cip: tuple[int, int] = (150, 260)
    min_days_between_cip: int = 18
    other_reason_cip_per_year: float = 1.0
    # availability: a train is on or off in blocks; a day is recorded only when
    # at least two trains run (the deposit's supply-mode filter)
    on_block_days: float = 12.0
    off_block_days: float = 6.0
    baseline_days: int = 14
    # measurement
    dp_noise_rel: tuple[float, float] = (0.008, 0.03)  # daily median, DP transmitter
    pt_noise_bar: float = 0.03                          # daily median, one pressure transmitter
    pt_drift_bar_per_day: float = 0.004                 # zero drift, random walk
    outlier_probability: float = 0.05
    outlier_scale: float = 4.0
    random_state: int = 42
