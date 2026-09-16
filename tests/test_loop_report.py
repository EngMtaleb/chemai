"""The weekly report - ranking, actions, owners, continuity.
Each test encodes a rule an engineer would notice if it broke."""
import numpy as np
import pandas as pd
import pytest

from chemai.config import ControlLoopConfig, ReportConfig
from chemai.evaluation import (REPORT_COLUMNS, action_for, build_report, confidence,
                               confidence_band, diagnose)

cfg, loop_cfg = ReportConfig(), ControlLoopConfig()


def loops(**over):
    base = dict(loop=["A", "B", "C"], regularity=[4.8, 2.5, 0.4],
                shape=[0.75, 0.30, 0.65], period_min=[110.0, 8.0, 5.0])
    base.update(over)
    return pd.DataFrame(base)


# ------------------------------------------------------------- diagnosis

def test_baseline_diagnosis():
    assert diagnose(4.8, 0.75, cfg=loop_cfg) == "stiction"
    assert diagnose(4.8, 0.30, cfg=loop_cfg) == "tuning"
    assert diagnose(0.4, 0.90, cfg=loop_cfg) == "undetermined"      # no regular oscillation
    assert diagnose(4.8, np.nan, cfg=loop_cfg) == "undetermined"    # too few cycles


@pytest.mark.parametrize("flag,expected", [
    ("frozen_sensor", "frozen_sensor"), ("op_constant", "manual"),
    ("op_inactive", "manual"), ("saturated", "saturation")])
def test_data_quality_flags_outrank_the_shape(flag, expected):
    """A stuck transmitter is not a control diagnosis; calling it stiction
    would send maintenance after the wrong equipment."""
    assert diagnose(4.8, 0.9, flags=[flag], cfg=loop_cfg) == expected


# ------------------------------------------------------------ confidence

def test_confidence_scales_and_saturates():
    assert confidence(0.0, cfg) == 0.0
    assert confidence(1.5, cfg) == pytest.approx(0.5)
    assert confidence(9.0, cfg) == 1.0                              # clipped
    assert confidence(np.nan, cfg) == 0.0
    assert [confidence_band(v, cfg) for v in (0.9, 0.5, 0.1)] == ["high", "medium", "low"]


def test_low_confidence_lowers_priority_but_never_hides_a_loop():
    weak = loops(regularity=[4.8, 1.2, 0.4])
    table = build_report(weak, {"A": 1, "B": 3, "C": 1}, cfg=cfg).table
    assert set(table.loop) == {"A", "B", "C"}                       # all present
    assert table.iloc[0].loop == "A"                                # confident one first


# -------------------------------------------------------- action and owner

def test_action_and_owner_follow_the_physics():
    assert action_for("stiction", "high")[1] == "maintenance"
    assert action_for("tuning", "high")[1] == "control engineering"
    assert "NOT retune" in action_for("external_oscillation", "high")[0]
    assert action_for("saturation", "high")[1] == "process engineering"
    assert action_for("frozen_sensor", "low")[1] == "instrumentation"


def test_weak_stiction_asks_for_a_field_check_first():
    strong, _ = action_for("stiction", "high")
    weak, _ = action_for("stiction", "medium")
    assert "Field check before raising a work order" in weak and weak != strong


# ---------------------------------------------------------------- ranking

def test_location_weight_changes_the_order():
    """Same data, different plant knowledge: a tuning problem on a
    specification loop can outrank stiction on a utility."""
    df = loops(regularity=[2.0, 2.9, 0.4], shape=[0.75, 0.30, 0.65])
    flat = build_report(df, cfg=cfg).table
    weighted = build_report(df, {"A": 1, "B": 3}, cfg=cfg).table
    assert flat.iloc[0].loop == "A"
    assert weighted.iloc[0].loop == "B"


def test_every_factor_is_visible():
    table = build_report(loops(), {"A": 2}, cfg=cfg).table
    for column in ("severity", "location", "fault_weight", "confidence", "priority"):
        assert column in table
    assert set(REPORT_COLUMNS) <= set(table.columns)


def test_missing_location_table_is_declared_not_invented():
    report = build_report(loops(), cfg=cfg)
    assert report.ranked_by_severity_only
    assert "WITHOUT plant location weights" in report.note
    assert (report.table.location == cfg.default_location_weight).all()


def test_constant_severity_is_excluded_and_declared():
    """Published plant data is normalised to unit variance (Eastman), so
    severity carries no information and must not silently multiply by 1."""
    report = build_report(loops(severity=[1.0, 1.0, 1.0]), {"A": 2}, cfg=cfg)
    assert "Severity unavailable" in report.note


def test_severity_is_used_when_it_varies():
    df = loops(regularity=[2.0, 2.0, 2.0], shape=[0.75, 0.75, 0.75], severity=[1.0, 9.0, 1.0])
    assert build_report(df, cfg=cfg).table.iloc[0].loop == "B"


# ------------------------------------------------------------- continuity

def test_a_new_loop_is_watched_and_a_returning_one_gets_a_work_order():
    week1 = build_report(loops(), {"A": 2}, cfg=cfg).table
    assert week1.iloc[0].status == "new" and "Watch" in week1.iloc[0].action
    week2 = build_report(loops(), {"A": 2}, previous=week1, cfg=cfg).table
    assert week2.iloc[0].status == "confirmed" and "Field-test" in week2.iloc[0].action


def test_a_frozen_sensor_is_never_only_watched():
    """Waiting a week while the integral drives the process away is not an option."""
    df = loops(); df["flags"] = [["frozen_sensor"], [], []]
    row = build_report(df, cfg=cfg).table.query("loop == 'A'").iloc[0]
    assert row.status == "new" and "URGENT" in row.action
