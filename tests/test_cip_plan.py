"""The weekly CIP plan: rules applied in a fixed order, so anyone can check why a
train landed where it did."""
import numpy as np
import pytest

from chemai.config import ROConfig
from chemai.evaluation.cip_plan import TrainRecord, build_plan, cleaning_recovery, plan_train

cfg = ROConfig()


def _cycle(rate, start, days=42, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days + 1, dtype=float)
    return list(t), list(start + rate * t + rng.normal(0, noise, len(t)))


def test_the_manufacturer_limit_overrides_a_falling_trend():
    d, x = _cycle(-0.001, 1.10)
    row = plan_train(TrainRecord("A", d, x, dp_vessel_bar=3.6), cfg)
    assert row.category == "limit" and row.week == 1 and row.owner == "operations"


def test_at_the_review_level_means_decide_this_week():
    d, x = _cycle(0.001, 1.14)
    row = plan_train(TrainRecord("A", d, x), cfg)
    assert row.category == "review" and row.week == 1


def test_a_crossing_inside_four_weeks_is_planned():
    d, x = _cycle(0.0015, 1.06)                     # 1.123 now, 1.15 in about 18 days
    row = plan_train(TrainRecord("A", d, x), cfg)
    assert row.category == "plan" and row.owner == "operations planning"
    assert 1 <= row.week <= 4


def test_a_falling_train_gets_no_cleaning():
    d, x = _cycle(-0.0012, 1.10)
    row = plan_train(TrainRecord("A", d, x), cfg)
    assert row.category == "none" and "falling" in row.action


def test_too_soon_after_the_cleaning_waits():
    d, x = _cycle(0.002, 1.05, days=6)
    row = plan_train(TrainRecord("A", d, x), cfg)
    assert row.category == "wait"


def test_a_cleaning_that_barely_moved_dp_is_a_procedure_problem():
    assert cleaning_recovery(1.30, 1.27, 1.05) == pytest.approx(0.12)
    d, x = _cycle(0.0005, 1.08)
    row = plan_train(TrainRecord("A", d, x, pre_cleaning=1.30, post_cleaning=1.27,
                                 previous_post_cleaning=1.05), cfg)
    assert any("CIP procedure" in f for f in row.flags)


def test_no_judgement_when_there_was_nothing_to_recover():
    assert cleaning_recovery(1.06, 1.05, 1.05) is None


def test_two_trains_are_not_cleaned_in_one_week_when_supply_needs_two():
    a = TrainRecord("A", *_cycle(0.001, 1.15))
    b = TrainRecord("B", *_cycle(0.001, 1.16))
    c = TrainRecord("C", *_cycle(-0.001, 1.08))
    plan = build_plan([a, b, c], cfg)
    weeks = [r.week for r in plan.rows if r.week is not None]
    assert sorted(weeks) == [1, 2]
    moved = [r for r in plan.rows if r.week == 2][0]
    assert any("online" in f for f in moved.flags)


def test_the_plan_puts_the_most_urgent_first():
    rows = build_plan([TrainRecord("C", *_cycle(-0.001, 1.08)),
                       TrainRecord("B", *_cycle(0.0015, 1.06)),
                       TrainRecord("A", *_cycle(-0.001, 1.10), dp_vessel_bar=3.5)], cfg).rows
    assert [r.category for r in rows] == ["limit", "plan", "none"]


def test_a_train_hovering_just_below_the_level_is_planned():
    """The margin trigger, adopted after the external test: on real data a flat
    train 0.02 below the level crossed it on day-to-day swings more often than any
    trend predicted (VALIDATION.md section 12)."""
    d, x = _cycle(0.0, 1.13, noise=0.004, seed=3)
    row = plan_train(TrainRecord("A", d, x), cfg)
    assert row.category == "plan" and "within 0.03" in row.action
