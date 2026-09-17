"""Cascade detection and source attribution.
The simulator is the ground truth: the fault is injected, so the answer is known."""
import numpy as np
import pandas as pd
import pytest

from chemai.features import attribute_source, find_cascades, simulate_cascade

rng = np.random.default_rng(0)
N = 2000


# ------------------------------------------------------------- detection

def _plant(n=N):
    """Three loops: 0 is a master whose OP drives loop 1's setpoint; 2 is alone."""
    master_op = 50 + 5 * np.sin(np.arange(n) / 40) + rng.normal(0, 0.1, n)
    setpoints = np.column_stack([np.full(n, 60.0), master_op, np.full(n, 30.0)])
    outputs = np.column_stack([master_op, 40 + rng.normal(0, 1, n), 20 + rng.normal(0, 1, n)])
    return setpoints, outputs, ["LC1", "FC1", "TC1"]


def test_finds_the_pair_and_nothing_else():
    pairs = find_cascades(*_plant())
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row.master, row.slave) == ("LC1", "FC1")
    assert row.correlation > 0.99 and row.scale == pytest.approx(1.0, abs=0.05)


def test_a_fixed_setpoint_has_no_master():
    sp, op, names = _plant()
    sp[:, 1] = 50.0                                   # the slave's SP stops moving
    assert len(find_cascades(sp, op, names)) == 0


def test_ratio_control_is_caught_by_the_scale_column():
    """Ratio control leaves the same correlation but scales the setpoint - the
    reason pairs stay candidates until an engineer confirms them."""
    sp, op, names = _plant()
    sp[:, 1] = 0.4 * op[:, 0] + 10                    # setpoint = 0.4 x another flow
    row = find_cascades(sp, op, names).iloc[0]
    assert row.correlation > 0.99 and row.scale == pytest.approx(0.4, abs=0.02)


def test_shape_and_name_checks():
    sp, op, names = _plant()
    with pytest.raises(ValueError):
        find_cascades(sp, op[:, :2], names)
    with pytest.raises(ValueError):
        find_cascades(sp, op, names[:2])


# ----------------------------------------------------------- attribution

@pytest.mark.parametrize("case,expected,owner", [
    ({}, "none", ""),
    (dict(stiction_s=3.0, slip_s=2.5), "slave", "maintenance"),
    (dict(kc_m=8.5, ti_m=15.0), "master", "control engineering"),
])
def test_attribution_finds_the_injected_fault(case, expected, owner):
    run = simulate_cascade(**case)
    verdict = attribute_source(run.sp_s, run.pv_s)
    assert verdict.source == expected and verdict.owner == owner


def test_a_master_fault_makes_both_signals_share_one_oscillation():
    run = simulate_cascade(kc_m=8.5, ti_m=15.0)
    v = attribute_source(run.sp_s, run.pv_s)
    assert v.slave_setpoint == pytest.approx(v.slave_error, rel=0.05)


def test_a_slave_fault_leaves_the_setpoint_irregular():
    run = simulate_cascade(stiction_s=3.0, slip_s=2.5)
    v = attribute_source(run.sp_s, run.pv_s)
    assert v.slave_error > 1 and v.slave_setpoint < 1


def test_length_check():
    with pytest.raises(ValueError):
        attribute_source(np.zeros(100), np.zeros(99))


# ------------------------------------------------------------ simulator

def test_the_masters_output_is_the_slaves_setpoint():
    run = simulate_cascade(T=400)
    np.testing.assert_allclose(run.sp_s, run.op_m)


def test_a_perfect_valve_keeps_the_slave_on_setpoint():
    run = simulate_cascade(T=2000, load_std=0.1)
    settled = run[run.t > 500]
    assert (settled.sp_s - settled.pv_s).abs().mean() < 0.5


def test_simulation_is_reproducible():
    a, b = simulate_cascade(T=300, seed=5), simulate_cascade(T=300, seed=5)
    pd.testing.assert_frame_equal(a, b)
