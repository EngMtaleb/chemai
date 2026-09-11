"""Data-quality checks - one test per trap, on signals whose answer is known.
The real-file cases (SACAC, ISDB) are reproduced by week2_data_quality.py."""
import numpy as np
import pytest

from chemai.config import ControlLoopConfig, DataQualityConfig
from chemai.data import sample_loop_spec, simulate_loop
from chemai.features import (assess, longest_flat_run, n_levels, pi_fit,
                             saturation_fraction, sp_segments)

rng = np.random.default_rng(0)
N = 2000


def _noisy(n=N, level=50.0, s=0.3):
    return level + s * rng.normal(size=n)


def test_longest_flat_run():
    assert longest_flat_run([1, 2, 3]) == (0, 0)
    assert longest_flat_run([5, 5, 5]) == (2, 0)
    assert longest_flat_run([1, 2, 2, 2, 2, 3, 3]) == (3, 1)


def test_clean_simulated_loop_raises_no_flag():
    spec = sample_loop_spec("F", "healthy", np.random.default_rng(1), ControlLoopConfig())
    d = simulate_loop(spec, 1500, seed=2)
    rep = assess(d.sp, d.pv, d.op)
    assert rep.flags == [] and rep.ok_for_harris and rep.pi_r2 > 0.5


def test_frozen_sensor_while_op_moves():
    pv = _noisy(); pv[1000:1400] = pv[999]                  # transmitter stuck
    op = 50 + np.cumsum(rng.normal(0, 0.1, N))              # controller keeps acting
    rep = assess(np.full(N, 50.0), pv, op)
    assert "frozen_sensor" in rep.flags and not rep.ok_for_harris


def test_flat_pv_with_flat_op_is_not_a_frozen_sensor():
    """Nothing moves: a stopped unit or manual, not a stuck transmitter."""
    pv = _noisy(); pv[1000:1400] = pv[999]
    op = 50 + np.cumsum(rng.normal(0, 0.1, N)); op[990:1500] = op[989]
    rep = assess(np.full(N, 50.0), pv, op)
    assert "frozen_sensor" not in rep.flags and "op_inactive" in rep.flags


def test_op_never_moving():
    rep = assess(np.full(N, 50.0), _noisy(), np.full(N, 40.0))
    assert "op_constant" in rep.flags and not rep.ok_for_harris and not rep.ok_for_op_shape


def test_coarse_op_skips_op_based_checks():
    """ISDB chem9: a stiction loop whose OP has 28 values - not manual, not saturated."""
    op = np.round(50 + 5 * np.sin(np.arange(N) / 50), 0)     # ~11 levels, long flat stretches
    rep = assess(np.full(N, 50.0), _noisy(), op)
    assert "op_coarse" in rep.flags
    assert not {"op_inactive", "saturated"} & set(rep.flags)


def test_saturation_is_a_stay_not_a_touch():
    op = 50 + 10 * np.sin(np.arange(N) / 40) + rng.normal(0, 0.2, N)
    assert saturation_fraction(op) < 0.01                    # peaks touch the max once
    op[500:900] = op.max() + 1                               # valve fully open for 400 samples
    assert saturation_fraction(op) == pytest.approx(0.2, abs=0.01)
    assert "saturated" in assess(np.full(N, 50.0), _noisy(), op).flags


def test_quantised_pv_spoils_pv_shape_only():
    pv = np.round(4.87 + 0.02 * rng.normal(size=N), 2)       # like quantisation-Q-horch
    op = 50 + np.cumsum(rng.normal(0, 0.1, N))
    rep = assess(np.full(N, 4.87), pv, op)
    assert n_levels(pv) < DataQualityConfig().pv_levels_min
    assert "quantised_pv" in rep.flags and not rep.ok_for_pv_shape and rep.ok_for_op_shape


def test_moving_setpoint_is_flagged():
    sp = 50 + np.cumsum(rng.normal(0, 0.05, N))              # changes every sample
    rep = assess(sp, sp + rng.normal(0, 0.2, N), 50 + rng.normal(0, 1, N))
    assert "moving_setpoint" in rep.flags and not rep.ok_for_harris


def test_setpoint_steps_give_the_longest_constant_segment():
    sp = np.r_[np.full(600, 70.0), np.full(300, 100.0), np.full(1100, 130.0)]
    n, frac, seg = sp_segments(sp)
    assert n == 2 and (seg.start, seg.stop) == (900, 2000)


def test_pi_fit_recovers_a_pi_law():
    sp, pv = np.full(N, 50.0), _noisy()
    e = sp - pv
    op = 50 + np.cumsum(np.r_[0, 1.5 * (np.diff(e) + 0.1 * e[1:])])
    r2, kc = pi_fit(sp, pv, op)
    assert r2 > 0.99 and kc == pytest.approx(1.5, rel=0.01)
    assert pi_fit(sp, pv, 50 + rng.normal(size=N))[0] < 0.5


def test_missing_op():
    rep = assess(np.full(N, 50.0), _noisy(), np.full(N, np.nan))
    assert "no_op" in rep.flags and rep.op_levels is None and not rep.ok_for_harris
