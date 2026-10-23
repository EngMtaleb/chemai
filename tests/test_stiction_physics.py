"""S read from CO and CV alone (chemai.features.stiction_physics, VALIDATION.md 23)."""
import dataclasses as dc

import numpy as np
import pytest

from chemai.config import ControlLoopConfig
from chemai.data import StictionValve, sample_loop_spec, simulate_loop
from chemai.features.stiction_physics import _valve_path_py, reversal_travel, turning_points


@pytest.fixture(scope="module")
def cfg():
    return ControlLoopConfig()


def _plant(cfg, loop_type="F", seed=1):
    return sample_loop_spec(loop_type, "stiction", np.random.default_rng(seed), cfg)


def _read(spec, cfg, s, j):
    d = simulate_loop(dc.replace(spec, stiction_s=s, stiction_j=j, load_std=0.3), 3000, seed=0, cfg=cfg)
    return reversal_travel(d.op.to_numpy(), d.pv.to_numpy())


# ------------------------------------------------------------- the reversals

def test_turning_points_find_every_reversal():
    t = np.arange(400)
    tri = np.abs((t % 100) - 50) / 25.0                 # triangle, swing 2, period 100
    assert turning_points(tri, 0.5) == [50, 100, 150, 200, 250, 300, 350]


def test_turning_points_ignore_wiggles_smaller_than_h():
    t = np.arange(400)
    noisy = np.abs((t % 100) - 50) / 25.0 + 0.1 * np.sin(1.3 * t)
    tp = turning_points(noisy, 0.5)
    assert len(tp) == 7                                 # no extra turns from the wiggles
    assert np.all(np.abs(np.array(tp) - np.arange(50, 400, 50)) <= 3)


# ------------------------------------------------------------- the reading of S

@pytest.mark.parametrize("loop_type", ["F", "P"])
def test_reading_recovers_s_of_a_sticky_valve(cfg, loop_type):
    r = _read(_plant(cfg, loop_type), cfg, 4.0, 3.0)
    assert r.reversals > 50
    assert r.s_co == pytest.approx(4.0, abs=1.0)        # % of travel, as CO is stored


@pytest.mark.parametrize("loop_type", ["F", "P"])
def test_reading_grows_with_s(cfg, loop_type):
    spec = _plant(cfg, loop_type)
    readings = [_read(spec, cfg, s, 0.75 * s).s_co for s in (2.0, 4.0, 6.0)]
    assert readings[0] < readings[1] < readings[2]


def test_a_healthy_valve_reads_far_less(cfg):
    spec = _plant(cfg)
    sticky, healthy = _read(spec, cfg, 4.0, 3.0), _read(spec, cfg, 0.0, 0.0)
    # a healthy loop still shows some travel - the process takes time to answer -
    # which is why the reading alone is a weak diagnosis
    assert healthy.s_co < 1.0
    assert healthy.s_co < 0.25 * sticky.s_co


def test_flat_or_short_records_give_no_reading():
    rng = np.random.default_rng(0)
    flat = reversal_travel(np.ones(500), rng.normal(size=500))        # CO never moves
    short = reversal_travel(np.arange(100.0), np.arange(100.0))       # under 200 samples
    for r in (flat, short):
        assert np.isnan(r.s) and np.isnan(r.lag) and r.reversals == 0


def test_reading_is_in_the_units_of_co(cfg):
    spec = _plant(cfg)
    d = simulate_loop(dc.replace(spec, stiction_s=4.0, stiction_j=3.0, load_std=0.3), 3000, seed=0, cfg=cfg)
    a = reversal_travel(d.op.to_numpy(), d.pv.to_numpy())
    b = reversal_travel(10 * d.op.to_numpy(), d.pv.to_numpy())       # CO stored in other units
    assert b.s == pytest.approx(a.s)                                  # same in CO standard deviations
    assert b.s_co == pytest.approx(10 * a.s_co)


# ------------------------------------------------------------- the rejected fit, kept honest

def test_valve_path_is_the_simulator_valve():
    """fit_valve was rejected at development (VALIDATION.md 23.1); the valve it
    used is still the simulator's own valve, step for step."""
    xs = 50 + 6 * np.sin(np.linspace(0, 12, 400)) + np.random.default_rng(0).normal(0, 0.3, 400)
    v = StictionValve(3.0, 1.5, initial=float(xs[0]))
    np.testing.assert_allclose(_valve_path_py(xs, 3.0, 1.5), [v.step(x) for x in xs])
