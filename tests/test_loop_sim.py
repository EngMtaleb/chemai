"""Simulator tests - valve physics, tuning definitions, and the guards that
stop a condition from silently drifting away from what its label says."""
import dataclasses as dc

import numpy as np
import pytest
from scipy.signal import detrend

from chemai.config import ControlLoopConfig
from chemai.data import (StictionValve, gain_margin, generate_dataset, sample_loop_spec,
                         simc_pi, simulate_loop)
from chemai.features import oscillation_regularity


def _ramp(v, xs):
    return np.array([v.step(x) for x in xs])


# ------------------------------------------------------------------ valve

def test_ideal_valve_is_transparent():
    v = StictionValve(0.0, 0.0, initial=50.0)
    xs = 50 + np.sin(np.linspace(0, 10, 200))
    np.testing.assert_allclose(_ramp(v, xs), xs)


def test_valve_rejects_slip_larger_than_band():
    with pytest.raises(ValueError):
        StictionValve(s=1.0, j=2.0)


def test_reversal_needs_the_full_band_s():
    """After a reversal the stem does not move until the command has
    travelled S back - the defining property of stiction."""
    s, j = 2.0, 0.5
    v = StictionValve(s, j, initial=50.0)
    _ramp(v, np.linspace(50, 60, 101))                 # moving up
    top = v.pos
    held = _ramp(v, np.linspace(60, 60 - 0.95 * s, 20))
    assert np.allclose(held, top)                      # stuck inside the band
    v.step(60 - 1.05 * s)
    assert v.pos < top                                 # slipped


def test_slip_jump_equals_j():
    """Stop, then resume in the same direction: the first move is a jump of J."""
    s, j = 3.0, 1.0
    v = StictionValve(s, j, initial=50.0)
    _ramp(v, np.linspace(50, 60, 101))
    for _ in range(5):
        v.step(60.0)                                   # command stops
    before = v.pos
    out = _ramp(v, np.linspace(60.0, 60 + 1.2 * j, 300))   # resume upward, slowly
    first = out[np.argmax(out != before)]
    assert first - before == pytest.approx(j, abs=0.01)
    assert np.all(out[out != before] > before)          # and it moved the right way


def test_pure_deadband_lags_by_half_band():
    v = StictionValve(2.0, 0.0, initial=50.0)
    out = _ramp(v, np.linspace(50, 60, 201))
    assert 60 - out[-1] == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- tuning

def test_simc_formulas():
    kc, ti = simc_pi(gain=2.0, tau=10.0, theta=1.0, tau_c=1.0, integrating=False)
    assert kc == pytest.approx(10.0 / (2.0 * 2.0)) and ti == pytest.approx(8.0)
    kc, ti = simc_pi(gain=0.01, tau=None, theta=2.0, tau_c=2.0, integrating=True)
    assert kc == pytest.approx(1 / (0.01 * 4.0)) and ti == pytest.approx(16.0)


def test_gain_margin_scales_inversely_with_kc():
    args = dict(ti=10.0, gain=1.0, tau=5.0, theta=1.0, integrating=False, dt=0.2)
    assert gain_margin(kc=2.0, **args) == pytest.approx(gain_margin(kc=1.0, **args) / 2, rel=1e-3)


@pytest.mark.parametrize("loop_type", ["F", "P", "L"])
def test_gain_margin_matches_simulated_stability(loop_type):
    """The computed gain margin is checked against the simulator itself:
    3% below it the loop is stable, 3% above it the loop diverges."""
    cfg = ControlLoopConfig()
    spec = sample_loop_spec(loop_type, "healthy", np.random.default_rng(3), cfg)
    quiet = dict(meas_std=0.0, load_std=0.05)

    def spread(f):
        s = dc.replace(spec, kc=spec.kc * spec.gain_margin * f, **quiet)
        pv = simulate_loop(s, 3000, seed=1, cfg=cfg).pv.to_numpy()
        return pv[-1000:].std()

    assert spread(0.97) < 0.2
    assert spread(1.03) > 5 * spread(0.97)


# -------------------------------------------------------------- conditions

@pytest.fixture(scope="module")
def cfg():
    return ControlLoopConfig()


@pytest.mark.parametrize("loop_type", ["F", "P", "L"])
def test_conditions_set_what_their_labels_say(cfg, loop_type):
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = sample_loop_spec(loop_type, "healthy", rng, cfg)
        t = sample_loop_spec(loop_type, "tuning_tight", rng, cfg)
        sl = sample_loop_spec(loop_type, "tuning_sluggish", rng, cfg)
        st = sample_loop_spec(loop_type, "stiction", rng, cfg)
        ex = sample_loop_spec(loop_type, "external_oscillation", rng, cfg)
        assert cfg.tight_gain_margin[0] <= t.gain_margin <= cfg.tight_gain_margin[1]
        assert sl.gain_margin > 3 * cfg.tight_gain_margin[1]
        assert h.gain_margin > 2.0
        assert h.stiction_s == 0 and st.stiction_s > 0 and 0 < st.stiction_j <= st.stiction_s
        assert h.ext_amplitude == 0 and ex.ext_amplitude > 0
        # healthy speed is capped by noise amplification to the valve
        assert h.kc * h.meas_std <= cfg.op_noise_limit + 1e-9


def test_unknown_condition_rejected(cfg):
    with pytest.raises(ValueError):
        sample_loop_spec("F", "leaking", np.random.default_rng(0), cfg)


def test_simulation_is_reproducible_and_records_mv(cfg):
    spec = sample_loop_spec("F", "stiction", np.random.default_rng(1), cfg)
    a = simulate_loop(spec, 500, seed=7, cfg=cfg)
    b = simulate_loop(spec, 500, seed=7, cfg=cfg)
    assert list(a.columns) == ["t", "sp", "pv", "op", "mv"] and len(a) == 500
    assert a.equals(b)
    assert (a.mv.between(0, 100)).all()
    assert (a.op != a.mv).any()                        # the valve is not the command


def test_ideal_valve_mv_equals_op(cfg):
    spec = sample_loop_spec("P", "healthy", np.random.default_rng(2), cfg)
    d = simulate_loop(spec, 400, seed=0, cfg=cfg)
    np.testing.assert_allclose(d.mv, d.op)


def test_dataset_is_fully_crossed_and_grouped(cfg):
    specs, runs = generate_dataset(n_loops_per_cell=1, seeds_per_loop=2, cfg=cfg,
                                   loop_types=("F", "L"))
    cells = {(s.loop_type, s.condition) for s in specs}
    assert cells == {(lt, c) for lt in ("F", "L") for c in cfg.conditions}
    assert len(runs) == 2 * len(specs)
    assert {k[0] for k in runs} == {s.loop_id for s in specs}


# ------------------------------------------------------- calibration guards

def _regular_fraction(cond, loop_type, cfg, n=6):
    rng = np.random.default_rng(5)
    hits = []
    for i in range(n):
        s = sample_loop_spec(loop_type, cond, rng, cfg)
        d = simulate_loop(s, 2000, seed=i, cfg=cfg)
        hits.append(oscillation_regularity(detrend((d.sp - d.pv).to_numpy()))[1] > 1)
    return float(np.mean(hits))


@pytest.mark.parametrize("loop_type", ["F", "P", "L"])
def test_external_oscillation_is_seen(cfg, loop_type):
    assert _regular_fraction("external_oscillation", loop_type, cfg) >= 0.8


@pytest.mark.parametrize("loop_type", ["F", "P", "L"])
def test_healthy_and_sluggish_do_not_oscillate(cfg, loop_type):
    assert _regular_fraction("healthy", loop_type, cfg) <= 0.2
    assert _regular_fraction("tuning_sluggish", loop_type, cfg) <= 0.2
