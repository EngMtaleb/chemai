"""The end-to-end pipeline: raw samples in, action and owner out.
Ground truth comes from the simulator, where the fault is injected."""
import numpy as np
import pytest

from chemai.config import ControlLoopConfig
from chemai.data import compress, sample_loop_spec, simulate_loop
from chemai.pipeline import analyse_loop, analyse_plant

cfg = ControlLoopConfig()


def _run(condition, loop_type="F", seed=3, n=3000, rng=4):
    spec = sample_loop_spec(loop_type, condition, np.random.default_rng(rng), cfg)
    return simulate_loop(spec, n, seed=seed, cfg=cfg), spec


def _as_loop(d, name="FC101", loop_type="F"):
    return {name: {"sp": d.sp.to_numpy(), "pv": d.pv.to_numpy(), "op": d.op.to_numpy(),
                   "loop_type": loop_type}}


# ------------------------------------------------------------- one loop

def test_a_sticky_valve_reaches_maintenance():
    d, spec = _run("stiction")
    r = analyse_loop(d.sp, d.pv, d.op, "FC101", "F", spec.ts, cfg)
    assert r.diagnosis == "stiction" and r.owner == "maintenance"
    assert r.shape > cfg.shape_triangular and r.regularity > cfg.regularity_threshold
    assert "regularity" in r.as_row()["evidence"]


def test_a_healthy_loop_is_not_accused():
    d, spec = _run("healthy")
    assert analyse_loop(d.sp, d.pv, d.op, "FC102", "F", spec.ts, cfg).diagnosis != "stiction"


def test_data_quality_outranks_the_shape():
    """A frozen transmitter is not a control diagnosis, however triangular the
    signal looks - it goes to instrumentation, urgently (VALIDATION.md 18.1)."""
    d, spec = _run("stiction")
    pv = d.pv.to_numpy().copy()
    pv[1000:1600] = pv[999]                         # transmitter stuck while OP keeps moving
    r = analyse_loop(d.sp, pv, d.op, "FC101", "F", spec.ts, cfg)
    assert r.diagnosis == "frozen_sensor" and r.owner == "instrumentation"
    assert "URGENT" in r.action


def test_a_coarse_archive_blocks_the_shape_instead_of_guessing():
    d, spec = _run("stiction")
    squeezed = compress(d, factor=10)
    r = analyse_loop(squeezed.sp, squeezed.pv, squeezed.op, "FC101", "F", spec.ts * 10, cfg)
    assert r.diagnosis != "stiction"
    assert np.isnan(r.shape) or "coarse_archive" in r.quality_flags or r.cycles < cfg.min_cycles


def test_level_loops_are_read_on_pv():
    d, spec = _run("stiction", loop_type="L", rng=7)
    assert analyse_loop(d.sp, d.pv, d.op, "LC1", "L", spec.ts, cfg).shape_signal == "PV"


def test_input_checks():
    with pytest.raises(ValueError):
        analyse_loop([1, 2, 3], [1, 2, 3], [1, 2, 3])          # too short
    with pytest.raises(ValueError):
        analyse_loop(np.zeros(100), np.zeros(100), np.zeros(99))


# ----------------------------------------------------------- whole plant

def test_a_single_loop_skips_the_plant_level_checks():
    d, spec = _run("stiction")
    out = analyse_plant(_as_loop(d), ts=spec.ts, cfg=cfg)
    assert out["propagation"] is None and out["cascades"] is None
    assert not out["simultaneous"] and len(out["loops"]) == 1


def test_victims_of_one_oscillation_get_no_work_orders():
    """The finding that changes Monday: ten faults are one fault and nine victims.

    A narrow-band oscillation is used so the test checks the plumbing, not the
    detection limits - those are measured in week5_propagation.py.
    """
    n, period = 3000, 40.0
    t = np.arange(n)
    rng = np.random.default_rng(0)
    wave = np.sin(2 * np.pi * t / period)
    loops = {}
    for k, (amp, noise) in enumerate([(1.0, 0.1), (0.8, 0.2), (0.6, 0.3)], start=1):
        pv = 50 + amp * wave + noise * rng.normal(size=n)
        loops[f"FC10{k}"] = {"sp": np.full(n, 50.0), "pv": pv,
                             "op": 50 + np.cos(2 * np.pi * t / period), "loop_type": "F"}
    out = analyse_plant(loops, ts=1.0, cfg=cfg)
    assert out["propagation"] is not None
    assert len(out["propagation"]["members"]) == 3
    victims = out["report"].table.query("action.str.contains('victim')", engine="python")
    assert len(victims) == 2                                  # all but the leading candidate


def test_a_cascade_is_reported_as_structure():
    from chemai.features import simulate_cascade
    run = simulate_cascade(T=2000, stiction_s=3.0, slip_s=2.5)
    loops = {"TC1": {"sp": run.sp_m, "pv": run.pv_m, "op": run.op_m, "loop_type": "T"},
             "FC1": {"sp": run.sp_s, "pv": run.pv_s, "op": run.op_s, "loop_type": "F"}}
    out = analyse_plant(loops, ts=1.0, cfg=cfg)
    pairs = out["cascades"]
    assert pairs and pairs[0]["master"] == "TC1" and pairs[0]["slave"] == "FC1"


def test_missing_location_weights_are_declared_not_invented():
    d, spec = _run("stiction")
    out = analyse_plant(_as_loop(d), ts=spec.ts, cfg=cfg)
    assert "WITHOUT plant location weights" in out["report"].note


def test_location_weights_are_used_when_supplied():
    d, spec = _run("stiction")
    out = analyse_plant(_as_loop(d), ts=spec.ts, location_weights={"FC101": 3}, cfg=cfg)
    assert out["report"].table.iloc[0].location == 3
    assert "WITHOUT" not in out["report"].note
