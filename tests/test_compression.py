"""Archive compression - and the measured cost of it.
The simulator is the ground truth: stiction is injected, so the answer is known."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import detrend

from chemai.config import ControlLoopConfig, DataQualityConfig
from chemai.data import (apply_deadband, compress, downsample, resolution_is_sufficient,
                         sample_loop_spec, samples_per_cycle, simulate_loop)
from chemai.features import (assess, cycles_in_record, oscillation_regularity,
                             prepare_for_shape, shape_index, shape_signal)

cfg = ControlLoopConfig()


def _diagnose(sp, pv, op, loop_type="F"):
    """The adopted baseline, as the weekly report applies it."""
    e = detrend(np.asarray(sp) - np.asarray(pv))
    period, regularity = oscillation_regularity(e)
    if not np.isfinite(period) or cycles_in_record(len(e), period) < cfg.min_cycles:
        return "insufficient data"
    signal, _ = shape_signal(loop_type, pv, sp, op)
    shape = shape_index(prepare_for_shape(signal, period, cfg.shape_harmonics))
    if regularity <= cfg.regularity_threshold or not np.isfinite(shape):
        return "undetermined"
    return "stiction" if shape > cfg.shape_triangular else "tuning"


def _sticky_loops(n=6, samples=6000):
    return [simulate_loop(sample_loop_spec("F", "stiction", np.random.default_rng(200 + i), cfg),
                          samples, seed=i, cfg=cfg) for i in range(n)]


# ----------------------------------------------------------- the mechanics

def test_downsample_keeps_the_first_sample():
    x = np.arange(10.0)
    np.testing.assert_array_equal(downsample(x, 3), [0, 3, 6, 9])
    frame = pd.DataFrame({"pv": x})
    assert len(downsample(frame, 5)) == 2 and frame.pv.iloc[0] == 0
    with pytest.raises(ValueError):
        downsample(x, 0)


def test_deadband_holds_the_last_stored_value():
    x = np.array([0.0, 0.4, 0.9, 1.6, 1.7])
    np.testing.assert_allclose(apply_deadband(x, 1.0), [0, 0, 0, 1.6, 1.6])
    np.testing.assert_allclose(apply_deadband(x, 0.0), x)      # no compression
    with pytest.raises(ValueError):
        apply_deadband(x, -1)


def test_deadband_creates_repeated_samples():
    x = np.sin(np.arange(500) / 7)
    assert np.mean(np.diff(apply_deadband(x, 0.5)) == 0) > 0.5


def test_compress_applies_both():
    d = pd.DataFrame({"pv": np.sin(np.arange(200) / 5), "op": np.cos(np.arange(200) / 5)})
    out = compress(d, factor=4, deadband_std=0.5)
    assert len(out) == 50 and np.mean(np.diff(out.pv) == 0) > 0


def test_samples_per_cycle():
    assert samples_per_cycle(21, 2) == pytest.approx(10.5)
    assert samples_per_cycle(float("nan")) == 0.0
    assert resolution_is_sufficient(21, 2) and not resolution_is_sufficient(21, 5)


# ------------------------------------------------------------- the cost

def test_a_slower_archive_loses_the_stiction_diagnosis():
    loops = _sticky_loops()
    found = {k: sum(_diagnose(*[downsample(d, k)[c].to_numpy() for c in ("sp", "pv", "op")])
                    == "stiction" for d in loops) for k in (1, 10)}
    assert found[1] >= 4                                   # most are found at full rate
    assert found[10] == 0                                  # none survive a 10x slower archive


def test_a_deadband_turns_stiction_into_tuning():
    """The silent error: not 'I don't know' but a confident wrong answer that
    sends the control engineer instead of maintenance (VALIDATION.md 21.2)."""
    calls = []
    for d in _sticky_loops():
        squeezed = compress(d, deadband_std=1.0)
        calls.append(_diagnose(d.sp.to_numpy(), squeezed.pv.to_numpy(), squeezed.op.to_numpy()))
    assert calls.count("stiction") == 0
    assert calls.count("tuning") >= 3                      # wrong, and confident


def test_a_small_deadband_is_harmless():
    loops = _sticky_loops(n=4)
    fine = sum(_diagnose(d.sp.to_numpy(), *[compress(d, deadband_std=0.25)[c].to_numpy()
                                            for c in ("pv", "op")]) == "stiction" for d in loops)
    full = sum(_diagnose(*[d[c].to_numpy() for c in ("sp", "pv", "op")]) == "stiction"
               for d in loops)
    assert fine == full


# ------------------------------------------------- the data-quality guard

def test_quality_check_flags_a_coarse_archive():
    n = 2000
    t = np.arange(n)
    pv = 50 + np.sin(2 * np.pi * t / 8) + 0.05 * np.random.default_rng(0).normal(size=n)
    op = 50 + np.cos(2 * np.pi * t / 8)
    sp = np.full(n, 50.0)
    coarse = assess(sp, pv, op, period=8)                  # 8 samples per cycle
    fine = assess(sp, pv, op, period=40)                   # 40 samples per cycle
    assert "coarse_archive" in coarse.flags and not coarse.ok_for_pv_shape
    assert "coarse_archive" not in fine.flags and fine.ok_for_pv_shape


def test_the_threshold_comes_from_the_config():
    assert DataQualityConfig().min_samples_per_cycle == 10.0
