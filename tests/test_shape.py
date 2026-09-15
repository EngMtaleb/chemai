"""Shape index - known waveforms, and the three limits that matter."""
import numpy as np
import pytest

from chemai.config import ControlLoopConfig
from chemai.features import (half_cycle_fits, prepare_for_shape, shape_index, shape_signal)

T = 60.0
t = np.arange(3000)
SINE = np.sin(2 * np.pi * t / T)
TRIANGLE = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * t / T))
SAWTOOTH = (t % T) / T * 2 - 1
SQUARE = np.sign(np.sin(2 * np.pi * t / T))
cfg = ControlLoopConfig()


def test_clean_waveforms():
    assert shape_index(SINE) < cfg.shape_sinusoidal
    assert shape_index(TRIANGLE) > cfg.shape_triangular
    assert shape_index(SAWTOOTH) > cfg.shape_triangular


def test_square_wave_reads_as_a_sine():
    """The limit that dictates which signal is measured: a flat-topped half
    cycle is neither shape, and the index lands near 0.5."""
    assert shape_index(SQUARE) < cfg.shape_triangular


def test_amplitude_does_not_matter():
    assert shape_index(17 * TRIANGLE) == pytest.approx(shape_index(TRIANGLE), abs=1e-9)


def test_half_cycle_fit_prefers_the_true_shape():
    zc = np.flatnonzero(np.diff(np.sign(TRIANGLE)) != 0)
    e_sine, e_tri, fit_s, fit_t = half_cycle_fits(TRIANGLE[zc[1] + 1:zc[2] + 1])
    assert e_tri < e_sine / 5 and len(fit_s) == len(fit_t)


def test_too_few_half_cycles_gives_nan():
    assert np.isnan(shape_index(SINE[:60]))          # one period
    assert np.isnan(shape_index(np.ones(500)))       # no zero crossings


def test_noise_collapses_the_index_and_filtering_restores_it():
    rng = np.random.default_rng(0)
    noisy_sine = SINE + 0.3 * rng.standard_normal(len(t))
    noisy_tri = TRIANGLE + 0.3 * rng.standard_normal(len(t))
    raw_gap = shape_index(noisy_tri) - shape_index(noisy_sine)
    filtered_gap = (shape_index(prepare_for_shape(noisy_tri, T))
                    - shape_index(prepare_for_shape(noisy_sine, T)))
    assert raw_gap < 0.1                              # unusable without filtering
    assert filtered_gap > 0.15                        # usable with it


def test_filtering_costs_a_little_sharpness():
    assert shape_index(prepare_for_shape(TRIANGLE, T)) < shape_index(TRIANGLE)


def test_prepare_removes_slow_drift():
    drifting = TRIANGLE + np.linspace(0, 8, len(t))
    assert shape_index(prepare_for_shape(drifting, T)) > cfg.shape_triangular
    with pytest.raises(ValueError):
        prepare_for_shape(TRIANGLE, 0)


@pytest.mark.parametrize("loop_type,expected", [("F", "OP"), ("P", "OP"), ("Q", "OP"),
                                                ("T", "OP"), ("L", "PV")])
def test_shape_signal_follows_loop_type(loop_type, expected):
    pv, sp, op = np.arange(10.0), np.zeros(10), np.arange(10.0) * 2
    assert shape_signal(loop_type, pv, sp, op)[1] == expected


def test_shape_signal_falls_back_when_op_is_unusable():
    pv, sp = np.arange(10.0), np.zeros(10)
    assert shape_signal("F", pv, sp, np.full(10, 42.0))[1] == "PV"      # OP never moves
    assert shape_signal("F", pv, sp, np.full(10, np.nan))[1] == "PV"    # OP not recorded
