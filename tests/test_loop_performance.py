"""Known-answer tests for the performance indices. Every expected value is
derived analytically, not taken from a previous run."""
import numpy as np
import pytest

from chemai.features import (delay_samples, harris_index, harris_sensitivity,
                             oscillation_regularity)


def _ar1(a, n=20000, seed=0):
    e = np.random.default_rng(seed).normal(size=n)
    y = np.zeros(n)
    for k in range(1, n):
        y[k] = a * y[k - 1] + e[k]
    return y


def test_delay_samples():
    assert delay_samples(0.0, 1.0) == 1          # zero-order hold alone
    assert delay_samples(4.0, 2.0) == 3
    assert delay_samples(3.9, 1.0) == 4
    with pytest.raises(ValueError):
        delay_samples(-1.0, 1.0)


@pytest.mark.parametrize("a,d", [(0.8, 3), (0.9, 5), (0.5, 2)])
def test_harris_uncontrolled_ar1_closed_form(a, d):
    """No control, AR(1) disturbance: eta = 1 - a^(2d) exactly."""
    assert harris_index(_ar1(a), d) == pytest.approx(1 - a ** (2 * d), abs=0.03)


def test_harris_at_minimum_variance_is_one():
    """Under minimum-variance control the output is MA(d-1) in the
    innovations - nothing left for any controller to remove."""
    e = np.random.default_rng(1).normal(size=20000)
    d, a = 3, 0.8
    y = e + a * np.roll(e, 1) + a ** 2 * np.roll(e, 2)
    assert harris_index(y[5:], d) == pytest.approx(1.0, abs=0.03)


def test_overestimated_delay_flatters_the_loop():
    s = harris_sensitivity(_ar1(0.9), range(1, 8))
    vals = list(s.values())
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_harris_input_checks():
    with pytest.raises(ValueError):
        harris_index(_ar1(0.5), 0)
    with pytest.raises(ValueError):
        harris_index(np.ones(1000), 2)            # zero variance
    with pytest.raises(ValueError):
        harris_index(_ar1(0.5, n=100), 2)          # too short for AR(30)


def test_regularity_sine_vs_noise():
    t = np.arange(3000)
    T, r = oscillation_regularity(np.sin(2 * np.pi * t / 60))
    assert r > 1 and T == pytest.approx(60, rel=0.05)
    _, r_noise = oscillation_regularity(np.random.default_rng(0).normal(size=3000))
    assert r_noise < 1
    assert oscillation_regularity(np.ones(100)) == (pytest.approx(np.nan, nan_ok=True), 0.0)
