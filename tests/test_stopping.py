"""The stopping rule and the square-root law (Project 5)."""
import numpy as np
import pandas as pd
import pytest

from chemai.models.stopping import (
    average_cost, cost_of_stopping_at, cumulative, marginal_equals_average, rolling_median, sqrt_law,
)


def linear_cycle(F=2.0, m0=1.0, s=0.001, q=1.0, dt=0.1, n=4000):
    """Output at a constant rate q; live cost rises linearly with output: m = m0 + s V."""
    t = np.arange(n) * dt
    V = q * t
    live = m0 + s * V
    K = cumulative(t, live * q)            # running cost = integral of m dV
    return t, V, live, K, F


def test_cumulative_is_trapezoid():
    t = np.array([0.0, 1.0, 3.0])
    y = np.array([1.0, 3.0, 3.0])
    assert np.allclose(cumulative(t, y), [0.0, 2.0, 8.0])


def test_sqrt_law_closed_form_matches_brute_force():
    F, m0, s = 3.0, 0.7, 0.02
    v_star, j_star = sqrt_law(F, s, m0)
    grid = np.linspace(1, 200, 200001)
    j = m0 + F / grid + s * grid / 2
    assert v_star == pytest.approx(grid[np.argmin(j)], rel=1e-3)
    assert j_star == pytest.approx(j.min(), rel=1e-6)
    assert cost_of_stopping_at(v_star, F, s, m0) == pytest.approx(j_star)


def test_sqrt_law_twice_the_fixed_cost_is_sqrt2_longer():
    v1, _ = sqrt_law(1.0, 0.1)
    v2, _ = sqrt_law(2.0, 0.1)
    assert v2 / v1 == pytest.approx(np.sqrt(2))


def test_sqrt_law_degenerate_inputs():
    assert sqrt_law(0.0, 0.1)[0] == float("inf")
    assert sqrt_law(1.0, 0.0)[0] == float("inf")


def test_rule_stops_at_the_minimum_of_the_average_cost():
    t, V, live, K, F = linear_cycle()
    dec = marginal_equals_average(t, live, K, V, F, window_s=1.0)
    j = average_cost(F, K, V)
    best = int(np.nanargmin(j))
    assert dec.index is not None
    assert abs(dec.index - best) <= 7          # a trailing median lags by half its window (5 samples)
    assert j[dec.index] / j[best] - 1 < 1e-5   # and the cost is flat there
    v_star, _ = sqrt_law(F, 0.001, 1.0)
    assert V[dec.index] == pytest.approx(v_star, rel=0.01)


def test_rule_never_fires_while_the_cost_is_still_falling():
    t, V, live, K, F = linear_cycle(F=500.0, n=200)     # optimum far beyond the record
    dec = marginal_equals_average(t, live, K, V, F, window_s=1.0)
    assert dec.index is None
    assert dec.marginal < dec.average


def test_rule_respects_the_minimum_elapsed_time_and_allowed_mask():
    t, V, live, K, F = linear_cycle()
    dec = marginal_equals_average(t, live, K, V, F, window_s=1.0, min_elapsed_s=100.0)
    assert t[dec.index] >= 100.0
    allowed = np.zeros(len(t), bool)
    allowed[3000:] = True
    dec2 = marginal_equals_average(t, live, K, V, F, window_s=1.0, allowed=allowed)
    assert dec2.index >= 3000


def test_rolling_median_uses_only_the_past():
    t = np.arange(10.0)
    x = np.array([0, 0, 0, 0, 0, 100, 100, 100, 100, 100], float)
    m = rolling_median(t, x, window_s=3.0, min_periods=1)
    assert m[4] == 0          # the jump at t=5 is not seen before it happens
    assert m[9] == 100


def test_noisy_live_cost_is_smoothed_before_comparing():
    rng = np.random.default_rng(0)
    t, V, live, K, F = linear_cycle()
    noisy = live + rng.normal(0, 0.5, len(live))       # noise larger than the gap near the stop
    dec_raw = marginal_equals_average(t, noisy, K, V, F, window_s=0.1, min_periods=1)
    dec_smooth = marginal_equals_average(t, noisy, K, V, F, window_s=30.0, min_periods=100)
    v_star, _ = sqrt_law(F, 0.001, 1.0)
    assert abs(V[dec_smooth.index] - v_star) < abs(V[dec_raw.index] - v_star)
