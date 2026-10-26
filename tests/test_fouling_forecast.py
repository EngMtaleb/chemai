"""The fouling forecast. Every number travels with its interval and its flag;
these tests pin the behaviour an engineer relies on."""
import numpy as np
import pytest

from chemai.config import ROConfig
from chemai.models.fouling_forecast import forecast_cycle, level_at, prepare_window

cfg = ROConfig()


def _rise(days=42, rate=0.002, start=1.03, noise=0.0, seed=0, gaps=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days + 1, dtype=float)
    if gaps:
        keep = rng.random(len(t)) > gaps
        keep[-1] = True
        t = t[keep]
    return t, start + rate * t + rng.normal(0, noise, len(t))


def test_a_straight_rise_is_forecast_to_the_right_day():
    t, x = _rise(rate=0.002, start=1.03)
    f = forecast_cycle(t, x, 1.20)
    # reaches 1.20 on day 85; last observation day 42
    assert f.status == "crossing"
    assert f.days_to_level == pytest.approx(43, abs=0.6)
    assert f.rate_per_week == pytest.approx(0.014, abs=1e-4)


def test_the_interval_contains_the_date_and_widens_with_noise():
    t, x = _rise(noise=0.004, seed=1)
    quiet = forecast_cycle(t, x, 1.20)
    t, x = _rise(noise=0.02, seed=1)
    noisy = forecast_cycle(t, x, 1.20)
    for f in (quiet, noisy):
        assert f.days_lo <= f.days_to_level <= (f.days_hi or np.inf)
    assert (noisy.days_hi or 1e9) - noisy.days_lo > quiet.days_hi - quiet.days_lo


def test_one_bad_day_cannot_tilt_the_line():
    t, x = _rise(noise=0.003, seed=2)
    clean = forecast_cycle(t, x, 1.20)
    x2 = x.copy()
    x2[-3] += 0.25                                   # a transmitter spike near the end
    spiked = forecast_cycle(t, x2, 1.20)
    assert spiked.days_to_level == pytest.approx(clean.days_to_level, abs=2.0)


def test_a_falling_trend_is_no_reason_to_clean():
    t, x = _rise(rate=-0.0015, start=1.12, noise=0.003, seed=3)
    f = forecast_cycle(t, x, 1.15)
    assert f.status == "falling" and f.days_to_level is None


def test_a_flat_noisy_record_is_not_in_sight():
    t, x = _rise(rate=0.0, start=1.05, noise=0.01, seed=4)
    f = forecast_cycle(t, x, 1.30)
    assert f.status in {"not_in_sight", "falling"}
    assert f.days_to_level is None


def test_already_above_the_level_needs_no_date():
    t, x = _rise(rate=0.002, start=1.18)
    f = forecast_cycle(t, x, 1.15)
    assert f.status == "at_or_above" and f.days_to_level == 0.0


def test_too_few_days_is_insufficient_not_guessed():
    t, x = _rise(days=8)
    f = forecast_cycle(t, x, 1.15)
    assert f.status == "insufficient" and f.days_to_level is None and not f.in_envelope
    assert "insufficient" in f.warnings[0]


def test_looking_far_beyond_the_window_is_flagged():
    t, x = _rise(days=20, rate=0.0005, start=1.02, noise=0.001)
    f = forecast_cycle(t, x, 1.10)
    assert f.extrapolation_ratio > cfg.extrapolation_ratio_max
    assert not f.in_envelope and any("further ahead" in w for w in f.warnings)


def test_a_mostly_censored_window_warns():
    t, x = _rise(rate=0.0004, start=0.99)
    cens = x <= 1.0
    x = np.maximum(x, 1.0)
    f = forecast_cycle(t, x, 1.15, censored=cens)
    assert any("censored" in w for w in f.warnings)


def test_the_flushing_days_after_a_cleaning_are_skipped():
    t, x = _rise()
    tw, _, _ = prepare_window(t, x, cfg)
    assert tw.min() >= cfg.skip_after_cip_days


def test_a_review_level_in_percent_is_refused():
    t, x = _rise()
    with pytest.raises(ValueError, match="1.15"):
        forecast_cycle(t, x, 15.0)


def test_repeated_or_negative_days_are_refused():
    t, x = _rise()
    with pytest.raises(ValueError, match="repeated"):
        forecast_cycle(np.r_[t, t[-1]], np.r_[x, x[-1]], 1.2)
    with pytest.raises(ValueError, match="negative"):
        forecast_cycle(t - 5, x, 1.2)


def test_the_gap_law_bends_earlier_but_only_by_a_tenth_at_these_levels():
    """The gap law is convex, so it always forecasts the crossing earlier than the
    line - but between 1.0 and 1.3 only by about a tenth of the horizon, less than
    the forecast's own P10-P90 spread (VALIDATION.md section 7)."""
    t, x = _rise(rate=0.002, start=1.05, noise=0.005, seed=5)
    a = forecast_cycle(t, x, 1.20, law="linear")
    b = forecast_cycle(t, x, 1.20, law="blockage")
    assert b.days_to_level < a.days_to_level
    assert (a.days_to_level - b.days_to_level) / a.days_to_level < 0.15


def test_level_at_projects_the_fitted_line():
    t, x = _rise(rate=0.002, start=1.03)
    assert level_at(t, x, 14) == pytest.approx(1.03 + 0.002 * 56, abs=1e-3)


def test_same_input_same_answer():
    t, x = _rise(noise=0.01, seed=6, gaps=0.3)
    assert forecast_cycle(t, x, 1.2).to_dict() == forecast_cycle(t, x, 1.2).to_dict()
