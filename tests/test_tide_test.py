"""The tide test: exact surrogates, calibration, and what it must and must not find."""
import numpy as np
import pandas as pd
import pytest

from chemai.evaluation.tide_test import (
    CONSTITUENTS, PLACEBO, Prepared, TideTestConfig, _fit, block_shifts, confirm_out_of_sample,
    harmonic_block, prepare, rotation_stack, shift_blocks, tide_design, tide_test,
)


def synthetic_segments(seed=0, n_episodes=24, signal=None, phase_per_episode=False,
                       clock_wave=0.0, noise=1.0, day=1.5):
    """Segments every 30 min in multi-day episodes spread over two years."""
    rng = np.random.default_rng(seed)
    rows = []
    starts = np.sort(rng.uniform(0, 700, n_episodes))            # days
    for e, s in enumerate(starts):
        hours = rng.uniform(48, 200)
        t = pd.Timestamp("2013-01-01") + pd.to_timedelta(s, "D") + pd.to_timedelta(np.arange(0, hours, 0.5), "h")
        th = (t - pd.Timestamp("2000-01-01")) / pd.Timedelta(hours=1)
        y = 10 + rng.normal(0, noise, len(t)) + 0.02 * np.arange(len(t)) / 48
        tod = t.hour + t.minute / 60
        y += day * np.cos(2 * np.pi * tod / 24) + clock_wave * np.cos(2 * np.pi * tod / 12)
        if signal:
            ph = rng.uniform(0, 2 * np.pi) if phase_per_episode else 0.7
            y += signal * np.cos(2 * np.pi * th / CONSTITUENTS["M2"] - ph)
        n_runs = 2 if hours > 100 else 1
        run = np.where(np.arange(len(t)) < len(t) // n_runs, f"r{e}a", f"r{e}b")
        rows.append(pd.DataFrame({"t_std": t, "y": y, "run": run, "episode": e}))
    return pd.concat(rows, ignore_index=True)


def prep_for(seg, **kw):
    return prepare(seg, "y", TideTestConfig(**kw), covariates=False)


def test_rotated_surrogates_equal_brute_force():
    seg = synthetic_segments(seed=1, n_episodes=8)
    prep = prep_for(seg)
    design = tide_design(prep, CONSTITUENTS)
    rng = np.random.default_rng(5)
    shifts = block_shifts(len(design.blocks), rng, 52)
    ss_fast, beta_fast = _fit(design, design.zcat.T @ prep.y_res, rotation_stack(design, shifts))
    lut = dict(zip(design.blocks, shifts))
    x, _ = harmonic_block(prep.t_hours + np.array([lut[b] for b in prep.block]), CONSTITUENTS)
    xr = prep.residualise(x)
    beta = np.linalg.lstsq(xr, prep.y_res, rcond=None)[0]
    assert np.allclose(beta, beta_fast, atol=1e-8)
    assert ss_fast == pytest.approx(float((xr @ beta) @ (xr @ beta)), rel=1e-9)


def test_finds_a_coherent_tide():
    seg = synthetic_segments(seed=2, signal=0.6)
    res = tide_test(prep_for(seg, n_surrogates=199), with_ranges=False)
    assert res.p_value <= 0.01
    assert res.amplitudes["M2"] == pytest.approx(0.6, abs=0.15)


def test_ignores_a_tidal_period_without_a_tidal_phase():
    """Same 12.42 h period in every episode, random phase: not a tide."""
    seg = synthetic_segments(seed=3, signal=0.6, phase_per_episode=True)
    res = tide_test(prep_for(seg, n_surrogates=199), with_ranges=False)
    assert res.p_value > 0.05


def test_ignores_a_12_hour_clock_wave():
    seg = synthetic_segments(seed=4, clock_wave=2.0)
    res = tide_test(prep_for(seg, n_surrogates=199), with_ranges=False)
    assert res.p_value > 0.05


def test_null_is_calibrated_without_signal():
    ps = [tide_test(prep_for(synthetic_segments(seed=100 + i), n_surrogates=99), with_ranges=False).p_value
          for i in range(20)]
    assert np.mean(np.array(ps) <= 0.05) <= 0.20
    assert np.mean(ps) == pytest.approx(0.5, abs=0.2)


def test_placebo_periods_are_away_from_tides_and_the_day():
    for p in PLACEBO.values():
        assert min(abs(p - q) for q in [*CONSTITUENTS.values(), 24, 12, 8, 6]) > 0.3
        # a whole-week shift must rotate the placebo, or it is no test at all
        assert abs((168 / p) % 1 - 0.5) < 0.45


def test_out_of_sample_confirmation_uses_the_frozen_component():
    seg = synthetic_segments(seed=6, signal=0.6, n_episodes=30)
    cut = seg.t_std.quantile(0.5)
    first = seg[seg.episode.isin(seg[seg.t_std < cut].episode.unique())]
    second = seg[~seg.episode.isin(first.episode.unique())]
    out = confirm_out_of_sample(prep_for(first), prep_for(second), cfg=TideTestConfig(n_surrogates=199))
    assert out["p_value"] <= 0.05 and out["oos_variance_share"] > 0
    noise = synthetic_segments(seed=7, n_episodes=30)
    out = confirm_out_of_sample(prep_for(first), prep_for(noise), cfg=TideTestConfig(n_surrogates=199))
    assert out["p_value"] > 0.05


def test_shift_blocks_merge_close_episodes():
    t = np.array([0, 10, 50, 60, 500, 510.0])
    ep = np.array([1, 1, 2, 2, 3, 3])
    b = shift_blocks(t, ep, gap_hours=72)
    assert b[0] == b[2] and b[4] != b[0]


# ------------------------------------------------------------- the sun's clock

from chemai.evaluation.tide_test import DAY_HARMONICS  # noqa: E402


def with_day_wave(seed, amp, phase_per_episode=False, tide=0.0):
    seg = synthetic_segments(seed=seed, n_episodes=30, day=0.0)      # no built-in daily wave
    rng = np.random.default_rng(seed + 1)
    tod = seg.t_std.dt.hour + seg.t_std.dt.minute / 60
    ph = {e: (rng.uniform(0, 2 * np.pi) if phase_per_episode else 1.0) for e in seg.episode.unique()}
    seg["y"] = seg["y"] + amp * np.cos(2 * np.pi * tod / 24 - seg.episode.map(ph))
    th = (seg.t_std - pd.Timestamp("2000-01-01")) / pd.Timedelta(hours=1)
    seg["y"] = seg["y"] + tide * np.cos(2 * np.pi * th / CONSTITUENTS["M2"])
    return seg


def sun_prep(seg):
    return prepare(seg, "y", TideTestConfig(n_surrogates=199), covariates=False, tod=False)


def test_sun_clock_finds_a_daily_wave_with_one_phase():
    res = tide_test(sun_prep(with_day_wave(20, amp=0.8)), DAY_HARMONICS, TideTestConfig(n_surrogates=199),
                    with_ranges=False, shift="hours")
    assert res.p_value <= 0.01


def test_sun_clock_ignores_a_daily_wave_whose_phase_wanders():
    res = tide_test(sun_prep(with_day_wave(21, amp=0.8, phase_per_episode=True)), DAY_HARMONICS,
                    TideTestConfig(n_surrogates=199), with_ranges=False, shift="hours")
    assert res.p_value > 0.05


def test_sun_clock_ignores_the_moon():
    res = tide_test(sun_prep(with_day_wave(22, amp=0.0, tide=1.0)), DAY_HARMONICS,
                    TideTestConfig(n_surrogates=199), with_ranges=False, shift="hours")
    assert res.p_value > 0.05
