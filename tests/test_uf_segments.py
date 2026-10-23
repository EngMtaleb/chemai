"""Filtration segments: viscosity, debouncing, cutting, and the fouling slope."""
import numpy as np
import pandas as pd
import pytest

from chemai.config import UFSegmentConfig
from chemai.features.uf_segments import configuration, cut_segments, segment_table, water_viscosity


def test_viscosity_of_water():
    assert water_viscosity(20.0) == pytest.approx(1.002e-3, rel=0.01)
    assert water_viscosity(10.0) > water_viscosity(25.0)


def rotating_run(hours=3.0, dt=10.0, period_min=24.0, rate_per_h=0.12, temp_wave=0.0, seed=0):
    """Two of three elements filter; the pair changes every `period_min`.

    Resistance grows exponentially at `rate_per_h` within each pair period and
    resets at the change. Temperature may swing; TMP follows viscosity.
    """
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2014-03-18 12:00") + pd.to_timedelta(np.arange(0, hours * 3600, dt), "s")
    since = (np.arange(len(t)) * dt) % (period_min * 60) / 3600
    pair = ((np.arange(len(t)) * dt) // (period_min * 60)).astype(int) % 3
    q = np.full(len(t), 20.0)
    e = np.zeros((len(t), 3))
    for k, (a, b) in enumerate([(0, 1), (1, 2), (0, 2)]):
        sel = pair == k
        e[sel, a] = e[sel, b] = 10.0
    temp = 15 + temp_wave * np.sin(np.arange(len(t)) * dt / 3600 * 2 * np.pi / 6)
    mu = water_viscosity(temp)
    r = np.exp(rate_per_h * since)
    tmp = r * mu * q * 400 + rng.normal(0, 0.005, len(t))
    return pd.DataFrame({"q_uf": q, "q_e1": e[:, 0], "q_e2": e[:, 1], "q_e3": e[:, 2], "q_bw": 0.0,
                         "p_uf_in": 25 + tmp, "p_uf_out": 25.0, "temp_c": temp, "turbidity": 0.5,
                         "pump_speed": 0.9, "p_mf_in": 3.0}, index=pd.DatetimeIndex(t, name="t_local"))


def test_debounce_ignores_flicker():
    df = rotating_run(hours=1.0, period_min=60)
    df.iloc[100:103, df.columns.get_loc("q_e3")] = 10.0          # 30 s blip on a third element
    code = configuration(df)
    assert len(np.unique(code)) == 1


def test_segments_follow_the_pair_changes():
    df = rotating_run(hours=2.0, period_min=24)
    segs = cut_segments(df)
    cfg = UFSegmentConfig()
    expected = 24 - (cfg.trim_start_s + cfg.trim_end_s) / 60
    lengths = [(df.index[b] - df.index[a]).total_seconds() / 60 for a, b in segs]
    assert len(segs) >= 4
    assert np.median(lengths) == pytest.approx(expected, abs=1.0)


def test_fouling_slope_is_recovered_and_not_fooled_by_temperature():
    df = rotating_run(hours=3.0, rate_per_h=0.12, temp_wave=1.5)
    seg = segment_table(df, "run")
    assert seg.fouling_pct_h.median() == pytest.approx(12.0, abs=0.5)
    assert set(seg.n_online) == {2}
