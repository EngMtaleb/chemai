"""UF backwash events and cleaning days (Project 5)."""
import numpy as np
import pandas as pd
import pytest

from chemai.features.uf_cycles import backwash_events, cleaning_days, daily_baseline, run_summary


def synthetic_run(minutes=120, bw_every=20, dt=10.0):
    t = pd.date_range("2014-01-01", periods=int(minutes * 60 / dt), freq=f"{int(dt)}s")
    n = len(t)
    q_bw = np.zeros(n)
    k = np.arange(n) * dt / 60.0
    for start in np.arange(bw_every, minutes, bw_every):
        q_bw[(k >= start) & (k < start + 1)] = 34.0          # high pulse, 1 min
        q_bw[(k >= start + 1.5) & (k < start + 4)] = 13.0     # rinse after a short pause
    df = pd.DataFrame({"q_uf": 20.0, "q_bw": q_bw, "p_bw_filt": 40.0, "p_uf_in": 35.0, "p_uf_out": 28.0}, index=t)
    return df


def test_pulses_merge_into_one_backwash():
    ev = backwash_events(synthetic_run())
    assert len(ev) == 5                                       # at 20, 40, 60, 80, 100 min
    gal = 34 * 1 + 13 * 2.5                                   # gallons per backwash
    assert ev.volume_m3.iloc[0] == pytest.approx(gal * 3.785411784e-3, rel=0.05)


def test_short_noise_is_not_a_backwash():
    df = synthetic_run()
    df["q_bw"] = 0.0
    df.iloc[30, df.columns.get_loc("q_bw")] = 30.0           # one 10-s blip
    assert backwash_events(df).empty


def test_run_summary_volumes():
    s = run_summary(synthetic_run())
    assert s["backwashes"] == 5
    assert s["v_in_m3"] == pytest.approx(20 * 120 * 3.785411784e-3, rel=0.01)
    assert s["tmp_bar"] == pytest.approx(7 * 0.0689476)


def test_cleaning_days_need_a_large_fall_over_a_short_gap():
    seg = []
    days = list(pd.date_range("2014-01-01", periods=60, freq="D")) + list(pd.date_range("2014-06-01", periods=10, freq="D"))
    lnr = np.r_[np.linspace(6.2, 7.0, 30), np.linspace(6.25, 6.6, 30), np.full(10, 6.3)]
    for d, v in zip(days, lnr):
        for k in range(6):
            seg.append({"config": 3, "n_online": 2, "ln_r_start": v - np.log(2), "t_start": d + pd.Timedelta(hours=k),
                        "tmp_psi": 6.0, "q_gpm": 20.0, "temp_c": 15.0})
    daily = daily_baseline(pd.DataFrame(seg))
    assert daily.lnr.iloc[0] == pytest.approx(6.2)
    found = cleaning_days(daily)
    assert found == [pd.Timestamp("2014-01-31")]              # the fall across the 3-month gap is not counted
