"""CCRO batch features on synthetic files shaped like the testbed record (Project 5)."""
import numpy as np
import pandas as pd
import pytest

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import batch_id, read_csv_bytes, snake
from chemai.features.ccro_batches import batch_curve, batch_table, phase_of, score_batch

DT = 0.5  # s


def synthetic_file(seq, t0, filt_min=60.0, drain=True, slope=0.02, exp="Exp1", p_cp=0.5):
    """One file: [drain of the previous brine] + start-up + stabilisation + filtration."""
    rows = []
    t = t0

    def add(state, seconds, **kw):
        nonlocal t
        for _ in range(int(seconds / DT)):
            rows.append(dict(state=state, runtime_hr=t / 3600.0, **kw))
            t += DT

    if drain:
        add("Discharge P2", 120, reject_flow_l_per_min=7.5, feed_flow_l_per_min=7.0,
            permeate_flow_l_per_min=0.07, feed_pump_power_draw_kw=0.5, recirculation_pump_power_draw_kw=0.05,
            reject_conductivity_ms_per_cm=5.0)
    add("Startup P2", 30, reject_flow_l_per_min=30.0, feed_flow_l_per_min=1.0, permeate_flow_l_per_min=1.0,
        feed_pump_power_draw_kw=0.5, recirculation_pump_power_draw_kw=0.4, reject_conductivity_ms_per_cm=3.0)
    add("Stabilization", 60, reject_flow_l_per_min=45.0, feed_flow_l_per_min=2.4, permeate_flow_l_per_min=2.4,
        feed_pump_power_draw_kw=0.7, recirculation_pump_power_draw_kw=p_cp, reject_conductivity_ms_per_cm=3.5)
    n = int(filt_min * 60 / DT)
    v = np.arange(n) * DT * 2.4 / 60.0                         # litres of permeate in filtration
    for i in range(n):
        rows.append(dict(state="Experiment", runtime_hr=t / 3600.0, reject_flow_l_per_min=45.0,
                         feed_flow_l_per_min=2.4, permeate_flow_l_per_min=2.4,
                         feed_pump_power_draw_kw=0.7 + slope * v[i] / 10.0,
                         recirculation_pump_power_draw_kw=p_cp,
                         reject_conductivity_ms_per_cm=4.0 + 0.08 * v[i]))
        t += DT
    df = pd.DataFrame(rows)
    df["dataset"], df["exp"], df["file"], df["seq"] = "dataset9", exp, f"Dataset{seq}.csv", seq
    df["row"] = np.arange(len(df))
    df["ts"] = pd.Timestamp("2021-10-01") + pd.to_timedelta(df.runtime_hr * 3600, unit="s")
    for c in ("feed_pressure_psi", "feed_temperature_c", "permeate_conductivity_us_per_cm", "flux_lmh",
              "conductivity_critical", "set_low_rej_cond_ms_per_cm"):
        df[c] = 1.0
    df["feed_pump_set_flux_lmh"], df["recirculation_pump_set_flow_l_per_min"] = 20.0, 45.0
    return df, t


def three_files(**kw):
    frames, t = [], 0.0
    for k in (1, 2, 3):
        f, t = synthetic_file(k, t, drain=(k > 1), **kw)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_phase_names():
    assert phase_of("Discharge P1") == "drain"
    assert phase_of("Initialize P2") == "startup"
    assert phase_of("Stabilization") == "stab"
    assert phase_of("Experiment") == "filt"
    assert phase_of(None) == "other"


def test_snake_and_batch_id():
    assert snake("Feed Pump Power Draw (kW)") == "feed_pump_power_draw_kw"
    assert snake("Specific Energy (kWh/m^3)") == "specific_energy_kwh_per_m3"
    assert batch_id("Dataset6_20221_10_BayesianOptimization/Exp2/Dataset14.csv") == ("dataset6", "Exp2", "Dataset14.csv", 14)


def test_reader_skips_title_line_and_parses():
    raw = (b"Date:,6/3/21,Time:,1:33 PM,Operator:,\n"
           b"Date,Time,Reject Conductivity (mS.cm),STATE\n6/3/21,1:33:33 PM,21.5,Experiment\n"
           b"6/3/21,1:33:34 PM,NaN,Experiment\n")
    df = read_csv_bytes(raw)
    assert list(df.columns) == ["ts", "reject_conductivity_ms_per_cm", "state"]
    assert df.reject_conductivity_ms_per_cm.iloc[0] == pytest.approx(21.5)
    assert np.isnan(df.reject_conductivity_ms_per_cm.iloc[1])
    assert read_csv_bytes(b"\r\n") is None


def test_batch_pairs_its_brine_with_the_next_files_drain():
    b = batch_table(three_files())
    assert len(b) == 3
    assert b.brine_l.iloc[0] == pytest.approx(7.5 * 2.0, rel=0.01)       # 2 min at 7.5 L/min
    assert b.brine_l.iloc[1] == pytest.approx(15.0, rel=0.01)
    assert np.isnan(b.brine_l.iloc[2])                                    # the last brine is not in the record
    assert b.prior_brine_l.iloc[0] == CCROConfig().default_brine_l        # first batch: no drain before it
    assert b.prior_brine_l.iloc[1] == pytest.approx(15.0, rel=0.01)
    assert b.constant_setpoints.all()
    assert b.filt_min.iloc[0] == pytest.approx(60.0, abs=0.1)


def test_a_gap_breaks_the_pairing():
    f1, t = synthetic_file(1, 0.0, drain=False)
    f2, _ = synthetic_file(2, t + 3600.0)            # an hour later
    b = batch_table(pd.concat([f1, f2], ignore_index=True))
    assert np.isnan(b.brine_l.iloc[0])


def test_rule_finds_the_best_end_of_a_long_batch():
    cfg = CCROConfig()
    df = three_files(filt_min=120.0)
    b = batch_table(df, cfg)
    g = df[df.file == "Dataset2.csv"]                 # its fixed cost is known from the drain before it
    curve = batch_curve(g, cfg)
    s = score_batch(curve, b.iloc[1], b=1.0, cfg=cfg)
    assert s["evaluable"] and s["inside"] and s["rule_stopped"]
    assert s["regret_rule_pct"] < 0.05
    assert s["regret_end_pct"] > s["regret_rule_pct"]


def test_short_batch_is_not_inside_and_the_rule_waits():
    cfg = CCROConfig()
    df = three_files(filt_min=12.0)
    b = batch_table(df, cfg)
    curve = batch_curve(df[df.file == "Dataset1.csv"], cfg)
    s = score_batch(curve, b.iloc[0], b=10.0, cfg=cfg)
    assert s["evaluable"] and not s["inside"]
    assert s["falling_at_end"]
    assert not s["rule_stopped"]
