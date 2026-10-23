"""UF pilot record: layouts, the clock audit, daylight saving, episodes."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from chemai.config import UFPilotConfig
from chemai.data.uf_pilot import (
    TAG_ORDER, assign_episodes, count_bad_lines, file_kind, in_daylight_saving, read_run,
    to_standard_time,
)

HEADER = "Time,Time Elapsed," + ",".join(TAG_ORDER)
UNITS = "Stamp,(sec)," + ",".join(["x"] * len(TAG_ORDER))


def write_clock_csv(path, start_excel=41430.5, n=5, bad_line=False):
    rows = [HEADER, UNITS]
    for i in range(n):
        vals = [f"{start_excel + i / 86400:.9f}", f"{i:.1f}"] + [f"{1.0 + j:.3f}" for j in range(len(TAG_ORDER))]
        rows.append(",".join(vals))
    if bad_line:
        rows.insert(4, ",".join(["1"] * 39))
    path.write_text("\n".join(rows) + "\n")
    return path


def test_layouts_are_told_apart(tmp_path):
    a = write_clock_csv(tmp_path / "20130606.csv")
    b = tmp_path / "20130607.csv"
    b.write_text("41431.5,0," + ",".join(["1"] * len(TAG_ORDER)) + "\n")
    c = tmp_path / "20130608.csv"
    c.write_text("UF Inflow Rate,UF Element 1 (E1) Inflow rate\n1,2\n")
    d = tmp_path / "20130609.csv"
    d.write_bytes(b"PK\x03\x04 not really a sheet")
    assert [file_kind(p) for p in (a, b, c, d)] == ["csv_clock", "csv_clock_noheader", "csv_noclock", "sheet"]


def test_read_run_uses_the_logged_clock(tmp_path):
    p = write_clock_csv(tmp_path / "20130606.csv", start_excel=41431.5)   # 2013-06-06 12:00
    df = read_run(p)
    assert df.index[0] == pd.Timestamp("2013-06-06 12:00:00")
    assert list(df.columns)[:3] == ["q_uf", "q_e1", "q_e2"]
    assert df["temp_c"].iloc[0] == pytest.approx(16.0)


def test_malformed_lines_are_counted_and_skipped(tmp_path):
    p = write_clock_csv(tmp_path / "20130606.csv", n=6, bad_line=True)
    assert count_bad_lines(p) == 1
    assert len(read_run(p)) == 6


def test_daylight_saving_window():
    t = pd.DatetimeIndex(["2013-03-10 01:30", "2013-03-10 03:30", "2013-11-03 00:30",
                          "2013-11-03 01:30", "2013-07-01 12:00", "2014-01-15 12:00"])
    assert in_daylight_saving(t).tolist() == [False, True, True, False, True, False]
    std = to_standard_time(t)
    assert std[1] == pd.Timestamp("2013-03-10 02:30") and std[5] == t[5]
    assert (to_standard_time(t, UFPilotConfig(remove_daylight_saving=False)) == t).all()


def test_episodes_chain_runs_that_follow_each_other():
    runs = pd.DataFrame({
        "t_start": pd.to_datetime(["2013-06-04 15:00", "2013-06-05 16:01", "2013-06-09 10:00"]),
        "t_end": pd.to_datetime(["2013-06-05 16:00", "2013-06-06 15:59", "2013-06-09 20:00"]),
    })
    ep = assign_episodes(runs)
    assert ep.iloc[0] == ep.iloc[1] != ep.iloc[2]
