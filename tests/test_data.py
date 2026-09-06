import numpy as np
import pandas as pd

from chemai.config import SoftSensorConfig
from chemai.data import add_periods, split_by_period
from chemai.data.loaders import check_derived_columns


def test_check_derived_columns_finds_exact_transform():
    df = pd.DataFrame({"Temp9": [400.0, 450.0, 500.0]})
    df["InvTemp3"] = 1000.0 / df["Temp9"]
    found = check_derived_columns(df)
    assert found.get("InvTemp3") == "1000/Temp9"


def test_check_derived_columns_ignores_unrelated():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 9.0, 1.0]})
    assert check_derived_columns(df) == {}


def test_periods_follow_breaks():
    cfg = SoftSensorConfig()
    df = pd.DataFrame({"x": range(250)})
    out = add_periods(df, cfg)
    assert out.loc[0, "period"] == 0
    assert out.loc[170, "period"] == 1
    assert out.loc[200, "period"] == 2


def test_split_is_not_random():
    """Train and test must come from different campaigns, with no overlap."""
    cfg = SoftSensorConfig()
    df = pd.DataFrame({"x": range(250)})
    tr, te = split_by_period(df, cfg)
    assert set(tr.index).isdisjoint(set(te.index))
    assert tr.period.nunique() == 1 and te.period.nunique() == 1
