import numpy as np
import pandas as pd
import pytest

from chemai.features import antoine_form, build_features
from chemai.config import SoftSensorConfig


def test_antoine_form_is_inverse():
    t = np.array([400.0, 500.0])
    np.testing.assert_allclose(antoine_form(t), 1000.0 / t)


def test_antoine_form_rejects_non_absolute_temperature():
    """A Celsius value near zero would explode; catch it early."""
    with pytest.raises(ValueError):
        antoine_form(np.array([0.0, 25.0]))


def test_antoine_form_low_variance_is_expected():
    """This is why a scaler is mandatory before any penalised regressor."""
    t = np.linspace(400, 507, 100)
    assert antoine_form(t).std() < 0.2


def test_build_features_shape_and_names():
    cfg = SoftSensorConfig()
    df = pd.DataFrame({cfg.temperature_col: [450.0, 460.0],
                       cfg.pressure_col: [228.0, 230.0]})
    X = build_features(df, cfg)
    assert list(X.columns) == ["invT", "pressure"]
    assert len(X) == 2


def test_build_features_missing_column():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"Temp9": [450.0]}))
