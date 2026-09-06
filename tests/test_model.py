import numpy as np
import pandas as pd
import pytest

from chemai.config import SoftSensorConfig
from chemai.models import SoftSensor


@pytest.fixture
def synthetic():
    """A physically-shaped synthetic set: VP rises as 1/T rises."""
    rng = np.random.default_rng(0)
    cfg = SoftSensorConfig()
    T = rng.uniform(440, 470, 200)
    P = rng.uniform(220, 240, 200)
    vp = 80 * (1000 / T) + 0.15 * P - 200 + rng.normal(0, 0.5, 200)
    return cfg, pd.DataFrame({cfg.temperature_col: T, cfg.pressure_col: P,
                              cfg.target: vp})


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        SoftSensor().predict_one(450.0, 228.0)


def test_fit_and_predict(synthetic):
    cfg, df = synthetic
    m = SoftSensor(cfg).fit(df)
    assert m.residual_sigma_ > 0
    assert m.envelope_threshold_ > 0
    p = m.predict_one(455.0, 230.0)
    assert p.lower < p.value < p.upper


def test_coefficient_sign_is_physical(synthetic):
    """invT coefficient must be positive: lower T -> higher vapour pressure.

    If this ever flips, the model has learned a relationship inverted
    against the physics and nothing downstream means anything.
    """
    cfg, df = synthetic
    m = SoftSensor(cfg).fit(df)
    ridge = m.pipeline.named_steps["ridge"]
    assert ridge.coef_[0] > 0


def test_envelope_flags_far_inputs(synthetic):
    cfg, df = synthetic
    m = SoftSensor(cfg).fit(df)
    assert m.predict_one(455.0, 230.0).in_envelope is True
    far = m.predict_one(560.0, 180.0)
    assert far.in_envelope is False
    assert far.warning is not None


def test_interval_width_matches_sigma(synthetic):
    cfg, df = synthetic
    m = SoftSensor(cfg).fit(df)
    p = m.predict_one(455.0, 230.0)
    assert np.isclose(p.upper - p.lower, 2 * 1.96 * m.residual_sigma_, atol=0.05)
