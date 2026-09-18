"""API tests.

These use a model fitted on synthetic data, so they run in CI without the
industrial dataset - which is not committed. That is the point: the test
suite must pass on a clean machine.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from chemai.config import SoftSensorConfig
from chemai.models import SoftSensor
from chemai.api import app as app_module


@pytest.fixture
def client(monkeypatch):
    """Client with a model fitted on synthetic, physically-shaped data."""
    cfg = SoftSensorConfig()
    rng = np.random.default_rng(0)
    T = rng.uniform(440, 470, 200)
    P = rng.uniform(220, 240, 200)
    vp = 80 * (1000 / T) + 0.15 * P - 200 + rng.normal(0, 0.5, 200)
    df = pd.DataFrame({cfg.temperature_col: T, cfg.pressure_col: P, cfg.target: vp})

    monkeypatch.setattr(app_module, "_fit_model", lambda c: (SoftSensor(c).fit(df), len(df)))
    with TestClient(app_module.create_app(cfg)) as c:
        yield c


def test_health_reports_fitted(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_fitted"] is True
    assert body["training_rows"] == 200


def test_predict_returns_all_three_fields(client):
    r = client.post("/predict", json={"temperature": 455.0, "pressure": 230.0})
    assert r.status_code == 200
    b = r.json()
    assert b["interval_lower"] < b["vapour_pressure"] < b["interval_upper"]
    assert b["in_envelope"] is True
    assert b["warning"] is None


def test_predict_flags_extrapolation(client):
    """Outside the envelope the response must say so, not fail silently."""
    r = client.post("/predict", json={"temperature": 560.0, "pressure": 180.0})
    assert r.status_code == 200
    b = r.json()
    assert b["in_envelope"] is False
    assert b["warning"] is not None


def test_rejects_celsius_temperature(client):
    """A Celsius value would silently produce a wrong 1000/T."""
    r = client.post("/predict", json={"temperature": 45.0, "pressure": 230.0})
    assert r.status_code == 422


def test_rejects_non_physical_input(client):
    assert client.post("/predict", json={"temperature": -10.0, "pressure": 230.0}).status_code == 422
    assert client.post("/predict", json={"temperature": 455.0, "pressure": 0.0}).status_code == 422


def test_rejects_missing_field(client):
    assert client.post("/predict", json={"temperature": 455.0}).status_code == 422


def test_service_starts_degraded_without_data(monkeypatch):
    """No data must not crash the service - it must report degraded."""
    def _raise(cfg):
        raise FileNotFoundError("no data")
    monkeypatch.setattr(app_module, "_fit_model", _raise)

    with TestClient(app_module.create_app()) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded"
        assert h["model_fitted"] is False
        assert c.post("/predict", json={"temperature": 455.0, "pressure": 230.0}).status_code == 503


def test_the_root_url_lands_somewhere(client):
    """Two projects share one deployment; a visitor opening the root must not
    meet a 404."""
    r = client.get("/")
    assert r.status_code == 200
    assert "loops/" in r.text and "docs" in r.text
