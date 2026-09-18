"""Model serialisation carries its provenance.

A pickled scikit-learn estimator is only valid for the version that wrote it.
These tests keep that fact visible instead of leaving it in a log."""
import pickle

import numpy as np
import pytest
import sklearn

from chemai.config import SoftSensorConfig
from chemai.models import SoftSensor, load_model, save_model


@pytest.fixture
def fitted():
    import pandas as pd
    cfg = SoftSensorConfig()
    rng = np.random.default_rng(0)
    T, P = rng.uniform(440, 470, 120), rng.uniform(220, 240, 120)
    vp = 80 * (1000 / T) + 0.15 * P - 200 + rng.normal(0, 0.5, 120)
    df = pd.DataFrame({cfg.temperature_col: T, cfg.pressure_col: P, cfg.target: vp})
    return SoftSensor(cfg).fit(df)


def test_round_trip_records_the_versions(fitted, tmp_path):
    path = tmp_path / "model.pkl"
    meta = save_model(fitted, path)
    assert meta["sklearn_version"] == sklearn.__version__ and meta["created"]
    model, loaded = load_model(path)
    assert loaded["version_match"] is True
    assert model.predict_one(450.0, 228.0).value == pytest.approx(
        fitted.predict_one(450.0, 228.0).value)


def test_a_model_from_another_version_is_flagged(fitted, tmp_path):
    path = tmp_path / "old.pkl"
    save_model(fitted, path)
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    payload["sklearn_version"] = "0.0.1"                  # as if fitted long ago
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    _, meta = load_model(path)
    assert meta["version_match"] is False


def test_a_legacy_pickle_still_loads_but_is_flagged(fitted, tmp_path):
    """Models saved before this format existed must not break the service -
    but their provenance is unknown, and that is reported."""
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as fh:
        pickle.dump(fitted, fh)
    model, meta = load_model(path)
    assert meta["format"] == 1 and meta["version_match"] is False
    assert "re-run training" in meta["note"]
    assert model.predict_one(450.0, 228.0).value == pytest.approx(
        fitted.predict_one(450.0, 228.0).value)


def test_health_reports_the_version_pair(monkeypatch, fitted, tmp_path):
    from fastapi.testclient import TestClient
    from chemai.api import app as app_module

    path = tmp_path / "model.pkl"
    save_model(fitted, path)
    monkeypatch.setattr(app_module, "MODEL_PATH", path)
    with TestClient(app_module.create_app()) as client:
        body = client.get("/health").json()
    assert body["model_sklearn_version"] == sklearn.__version__
    assert body["sklearn_version"] == sklearn.__version__
    assert body["version_match"] is True
