"""RO fouling API. No model, no data files: the service is stateless, so these run
anywhere."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from chemai.api.app import create_app
from chemai.api.ro_app import create_ro_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_ro_app()) as c:
        yield c


def _cycle(rate=0.0015, start=1.05, days=42):
    t = list(np.arange(0, days + 1, dtype=float))
    return {"name": "Train A", "day": t, "dp_norm": [start + rate * d for d in t]}


def test_health_serves_the_limits(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["interval"] == "P10-P90"
    assert any("Normalisation" in limit for limit in body["limits"])
    assert "Zenodo" in body["data"]


def test_health_leads_with_the_real_data_result(client):
    """The headline failure is served first, not buried: on the real plant the date
    forecast did not beat a simple margin rule."""
    first = client.get("/health").json()["limits"][0]
    assert "did not beat" in first and "real plant" in first


def test_a_forecast_never_comes_without_its_interval(client):
    r = client.post("/forecast", json={"cycle": _cycle(), "review_level": 1.15})
    assert r.status_code == 200
    f = r.json()
    assert f["status"] == "crossing"
    assert f["days_lo"] <= f["days_to_level"] <= f["days_hi"]
    assert f["in_envelope"] in (True, False) and "extrapolation_ratio" in f


def test_dp_norm_sent_in_percent_is_refused(client):
    c = _cycle()
    c["dp_norm"] = [v * 100 for v in c["dp_norm"]]
    assert client.post("/forecast", json={"cycle": c}).status_code == 422


def test_mismatched_lengths_and_negative_days_are_refused(client):
    c = _cycle()
    c["dp_norm"] = c["dp_norm"][:-1]
    assert client.post("/forecast", json={"cycle": c}).status_code == 422
    c = _cycle()
    c["day"] = [d - 3 for d in c["day"]]
    assert client.post("/forecast", json={"cycle": c}).status_code == 422


def test_the_plan_gives_every_train_an_action_and_an_owner(client):
    ex = client.get("/examples/plant").json()["payload"]
    r = client.post("/plan", json=ex)
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3 and all(row["action"] and row["owner"] for row in rows)
    assert rows[0]["category"] in {"limit", "review"}


def test_duplicate_train_names_are_refused(client):
    t = {**_cycle(), "name": "A"}
    assert client.post("/plan", json={"trains": [t, t]}).status_code == 422


def test_normalise_measures_the_plants_own_exponent(client):
    rng = np.random.default_rng(0)
    recs = []
    for d in range(120):
        qp = 10.0 * (0.8 if rng.random() < 0.5 else 1.05)
        qf, qc = qp / 0.42, qp / 0.42 - qp
        dp = 1.3 * ((qf + qc) / 2 / 18.8) ** 1.75 * (1 + rng.normal(0, 0.005))
        recs.append(dict(day=d, feed_flow=qf, conc_flow=qc, perm_flow=qp, feed_pressure=60,
                         conc_pressure=60 - dp, perm_pressure=1.0, temperature_c=25.0,
                         feed_tds=35.0, perm_tds=0.15, dp=dp))
    r = client.post("/normalise", json={"records": recs})
    assert r.status_code == 200
    body = r.json()
    assert body["flow_exponent"] == pytest.approx(1.75, abs=0.05)
    dpn = [row["dp_norm"] for row in body["rows"]]
    assert np.std(dpn) < 0.01                         # flow swings removed


def test_normalise_refuses_kelvin(client):
    rec = dict(day=0, feed_flow=24, conc_flow=14, perm_flow=10, feed_pressure=60, conc_pressure=58.7,
               perm_pressure=1, temperature_c=298.0, feed_tds=35, perm_tds=0.15)
    r = client.post("/normalise", json={"records": [{**rec, "day": d} for d in range(20)]})
    assert r.status_code == 422


def test_examples_are_listed_and_build(client):
    keys = [e["key"] for e in client.get("/examples").json()]
    assert "plant" in keys
    for k in keys:
        assert client.get(f"/examples/{k}").status_code == 200
    assert client.get("/examples/nope").status_code == 422


def test_the_deployment_front_door_lists_the_ro_service():
    with TestClient(create_app()) as c:
        assert "RO fouling" in c.get("/").text
        assert c.get("/ro/health").status_code == 200
        assert c.get("/ro/").status_code == 200
