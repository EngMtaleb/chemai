"""The when-to-stop service (Project 5)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from chemai.api.app import create_app
from chemai.api.stop_app import create_stop_app
from chemai.api.stop_examples import EXAMPLES
from chemai.models.stopping import sqrt_law


@pytest.fixture(scope="module")
def client():
    return TestClient(create_stop_app())


def synthetic_batch(minutes=120, fixed_slope=0.02):
    """Permeate 2.4 L/min; live cost rises linearly with permeate."""
    t = np.arange(0, minutes * 60, 2.0)
    q = np.full(len(t), 2.4)
    V_l = t / 60 * 2.4
    live = 8.0 + fixed_slope * V_l                          # kWh/m3, rises 0.02 per litre
    power = live * q * 60 / 1000
    return {"t": t.tolist(), "power_kw": power.tolist(), "permeate_lpm": q.tolist(),
            "conductivity": (4 + 0.08 * V_l).tolist()}


def test_page_health_examples(client):
    assert "When to Stop" in client.get("/").text
    h = client.get("/health").json()
    assert h["results"]["batch_rule_test"]["batches"] == 153
    assert len(h["limits"]) >= 4
    ex = client.get("/examples").json()
    assert set(ex) == {"batch_long", "batch_short", "ro_cycle", "uf_interval"}


def test_mounted_in_the_main_app():
    c = TestClient(create_app())
    assert c.get("/stop/health").status_code == 200
    assert "stop/" in c.get("/").text


def test_batch_stops_at_the_square_root_law(client):
    body = synthetic_batch()
    body.update(drain_kwh=0.05, brine_l=15.0, b=1.0, window_s=60, min_minutes=5)
    j = client.post("/batch", json=body).json()
    assert j["status"] == "stop"
    F = 0.05 + 1.0 * 15 / 1000                          # kWh
    v_star, _ = sqrt_law(F, 0.02 * 1000, 8.0)          # slope per m3
    assert j["stop"]["permeate_l"] == pytest.approx(v_star * 1000, rel=0.05)
    assert j["regret_pct"] < 0.05
    assert j["law"]["v_star_l"] == pytest.approx(v_star * 1000, rel=0.05)
    assert len(j["curve"]["minutes"]) <= 401


def test_batch_says_run_longer_when_the_cost_still_falls(client):
    body = synthetic_batch(minutes=12)
    body.update(drain_kwh=0.05, brine_l=15.0, b=10.0)
    j = client.post("/batch", json=body).json()
    assert j["status"] == "run_longer"
    assert j["stop"] is None
    assert any("beyond" in n for n in j["notes"])


def test_real_examples_behave_as_in_the_record(client):
    e = EXAMPLES["batch_long"]
    j = client.post("/batch", json={k: e[k] for k in ("t", "power_kw", "permeate_lpm", "conductivity",
                                                      "drain_kwh", "brine_l")} | {"b": 1.0}).json()
    assert j["status"] == "stop"
    assert j["regret_pct"] < 0.5
    assert 14 < j["stop"]["conductivity"] < 22
    s = EXAMPLES["batch_short"]
    j2 = client.post("/batch", json={k: s[k] for k in ("t", "power_kw", "permeate_lpm", "conductivity",
                                                       "drain_kwh", "brine_l")} | {"b": 1.0}).json()
    assert j2["status"] == "run_longer"


def test_bad_logs_are_refused(client):
    body = synthetic_batch()
    bad = dict(body, t=body["t"][::-1])
    assert client.post("/batch", json=bad).status_code == 422
    short = {k: v[:10] for k, v in body.items()}
    assert client.post("/batch", json=short).status_code == 422
    mismatch = dict(body, power_kw=body["power_kw"][:-1])
    assert client.post("/batch", json=mismatch).status_code == 422
    neg = dict(body, power_kw=[-1.0] + body["power_kw"][1:])
    assert client.post("/batch", json=neg).status_code == 422


def test_interval_on_the_real_cycles(client):
    ro = EXAMPLES["ro_cycle"]
    j = client.post("/interval", json={"day": ro["day"], "live_cost": ro["live_cost"],
                                       "fixed_cost": ro["fixed_cost"]}).json()
    assert j["rule_day"] is None                         # energy never paid within the cycle
    assert j["t_star_days"] > ro["actual_days"]
    uf = EXAMPLES["uf_interval"]
    j2 = client.post("/interval", json={"day": uf["day"], "live_cost": uf["live_cost"],
                                        "fixed_cost": uf["fixed_cost"]}).json()
    assert 1.0 < j2["t_star_days"] / uf["actual_days"] < 1.6


def test_interval_with_no_rise(client):
    j = client.post("/interval", json={"day": list(range(1, 40)), "live_cost": [0.0] * 39,
                                       "fixed_cost": 1.0}).json()
    assert j["t_star_days"] is None
    assert "never rose" in j["verdict"]


def test_law(client):
    j = client.get("/law", params={"fixed": 2.0, "slope": 0.001}).json()
    assert j["best_length"] == pytest.approx(np.sqrt(4000))
    j0 = client.get("/law", params={"fixed": 2.0, "slope": 0.0}).json()
    assert j0["best_length"] is None
