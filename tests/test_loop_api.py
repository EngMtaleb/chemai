"""Loop-diagnosis API. No model, no data files - the service is stateless,
so these run anywhere."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from chemai.api.loop_app import create_loop_app
from chemai.config import ControlLoopConfig
from chemai.data import sample_loop_spec, simulate_loop

cfg = ControlLoopConfig()


@pytest.fixture(scope="module")
def client():
    with TestClient(create_loop_app(cfg)) as c:
        yield c


@pytest.fixture(scope="module")
def sticky():
    spec = sample_loop_spec("F", "stiction", np.random.default_rng(4), cfg)
    d = simulate_loop(spec, 2000, seed=3, cfg=cfg)
    return {"name": "FC101", "loop_type": "F", "sp": d.sp.tolist(),
            "pv": d.pv.tolist(), "op": d.op.tolist()}


def test_health_serves_the_limits(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "stiction" in body["diagnoses"]
    assert any("valve position" in limit for limit in body["limits"])
    assert any("samples" in limit for limit in body["limits"])


def test_a_sticky_valve_comes_back_with_an_owner(client, sticky):
    r = client.post("/analyse/loop", json={"loop": sticky, "ts": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["diagnosis"] == "stiction" and body["owner"] == "maintenance"
    assert body["confidence_band"] in {"high", "medium", "low"}
    assert "regularity" in body["evidence"]              # never a bare verdict


def test_a_short_record_is_refused_not_guessed(client):
    bad = {"name": "x", "sp": [1, 2], "pv": [1, 2], "op": [1, 2]}
    assert client.post("/analyse/loop", json={"loop": bad}).status_code == 422


def test_mismatched_lengths_are_refused(client):
    bad = {"name": "x", "sp": [0.0] * 50, "pv": [0.0] * 50, "op": [0.0] * 49}
    assert client.post("/analyse/loop", json={"loop": bad}).status_code == 422


def test_unknown_loop_type_is_refused(client, sticky):
    loop = {**sticky, "loop_type": "Z"}
    assert client.post("/analyse/loop", json={"loop": loop}).status_code == 422


def test_plant_endpoint_reports_structure_and_victims(client):
    n, period = 2000, 40.0
    t = np.arange(n)
    rng = np.random.default_rng(0)
    loops = []
    for k, (amp, noise) in enumerate([(1.0, 0.1), (0.7, 0.2)], start=1):
        pv = 50 + amp * np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)
        loops.append({"name": f"FC10{k}", "loop_type": "F", "sp": [50.0] * n,
                      "pv": pv.tolist(),
                      "op": (50 + np.cos(2 * np.pi * t / period)).tolist()})
    r = client.post("/analyse/plant", json={"loops": loops, "ts": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["simultaneous"] is True
    assert body["propagation"]["source_candidate"] in {"FC101", "FC102"}
    assert "not proof" in body["propagation"]["note"]
    assert len(body["loops"]) == 2 and len(body["ranking"]) == 2


def test_missing_location_weights_are_declared(client, sticky):
    r = client.post("/analyse/plant", json={"loops": [sticky], "ts": 1.0})
    assert "WITHOUT plant location weights" in r.json()["ranking_note"]


def test_location_weights_must_be_one_two_or_three(client, sticky):
    r = client.post("/analyse/plant",
                    json={"loops": [sticky], "ts": 1.0, "location_weights": {"FC101": 7}})
    assert r.status_code == 422


def test_duplicate_loop_names_are_refused(client, sticky):
    r = client.post("/analyse/plant", json={"loops": [sticky, sticky], "ts": 1.0})
    assert r.status_code == 422


def test_the_soft_sensor_service_still_mounts_it(sticky):
    """Both projects ride on one deployment: /loops must answer under it."""
    from chemai.api.app import create_app
    with TestClient(create_app()) as c:
        assert c.get("/loops/health").json()["status"] == "ok"


# ------------------------------------------------------------- demo page

def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Control Loop Diagnosis" in r.text
    assert "analyse/loop" in r.text                       # it calls the real endpoint
    assert "<script" in r.text and "cdn" not in r.text.lower()   # no third-party script


def test_examples_are_listed_and_fetchable(client):
    listing = client.get("/examples").json()
    assert {e["key"] for e in listing} == {"stiction", "tuning", "healthy", "frozen"}
    for entry in listing:
        loop = client.get(f"/examples/{entry['key']}").json()
        assert len(loop["pv"]) == len(loop["sp"]) == len(loop["op"]) > 100
        assert loop["ts"] > 0 and loop["loop_type"] in set("FPLTQ")


def test_an_unknown_example_is_refused(client):
    assert client.get("/examples/nonsense").status_code == 422


@pytest.mark.parametrize("key,expected", [("stiction", "stiction"), ("tuning", "tuning"),
                                          ("frozen", "frozen_sensor")])
def test_each_example_shows_what_it_promises(client, key, expected):
    """The page's worked examples must actually demonstrate their fault -
    otherwise the demo teaches the wrong thing."""
    loop = client.get(f"/examples/{key}").json()
    body = {"loop": {"name": loop["name"], "loop_type": loop["loop_type"],
                     "sp": loop["sp"], "pv": loop["pv"], "op": loop["op"]}, "ts": loop["ts"]}
    assert client.post("/analyse/loop", json=body).json()["diagnosis"] == expected
