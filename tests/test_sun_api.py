"""Sun-and-moon API (Project 4). Stateless: a table of numbers and a test that runs on
the caller's data, so these run anywhere."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from chemai.api.app import create_app
from chemai.api.sun_app import create_sun_app
from chemai.api.sun_designs import DESIGNS


@pytest.fixture(scope="module")
def client():
    with TestClient(create_sun_app()) as c:
        yield c


def _log(wave="none", runs=10, days=6, step_h=0.5, seed=3, amp=0.25, as_text=True):
    """Runs of a fouling-like signal: slow drift, red noise, and an optional wave.

    Runs start exactly 9 days apart - the beat period of a 24-h wave and the 27-h
    placebo, the worst case for a daily wave leaking into a placebo."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(runs):
        t0 = pd.Timestamp("2014-01-06") + pd.Timedelta(days=9 * k)
        h = np.arange(0, days * 24, step_h)
        t = t0 + pd.to_timedelta(h, unit="h")
        th = (t - pd.Timestamp("2000-01-01")) / pd.Timedelta(hours=1)
        e = np.zeros(len(h))
        for i in range(1, len(h)):
            e[i] = 0.8 * e[i - 1] + 0.6 * rng.standard_normal()
        y = 2 + 0.002 * h + 0.25 * e
        if wave == "tide":
            y = y + amp * np.cos(2 * np.pi * np.asarray(th) / 12.4206 + 1)
        elif wave == "clock":
            y = y + amp * np.cos(2 * np.pi * (np.asarray(t.hour + t.minute / 60) - 14) / 24)
        out += [{"t": ti.isoformat() if as_text else ti, "value": round(float(v), 4), "run": f"run {k + 1}"}
                for ti, v in zip(t, y)]
    return out


def test_health_serves_results_and_limits(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["results"]["tide_fouling"]["p"] == pytest.approx(0.244)
    assert any("One pilot" in s for s in body["limits"])
    assert "Zenodo 4630769" in body["data"]


def test_the_page_is_served_without_any_cdn(client):
    html = client.get("/").text
    assert "Desalination on the Sun" in html and "Test your own log" in html
    assert "cdn" not in html.lower() and "<script src" not in html


def test_every_design_is_listed(client):
    d = client.get("/designs").json()
    assert len(d["rows"]) == len(DESIGNS) == 624
    assert d["sec_kwh_m3"] == pytest.approx(3.26)


def test_the_90_percent_design_matches_the_declared_result(client):
    d = client.get("/design", params=dict(weather="sunny", capacity=3, storage_h=16, pv_kwp=0.7,
                                         tidal_kw=0)).json()
    assert d["renewable_fraction"] == pytest.approx(0.930, abs=0.002)
    assert d["grid_kwh_m3"] == pytest.approx(0.23, abs=0.01)
    assert d["per_1000_m3_day"]["pv_kwp"] == 700 and d["per_1000_m3_day"]["tank_m3"] == 667
    assert d["per_1000_m3_day"]["grid_mwh_year_without"] == pytest.approx(1189.9)
    assert d["rule_fraction"] is not None


def test_a_tidal_design_says_the_rule_was_not_run(client):
    d = client.get("/design", params=dict(weather="sunny", capacity=2, storage_h=8, pv_kwp=0.9,
                                         tidal_kw=0.1)).json()
    assert d["renewable_fraction"] >= 0.9 and d["rule_fraction"] is None


def test_a_design_off_the_grid_is_refused(client):
    r = client.get("/design", params=dict(capacity=3.3))
    assert r.status_code == 404 and "/designs" in r.json()["detail"]


def test_a_planted_tide_is_found_and_the_placebos_stay_quiet(client):
    r = client.post("/test", json={"records": _log("tide"), "kind": "tide", "n_surrogates": 199})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["verdict"] == "locked" and d["p_value"] <= 0.01
    assert d["placebo_p_value"] > 0.01
    assert d["amplitudes"]["M2"] > 2 * max(d["amplitudes"]["N2"], d["amplitudes"]["O1"])


def test_a_daily_wave_is_not_mistaken_for_the_tide(client):
    """Time of day sits in the tide test's base model - a solar plant's daily rhythm
    must not come out as a tide."""
    d = client.post("/test", json={"records": _log("clock"), "kind": "tide", "n_surrogates": 199}).json()
    assert d["status"] == "ok" and d["verdict"] == "not locked"


def test_the_clock_test_finds_the_daily_wave(client):
    d = client.post("/test", json={"records": _log("clock"), "kind": "clock", "n_surrogates": 199}).json()
    assert d["verdict"] == "locked" and set(d["amplitudes"]) == {"D24", "D12", "D8", "D6"}
    assert d["amplitudes"]["D24"] > d["amplitudes"]["D12"]


def test_a_log_with_no_wave_is_not_locked(client):
    d = client.post("/test", json={"records": _log("none", seed=5), "kind": "tide", "n_surrogates": 199}).json()
    assert d["verdict"] == "not locked" and d["p_value"] > 0.01


def test_a_short_log_is_answered_insufficient_not_guessed(client):
    recs = _log("tide", runs=1, days=2.5)                  # 120 points over 60 hours
    d = client.post("/test", json={"records": recs, "kind": "tide", "n_surrogates": 99}).json()
    assert d["status"] == "insufficient" and d["verdict"] is None and d["notes"]


def test_without_run_labels_a_long_gap_starts_a_new_run(client):
    recs = [{k: v for k, v in r.items() if k != "run"} for r in _log("tide")]
    d = client.post("/test", json={"records": recs, "kind": "tide", "n_surrogates": 199}).json()
    assert d["status"] == "ok" and d["n_blocks"] == 10 and d["verdict"] == "locked"


def test_offsets_and_space_separated_stamps_are_read_as_wall_time(client):
    plain = _log("tide")
    spaced = [dict(r, t=r["t"].replace("T", " ")) for r in plain]
    offset = [dict(r, t=r["t"] + "+03:00") for r in plain]
    a, b, c = (client.post("/test", json={"records": x, "kind": "tide", "n_surrogates": 99}).json()
               for x in (plain, spaced, offset))
    assert a["partial_r2"] == pytest.approx(b["partial_r2"]) == pytest.approx(c["partial_r2"])


def test_bad_logs_are_refused(client):
    recs = _log("none")
    assert client.post("/test", json={"records": recs[:50]}).status_code == 422
    dup = recs[:200] + recs[:1]
    assert client.post("/test", json={"records": dup}).status_code == 422
    assert client.post("/test", json={"records": recs, "kind": "moon"}).status_code == 422
    assert client.post("/test", json={"records": recs, "n_surrogates": 5000}).status_code == 422


def test_the_service_is_mounted_on_the_main_deployment():
    with TestClient(create_app()) as c:
        assert c.get("/sun/health").json()["status"] == "ok"
        assert 'href="sun/"' in c.get("/").text
