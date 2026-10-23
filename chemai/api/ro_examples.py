"""Worked examples for the RO demo page - simulated, so the cause is known.

Each one is built from the gap law in chemai.features.ro_physics with stated
growth, noise and gaps, not cut from the real deposit: the deposit is not
redistributed by this service (it is downloaded from Zenodo, with its
citation, by anyone who wants it).
"""
from __future__ import annotations

import numpy as np

from chemai.features.ro_physics import dp_from_blockage


def _cycle(days: int, phi_fn, seed: int, noise: float = 0.012, gaps: float = 0.3,
           lead: float = 0.2, rest: float = 0.15, n: float = 2.4) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    t = np.arange(0, days + 1, dtype=float)
    keep = rng.random(len(t)) > gaps
    keep[-1] = True
    t = t[keep]
    phi = np.clip(phi_fn(t), 0.0, 0.8)
    x = dp_from_blockage(phi, n, lead, rest) * (1 + rng.normal(0, noise, len(t)))
    return [float(v) for v in t], [round(float(v), 4) for v in x]


def _steady():
    return _cycle(45, lambda t: 0.05 + 0.0016 * t, seed=11)


def _accelerating():
    return _cycle(40, lambda t: 0.06 * np.exp(0.022 * t), seed=12)


def _relaxing():
    rng = np.random.default_rng(13)
    t = np.array(sorted(set(rng.choice(np.arange(2, 50), 30, replace=False)) | {49}), float)
    x = (1.07 + 0.06 * np.exp(-t / 12.0)) * (1 + rng.normal(0, 0.008, len(t)))
    return [float(v) for v in t], [round(float(v), 4) for v in x]


def _weak_cleaning():
    return _cycle(30, lambda t: 0.17 + 0.0012 * t, seed=14)


def _too_soon():
    return _cycle(6, lambda t: 0.05 + 0.002 * t, seed=15, gaps=0.0)


EXAMPLES = {
    "steady": dict(title="Steady fouling", kind="forecast",
                   blurb="Foulant builds at a steady rate after a cleaning; 45 days of data, "
                         "30 % of days missing."),
    "accelerating": dict(title="Biofilm taking off", kind="forecast",
                         blurb="Biofilm grows exponentially in the lead element: the pressure "
                               "drop accelerates."),
    "relaxing": dict(title="Still recovering", kind="forecast",
                     blurb="Loosened foulant keeps flushing out for weeks after a good cleaning: "
                           "the pressure drop is still falling."),
    "plant": dict(title="Weekly plan, three trains", kind="plan",
                  blurb="Three trains in different states. Supply needs two of them running, so "
                        "only one can be cleaned per week."),
}


def example_index() -> list[dict]:
    return [{"key": k, "title": v["title"], "blurb": v["blurb"], "kind": v["kind"]}
            for k, v in EXAMPLES.items()]


def build_example(key: str) -> dict:
    if key not in EXAMPLES:
        raise ValueError(f"unknown example '{key}'; choose one of {list(EXAMPLES)}")
    meta = EXAMPLES[key]
    if meta["kind"] == "forecast":
        day, x = {"steady": _steady, "accelerating": _accelerating, "relaxing": _relaxing}[key]()
        payload = {"cycle": {"name": meta["title"], "day": day, "dp_norm": x}, "review_level": 1.15}
    else:
        a_day, a_x = _accelerating()
        b_day, b_x = _weak_cleaning()
        c_day, c_x = _relaxing()
        payload = {"review_level": 1.15, "trains": [
            {"name": "Train A", "day": a_day, "dp_norm": a_x, "dp_vessel_bar": 1.9},
            {"name": "Train B", "day": b_day, "dp_norm": b_x, "pre_cleaning": 1.24,
             "post_cleaning": 1.20, "previous_post_cleaning": 1.05},
            {"name": "Train C", "day": c_day, "dp_norm": c_x},
        ]}
    return {"key": key, **meta, "payload": payload}
