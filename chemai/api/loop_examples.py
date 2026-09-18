"""Worked examples for the demo page.

Four loops whose answer is known because the fault was injected: a sticking
valve, a controller tuned too tight, a healthy loop, and a frozen transmitter.
They are generated from the project's own simulator at import time, with fixed
seeds, so the page has something to show without asking a visitor for data.

They are simulated, and the page says so. The real evidence lives in the
validation report, which was measured on 141 plant loops.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from chemai.config import ControlLoopConfig
from chemai.data import sample_loop_spec, simulate_loop

SAMPLES = 1200

EXAMPLES = {
    "stiction": {
        "title": "Sticking valve",
        "blurb": "A flow loop whose valve sticks and then jumps. The integral keeps "
                 "pushing while the stem does not move, so OP ramps into triangles.",
        "condition": "stiction", "loop_type": "F", "rng": 4, "seed": 3,
    },
    "tuning": {
        "title": "Controller tuned too tight",
        "blurb": "A level loop oscillating for a different reason: the gain is too "
                 "high. Smooth and sinusoidal - and reducing the gain fixes it, which "
                 "is why the owner is not maintenance.",
        "condition": "tuning_tight", "loop_type": "L", "rng": 1, "seed": 1,
    },
    "healthy": {
        "title": "Healthy loop",
        "blurb": "A well-tuned loop rejecting the same disturbances. Nothing to report, "
                 "which is what most of a plant should look like.",
        "condition": "healthy", "loop_type": "F", "rng": 2, "seed": 1,
    },
    "frozen": {
        "title": "Frozen transmitter",
        "blurb": "The screen shows the calmest loop in the plant. In fact the sensor "
                 "stopped updating and the integral is driving the process away.",
        "condition": "healthy", "loop_type": "F", "rng": 6, "seed": 8, "freeze_from": 0.55,
    },
}


@lru_cache(maxsize=None)
def build_example(key: str) -> dict:
    """One example loop as plain lists, ready to POST to /analyse/loop."""
    if key not in EXAMPLES:
        raise ValueError(f"unknown example: {key}")
    meta = EXAMPLES[key]
    cfg = ControlLoopConfig()
    spec = sample_loop_spec(meta["loop_type"], meta["condition"],
                            np.random.default_rng(meta["rng"]), cfg)
    run = simulate_loop(spec, SAMPLES, seed=meta["seed"], cfg=cfg)
    sp, pv, op = (run[c].to_numpy().copy() for c in ("sp", "pv", "op"))

    if "freeze_from" in meta:                      # the transmitter stops updating
        start = int(len(pv) * meta["freeze_from"])
        pv[start:] = pv[start]
        op[start:] = np.linspace(op[start], op[start] + 25, len(op) - start)

    return {"key": key, "title": meta["title"], "blurb": meta["blurb"],
            "loop_type": meta["loop_type"], "ts": float(spec.ts),
            "name": {"stiction": "FC101", "tuning": "LC201",
                     "healthy": "FC103", "frozen": "FC104"}[key],
            "sp": [round(float(v), 4) for v in sp],
            "pv": [round(float(v), 4) for v in pv],
            "op": [round(float(v), 4) for v in op]}


def example_index() -> list[dict]:
    return [{"key": k, "title": v["title"], "blurb": v["blurb"]} for k, v in EXAMPLES.items()]
