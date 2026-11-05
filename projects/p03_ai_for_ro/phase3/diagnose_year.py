"""Why the rule strategy's grid share of the fouling penalty is above 1 (median day) and below 0 (90th percentile).
Descriptive: re-computes the 2024 rule year for three states and splits the year into regimes. Writes diag_year.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import p4lib as L  # noqa: E402

SUN, SEA = Path(sys.argv[1]), Path(sys.argv[2])
sun, sea = L.load_sun_sea(SUN, SEA)
H = L.pv_per_kwp(sun)
L.fit_membrane()
perf0, hot0 = L.perf_table()
OTHER = L.other_energy(perf0)
CAP, KWP, TANK_H = 2.0, 0.8, 12
pv, T24, idx = L.sun_sea(H, sea, 2024); pv_kw = pv * KWP * 1000
STATES = {"clean": (0.0, 0.0), "median day": (0.1609, 0.0), "both at 90th percentile": (0.3412, 0.1295)}
out = {}
for name, (d, f) in STATES.items():
    perf, hot = (perf0, hot0) if name == "clean" else L.perf_table(d, f, hot=hot0)
    cv = L.curves(perf, CAP, OTHER)
    S_max = TANK_H * L.DEMAND
    q = L.run_rule(pv_kw, T24, cv, S_max)
    E, cp, n_on = L.hourly(q, T24, cv)
    keys, k = L.groups(cv, T24)
    qmax = np.array([cv[float(keys[j])]["q_max"] for j in k])
    S = np.cumsum(q - L.DEMAND)
    S_before = np.r_[0.0, S[:-1]]
    used = np.minimum(E, pv_kw)
    day = pv_kw > 0
    # regimes for daylight hours: what bound the production this hour
    at_cap = day & (q >= qmax - 1e-6)
    tank_full = day & ~at_cap & (np.abs(q - (S_max - S_before + L.DEMAND)) < 1e-6)
    topped = day & (E > pv_kw + 1e-6)                    # the grid topped up a daylight hour (tank low, weak sun)
    sun_limited = day & ~at_cap & ~tank_full & ~topped
    reg = {"capacity": at_cap, "tank full": tank_full, "grid top-up": topped, "sun-limited": sun_limited, "night": ~day}
    per_T = {float(T): {"f_min": round(c["f_min"], 3), "q_max": round(c["q_max"], 2), "R": c["R"]} for T, c in cv.items()}
    out[name] = {
        "water m3": round(float(q.sum()), 1), "tank at year end m3": round(float(S[-1]), 1),
        "energy kWh": round(float(E.sum()), 1), "sun used kWh": round(float(used.sum()), 1), "grid kWh": round(float((E - used).sum()), 1),
        "sun available kWh": round(float(pv_kw.sum()), 1),
        "hours": {r: int(m.sum()) for r, m in reg.items()},
        "water by regime m3": {r: round(float(q[m].sum()), 1) for r, m in reg.items()},
        "sun used by regime kWh": {r: round(float(used[m].sum()), 1) for r, m in reg.items()},
        "grid by regime kWh": {r: round(float((E - used)[m].sum()), 1) for r, m in reg.items()},
        "window by sea temperature": per_T,
    }
    print(name, json.dumps({k: v for k, v in out[name].items() if k != "window by sea temperature"}), flush=True)
(HERE / "out" / "diag_year.json").write_text(json.dumps(out, indent=1))
for name in STATES:
    print(name, {T: (w["f_min"], w["q_max"]) for T, w in out[name]["window by sea temperature"].items()})
