"""Every design on a grid, every year 2015-2024, follow-the-sun rule (no forecast): the water, energy and grid use
that the cost model needs. Same physics and rule as schedule.py, written to run fast (hours grouped by sea temperature).

    python design_grid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import schedule as Sc  # noqa: E402

CAPACITY = (1.0, 1.25, 1.5, 1.75, 2.0)
PV = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)
TANK = (0, 3, 6, 9, 12, 18, 24)


def groups(cv, T):
    keys = np.array(sorted(cv))
    return keys, np.abs(T[:, None] - keys[None, :]).argmin(1)


def rule_fast(pv_kw, cv, S_max, keys, k):
    n = len(pv_kw)
    q_sun, q_max = np.zeros(n), np.zeros(n)
    for j, key in enumerate(keys):
        m = k == j
        c = cv[float(key)]
        q_sun[m] = np.interp(pv_kw[m], c["E"], c["q"])
        q_max[m] = c["q_max"]
    q, S, D = np.zeros(n), 0.0, Sc.DEMAND
    for t in range(n):
        if pv_kw[t] > 0:
            qt = min(max(q_sun[t], D - S), q_max[t], S_max - S + D)
        else:
            qt = max(0.0, D - S)
        q[t] = qt
        S += qt - D
    return q


def score_fast(q, pv_kw, cv, keys, k):
    E, cp, n_on = np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q), int)
    for j, key in enumerate(keys):
        m = k == j
        c = cv[float(key)]
        E[m] = np.interp(q[m], c["q"], c["E"])
        cp[m] = np.interp(q[m], c["q"], c["cp"])
        lim = Sc.TRAINS * c["f_min"] * c["Q_tr"]
        n_on[m] = np.where(q[m] <= 1e-9, 0, np.where(q[m] >= lim - 1e-9, Sc.TRAINS,
                                                     np.ceil(q[m] / (c["f_min"] * c["Q_tr"]) - 1e-9))).astype(int)
    used = np.minimum(E, pv_kw)
    S = np.cumsum(q - Sc.DEMAND)
    water = q.sum()
    return {"water_m3": float(water), "energy_kWh": float(E.sum()), "grid_kWh": float((E - used).sum()),
            "solar_used_kWh": float(used.sum()), "pv_kWh": float(pv_kw.sum()), "short_hours": int((S < -1e-6).sum()),
            "permeate_mgL_avg": float((cp * q).sum() / water), "permeate_mgL_max": float(cp[q > 0].max()),
            "train_starts_per_day": float(np.maximum(np.diff(n_on), 0).sum() / (len(q) / 24)),
            "tank_max_m3": float(S.max())}


def main():
    perf = Sc.load_perf([HERE / "out" / "perf_table_part0.csv", HERE / "out" / "perf_table_part1.csv"])
    other = Sc.other_energy(perf)
    rows = []
    for year in range(2015, 2025):
        pv, T = Sc.load_sun_sea(year)
        for C in CAPACITY:
            cv = Sc.curves(perf, other, C)
            keys, k = groups(cv, T)
            for P in PV:
                for S_h in TANK:
                    q = rule_fast(pv * P * 1000.0, cv, S_h * Sc.DEMAND, keys, k)
                    s = score_fast(q, pv * P * 1000.0, cv, keys, k)
                    s.update(year=year, capacity=C, kwp_per_m3d=P, tank_h=S_h)
                    rows.append(s)
        print(year, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "out" / "design_grid.csv", index=False)
    print(len(t), "rows; short hours total", int(t.short_hours.sum()))


if __name__ == "__main__":
    main()
