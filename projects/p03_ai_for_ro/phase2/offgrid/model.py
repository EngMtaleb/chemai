"""Shared helpers for the off-grid study. Uses the phase-2 model as is (imported read-only from the parent folder).

What is new here, and only here:
  * rule_s0      a copy of design_grid.rule_fast with one extra argument, the tank level at the start (phase 2 always
                 starts every year with an empty tank). The rule itself is unchanged (checked in check_model.py).
  * run_cyclic   the ten years 2015-2024 run as one continuous series, the tank carried from one year to the next,
                 and the start level on 1 Jan 2015 set equal to the end level on 31 Dec 2024 (a repeating state).
  * min_storage  the smallest store (tank in m3, or battery in kWh) that never runs dry for a given hourly net
                 inflow, in the same repeating state: the classic sequent-peak calculation used to size reservoirs.
  * battery      a store that charges only from sun the plant cannot use and supplies the plant whenever the rule
                 would buy from the grid (part b). It does not change what the plant makes.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                      # never write .pyc files into the phase-2 folder
HERE = Path(__file__).parent
PHASE2 = HERE.parent
sys.path.insert(0, str(PHASE2))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

import schedule as Sc                               # noqa: E402  phase-2 physics, data and rule
import design_grid as DG                            # noqa: E402  phase-2 fast rule
import costs as CO                                  # noqa: E402  phase-2 cost method

D = Sc.DEMAND                                       # m3/h (1,000 m3/day)
YEARS = list(range(2015, 2025))


# ------------------------------------------------------------------ data
def load():
    perf = Sc.load_perf([PHASE2 / "out" / "perf_table_part0.csv", PHASE2 / "out" / "perf_table_part1.csv"])
    other = Sc.other_energy(perf)
    pv_y, T_y = {}, {}
    for y in YEARS:
        pv_y[y], T_y[y] = Sc.load_sun_sea(y)
    pv = np.concatenate([pv_y[y] for y in YEARS])
    T = np.concatenate([T_y[y] for y in YEARS])
    year = np.concatenate([np.full(len(pv_y[y]), y) for y in YEARS])
    pvi = pd.read_csv(PHASE2 / "out" / "pv_hourly_per_kwp.csv", index_col=0, usecols=[0])
    t = pd.to_datetime(pvi.index, utc=True).tz_convert("Asia/Kuwait")
    t = t[(t.year >= 2015) & (t.year <= 2024)]
    assert len(t) == len(pv)
    return dict(perf=perf, other=other, pv_y=pv_y, T_y=T_y, pv=pv, T=T, year=year, time=t)


def curves(data, C):
    cv = Sc.curves(data["perf"], data["other"], C)
    keys, k = DG.groups(cv, data["T"])
    return cv, keys, k


def q_sun_series(pv_kw, cv, keys, k):
    """Hour by hour, the output the sun alone can power (the rule's q_sun), and the plant's max output."""
    q_sun, q_max = np.zeros(len(pv_kw)), np.zeros(len(pv_kw))
    for j, key in enumerate(keys):
        m = k == j
        c = cv[float(key)]
        q_sun[m] = np.interp(pv_kw[m], c["E"], c["q"])
        q_max[m] = c["q_max"]
    return q_sun, q_max


def energy_of(q, cv, keys, k):
    E = np.zeros(len(q))
    for j, key in enumerate(keys):
        m = k == j
        c = cv[float(key)]
        E[m] = np.interp(q[m], c["q"], c["E"])
    return E


# ------------------------------------------------------------------ the phase-2 rule, with a start level
def rule_s0(pv_kw, cv, S_max, keys, k, S0=0.0):
    """design_grid.rule_fast, line for line, except that the tank starts at S0 instead of 0. Returns q and the end level."""
    q_sun, q_max = q_sun_series(pv_kw, cv, keys, k)
    n = len(pv_kw)
    q, S = np.zeros(n), float(S0)
    pv_l, qs_l, qm_l = pv_kw.tolist(), q_sun.tolist(), q_max.tolist()
    out = [0.0] * n
    for t in range(n):
        if pv_l[t] > 0:
            qt = min(max(qs_l[t], D - S), qm_l[t], S_max - S + D)
        else:
            qt = max(0.0, D - S)
        out[t] = qt
        S += qt - D
    q[:] = out
    return q, S


def run_cyclic(pv_kw, cv, S_max, keys, k, tol=1e-6, it_max=50):
    """Ten years as one series; start level = end level (iterated from a full tank). Returns q, start level, iterations."""
    S0 = S_max
    for i in range(it_max):
        q, S_end = rule_s0(pv_kw, cv, S_max, keys, k, S0)
        if abs(S_end - S0) <= tol:
            return q, S0, i + 1
        S0 = min(max(S_end, 0.0), S_max)
    raise RuntimeError("no repeating state")


def year_scores(q, pv_kw, cv, keys, k, year, S0, batt=None):
    """Per year: the same quantities as design_grid.score_fast (energy, grid, sun used, panels' energy, short hours),
    with the tank starting at S0. With a battery (dict from battery_run), grid and sun used include it."""
    E = energy_of(q, cv, keys, k)
    used = np.minimum(E, pv_kw)
    grid = E - used
    S = S0 + np.cumsum(q - D)
    rows = []
    for y in YEARS:
        m = year == y
        r = {"year": y, "water_m3": float(q[m].sum()), "energy_kWh": float(E[m].sum()),
             "grid_kWh": float(grid[m].sum()), "solar_used_kWh": float(used[m].sum()), "pv_kWh": float(pv_kw[m].sum()),
             "short_hours": int((S[m] < -1e-6).sum()), "tank_min_m3": float(S[m].min()), "tank_max_m3": float(S[m].max())}
        if batt is not None:
            r.update(grid_kWh=float(batt["grid"][m].sum()), batt_in_kWh=float(batt["charge_in"][m].sum()),
                     batt_out_kWh=float(batt["discharge"][m].sum()),
                     curtailed_kWh=float((batt["surplus"] - batt["charge_in"])[m].sum()),
                     batt_min_kWh=float(batt["B"][m].min()), batt_full_hours=int((batt["B"][m] >= batt["B_max"] - 1e-6).sum()))
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ smallest store that never runs dry
def min_storage(x):
    """Sequent peak, repeating state: x = net inflow per hour (in - out). Returns the smallest capacity K such that a
    store filled greedily (spilling when full) never goes below zero when the series repeats; inf if sum(x) < 0."""
    x = np.asarray(x, float)
    if x.sum() < -1e-9:
        return np.inf
    y = -np.cumsum(np.r_[x, x])                     # cumulative draw over two cycles
    run_min = np.minimum.accumulate(np.r_[0.0, y])[1:]
    return float(max((y - run_min).max(), 0.0))


# ------------------------------------------------------------------ battery (part b)
def battery_run(surplus, deficit, B_max, P_max, eta, B0):
    """Hour by hour: charge from surplus sun (at most P_max kW in, stores eta x input, as deliverable kWh), discharge to
    cover the deficit (at most P_max kW out). What the battery cannot cover is bought from the grid."""
    n = len(surplus)
    B = float(B0)
    s_l, d_l = surplus.tolist(), deficit.tolist()
    Bs, cin, dis, grid = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    for t in range(n):
        s, d = s_l[t], d_l[t]
        if s > 0:
            c = min(s, P_max, (B_max - B) / eta)
            B += eta * c
            cin[t] = c
        if d > 0:
            o = min(d, P_max, B)
            B -= o
            dis[t] = o
            grid[t] = d - o
        Bs[t] = B
    return dict(B=np.array(Bs), charge_in=np.array(cin), discharge=np.array(dis), grid=np.array(grid), B_end=B,
                surplus=surplus, B_max=B_max)


def battery_cyclic(surplus, deficit, B_max, P_max, eta, tol=1e-6, it_max=50):
    B0 = B_max
    for i in range(it_max):
        r = battery_run(surplus, deficit, B_max, P_max, eta, B0)
        if abs(r["B_end"] - B0) <= tol:
            r["B0"], r["iterations"] = B0, i + 1
            return r
        B0 = min(max(r["B_end"], 0.0), B_max)
    raise RuntimeError("no repeating state")


def battery_min_energy(surplus, deficit, P_max, eta):
    """Smallest deliverable energy (kWh) for zero grid with power rating P_max (kW); inf if P_max < the largest deficit."""
    if deficit.max() > P_max + 1e-9:
        return np.inf
    return min_storage(eta * np.minimum(surplus, P_max) - deficit)


# ------------------------------------------------------------------ costs (phase-2 method)
def cost_inputs():
    import json
    return json.loads((PHASE2 / "cost_inputs.json").read_text())


def design_costs(capacity, kwp, tank_h, water_m3, grid_kWh=0.0, price=0.0, extra_annual=0.0):
    """USD per m3 by part, with costs.py's annual_costs (same inputs), plus grid electricity and any extra annual cost."""
    x = cost_inputs()
    g = pd.DataFrame({"capacity": [capacity], "kwp_per_m3d": [kwp], "tank_h": [tank_h]})
    a = CO.annual_costs(g, x)
    a["grid electricity"] = price * grid_kWh
    a["battery"] = extra_annual
    return (a / water_m3).iloc[0]


def crf(i, n):
    return CO.crf(i, n)
