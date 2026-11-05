"""Phase 2 costs: which design is cheapest, at what grid electricity price.

Counts only the costs the design choice changes, per 1,000 m3/day of demand (USD per year, then per m3 delivered):
  RO plant capital    K_ro per m3/day x [1 + share x (capacity - 1)]   (share: how much of the plant grows with capacity)
  RO upkeep           membranes, cartridge filters and maintenance, per m3 at design output, x capacity
  panels              (K_pv x CRF + O&M) per kWp
  tank                K_tank per m3 x volume x (CRF + O&M fraction)
  grid electricity    price x kWh bought
Left out because every design pays them alike: chemicals, staff, waste disposal, monitoring, other O&M.

    python costs.py <inputs.json>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEMAND_M3D = 1000.0


def crf(i, n):
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def annual_costs(g, x):
    i = x["discount_rate"]
    out = pd.DataFrame(index=g.index)
    out["RO plant"] = crf(i, x["plant_life_y"]) * x["ro_capex_usd_per_m3d"] * DEMAND_M3D * (1 + x["ro_capacity_share"] * (g.capacity - 1))
    out["RO upkeep"] = x["ro_capacity_om_usd_per_m3"] * DEMAND_M3D * 365 * (1 + x.get("upkeep_grows", 1.0) * (g.capacity - 1))
    out["panels"] = (crf(i, x["pv_life_y"]) * x["pv_capex_usd_per_kwp"] + x["pv_om_usd_per_kwp_y"]) * g.kwp_per_m3d * DEMAND_M3D
    out["tank"] = (crf(i, x["tank_life_y"]) + x["tank_om_frac_y"]) * x["tank_capex_usd_per_m3"] * g.tank_h * DEMAND_M3D / 24
    return out


def per_m3(g, x, price):
    a = annual_costs(g, x)
    a["grid electricity"] = price * g.grid_kWh
    return a.div(g.water_m3, axis=0)


def designs(path=HERE / "out" / "design_grid.csv"):
    t = pd.read_csv(path)
    g = t.groupby(["capacity", "kwp_per_m3d", "tank_h"]).mean(numeric_only=True).reset_index()
    g["solar_share_pct"] = 100 * g.solar_used_kWh / g.energy_kWh
    g["grid_kWh_per_m3"] = g.grid_kWh / g.water_m3
    return g


def find(g, capacity, kwp, tank_h):
    return g[(g.capacity == capacity) & (g.kwp_per_m3d == kwp) & (g.tank_h == tank_h)].index[0]


def best_by_price(g, x, prices):
    base = find(g, 1.0, 0.0, 0)
    rows = []
    for p in prices:
        c = per_m3(g, x, p).sum(axis=1)
        b = c.idxmin()
        pv_only = c[(g.capacity == 1.0) & (g.tank_h == 0)].idxmin()
        rows.append({"grid price, USD/kWh": p, "capacity": g.capacity[b], "kWp per m3/d": g.kwp_per_m3d[b], "tank, h": g.tank_h[b],
                     "solar share %": g.solar_share_pct[b], "cost, USD/m3": c[b], "grid only, USD/m3": c[base],
                     "panels only: kWp": g.kwp_per_m3d[pv_only], "panels only: solar %": g.solar_share_pct[pv_only],
                     "panels only: USD/m3": c[pv_only]})
    return pd.DataFrame(rows)


def breakeven(g, x, a, b):
    """Grid price at which design a costs the same as design b."""
    fixed = annual_costs(g, x).sum(axis=1) / g.water_m3
    return float((fixed[a] - fixed[b]) / (g.grid_kWh_per_m3[b] - g.grid_kWh_per_m3[a]))


SCENARIOS = {"base": {}, "only the RO system grows (30 %)": {"ro_capacity_share": 0.30},
             "small steel tank, 750 USD/m3": {"tank_capex_usd_per_m3": 750.0}, "discount rate 5 %": {"discount_rate": 0.05},
             "RO plant 1,200 USD per m3/d": {"ro_capex_usd_per_m3d": 1200.0}, "RO plant 1,800 USD per m3/d": {"ro_capex_usd_per_m3d": 1800.0},
             "spare trains already there": {"ro_capacity_share": 0.0, "upkeep_grows": 0.0}}


def main():
    x0 = json.loads(Path(sys.argv[1]).read_text())
    g = designs()
    pd.set_option("display.width", 250)
    prices = np.round(np.arange(0.02, 0.401, 0.02), 3)
    print(best_by_price(g, x0, prices).round(3).to_string(index=False))
    B, B_steady, grid_only = find(g, 2.0, 0.8, 12), find(g, 1.0, 0.8, 0), find(g, 1.0, 0.0, 0)
    rows = []
    for name, change in SCENARIOS.items():
        x = {**x0, **change}
        best = best_by_price(g, x, [0.05, 0.10, 0.15, 0.20, 0.30]).set_index("grid price, USD/kWh")
        rows.append({"case": name,
                     "example design pays above (vs same panels, no tank), USD/kWh": breakeven(g, x, B, B_steady),
                     "panels 0.2 kWp pay above, USD/kWh": breakeven(g, x, find(g, 1.0, 0.2, 0), grid_only),
                     **{f"best at {p:.2f}": f"{best.loc[p, 'capacity']:g}x, {best.loc[p, 'kWp per m3/d']:g} kWp, {best.loc[p, 'tank, h']:g} h, "
                                            f"{best.loc[p, 'solar share %']:.0f} %" for p in best.index}})
    s = pd.DataFrame(rows).set_index("case")
    print(s.round(3).T.to_string())
    s.to_csv(HERE / "out" / "cost_scenarios.csv")
    a = annual_costs(g, x0)
    a["grid electricity at 0.10"] = 0.10 * g.grid_kWh
    parts = a.div(g.water_m3, axis=0).loc[[grid_only, B_steady, B]]
    parts.index = ["grid only", "steady + 0.8 kWp", "example design"]
    print(parts.round(3).to_string())


if __name__ == "__main__":
    main()
