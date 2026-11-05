"""Compare, with one cost method (costs.py's annual_costs + cost_inputs.json):
  the deck's design (2x, 0.8 kWp per m3/day, 12 h) buying from the grid, at the grid prices phase 2 uses
  the cheapest water-only off-grid design (sweep_water.py)
  the cheapest off-grid design with a battery (battery.py)
plus the grid price at which the deck's design costs as much as each off-grid option, and sensitivities (costs.py's
scenarios, re-optimised for each option; the older NREL battery price).

    python compare.py      -> results.json (the numbers for one deck page), out/compare.json, out/compare_table.csv
"""
import json

import numpy as np
import pandas as pd

import model as M

OUT = M.HERE / "out"
X0 = M.cost_inputs()
water = json.loads((OUT / "water_result.json").read_text())
batt = json.loads((OUT / "battery_result.json").read_text())
fine = pd.read_csv(OUT / "water_fine.csv")
screen = pd.read_csv(OUT / "battery_screen.csv")
N_HOURS = 87669
WATER_Y = M.D * N_HOURS / 10

# ---------------------------------------------------------------- the deck's design with the grid (phase 2, as costs.py)
g = M.CO.designs(M.PHASE2 / "out" / "design_grid.csv")
deck_i = M.CO.find(g, 2.0, 0.8, 12)
grid_csv = pd.read_csv(M.PHASE2 / "out" / "design_grid.csv")
deck_rows = grid_csv[(grid_csv.capacity == 2.0) & (grid_csv.kwp_per_m3d == 0.8) & (grid_csv.tank_h == 12)]
deck_share = 100 * (1 - deck_rows.grid_kWh / deck_rows.energy_kWh)
deck_unused = 100 * (1 - deck_rows.solar_used_kWh.sum() / deck_rows.pv_kWh.sum())
PRICES_MAIN = [0.05, 0.10, 0.15, 0.20, 0.30]                                       # costs.py main(): scenarios
PRICES_ALL = sorted({float(p) for p in np.round(np.arange(0.02, 0.401, 0.02), 3)} | set(PRICES_MAIN))   # + best_by_price
deck_fixed = float(M.CO.annual_costs(g, X0).sum(axis=1)[deck_i] / g.water_m3[deck_i])
deck_grid_kwh_m3 = float(g.grid_kWh_per_m3[deck_i])
deck_cost = {f"{p:.2f}": float(M.CO.per_m3(g, X0, p).sum(axis=1)[deck_i]) for p in PRICES_ALL}
deck_parts_010 = {k: float(v) for k, v in M.CO.per_m3(g, X0, 0.10).loc[deck_i].items()}

# ---------------------------------------------------------------- the two off-grid options
wc = water["cheapest"]
bc = batt["cheapest"]


def breakeven(cost_offgrid, fixed=deck_fixed, grid_kwh_m3=deck_grid_kwh_m3):
    return (cost_offgrid - fixed) / grid_kwh_m3


options = [
    dict(option="deck design, grid makes up the rest", design=dict(capacity=2.0, kwp_per_m3d=0.8, tank_h=12),
         sun_share_pct=float(deck_share.mean()), sun_share_pct_by_year_range=[float(deck_share.min()), float(deck_share.max())],
         unused_panel_energy_pct=float(deck_unused), battery_kWh=None, grid_kWh_per_m3=deck_grid_kwh_m3,
         usd_per_m3_by_grid_price=deck_cost, usd_per_m3_without_grid=deck_fixed),
    dict(option="off-grid, water storage only (cheapest)", design=dict(capacity=wc["capacity"], kwp_per_m3d=wc["kwp_per_m3d"],
                                                                        tank_h=wc["tank_h"]),
         sun_share_pct=wc["check"]["sun_share_pct_mean"], unused_panel_energy_pct=wc["check"]["unused_panel_energy_pct"],
         battery_kWh=None, usd_per_m3=wc["usd_per_m3"], usd_per_m3_parts=wc["usd_per_m3_parts"],
         deck_design_costs_the_same_at_grid_price=breakeven(wc["usd_per_m3"])),
    dict(option="off-grid with a battery (cheapest found)", design=dict(capacity=bc["capacity"], kwp_per_m3d=bc["kwp_per_m3d"],
                                                                         tank_h=bc["tank_h"]),
         sun_share_pct=bc["sun_share_pct"], unused_panel_energy_pct=bc["unused_panel_energy_pct"],
         battery_kWh=bc["check"]["battery_kWh"], battery_kW=bc["check"]["battery_kW"],
         battery_smallest_kWh_power_not_limiting=bc["E_min_kWh"], usd_per_m3=bc["usd_per_m3"],
         usd_per_m3_parts=bc["usd_per_m3_parts"], deck_design_costs_the_same_at_grid_price=breakeven(bc["usd_per_m3"]),
         battery_usd_per_m3_at_2023_4h_price=bc["sensitivity_4h_2023"]["usd_per_m3"]),
]
b2 = batt["cheapest_at_2x"]
options.append(dict(option="off-grid with a battery, keeping the deck's 2× plant (cheapest found)",
                    design=dict(capacity=b2["capacity"], kwp_per_m3d=b2["kwp_per_m3d"], tank_h=b2["tank_h"]),
                    sun_share_pct=b2["sun_share_pct"], unused_panel_energy_pct=b2["unused_panel_energy_pct"],
                    battery_kWh=b2["check"]["battery_kWh"], battery_kW=b2["check"]["battery_kW"],
                    battery_smallest_kWh_power_not_limiting=b2["E_min_kWh"], usd_per_m3=b2["usd_per_m3"],
                    usd_per_m3_parts=b2["usd_per_m3_parts"], deck_design_costs_the_same_at_grid_price=breakeven(b2["usd_per_m3"]),
                    battery_usd_per_m3_at_2023_4h_price=b2["sensitivity_4h_2023"]["usd_per_m3"]))

# ---------------------------------------------------------------- sensitivities, each option re-optimised
SCEN = {k: v for k, v in M.CO.SCENARIOS.items() if k != "spare trains already there"}
SCEN["plant costs in 2024 USD (x 1.31, phase-2 notebook)"] = {"ro_capex_usd_per_m3d": 1500.0 * 1.31, "ro_capacity_om_usd_per_m3": 0.07 * 1.31}
feas = fine[np.isfinite(fine.min_tank_h)].copy()
ok = screen[screen.usd_per_m3.notna() & (screen.E_kWh > 0.5)].drop_duplicates(["capacity", "kwp_per_m3d", "tank_h"]).copy()


def water_best(x):
    gg = feas.rename(columns={"min_tank_h": "tank_h"})
    c = M.CO.annual_costs(gg, x).sum(axis=1) / WATER_Y
    i = c.idxmin()
    edge = bool(gg.capacity[i] in (fine.capacity.min(), fine.capacity.max()) or gg.kwp_per_m3d[i] in (fine.kwp_per_m3d.min(), fine.kwp_per_m3d.max()))
    return float(c[i]), dict(capacity=float(gg.capacity[i]), kwp_per_m3d=float(gg.kwp_per_m3d[i]), tank_h=float(gg.tank_h[i]),
                             at_edge_of_search=edge)


def batt_best(x, e_col="E_kWh", capex=None, fom=0.04):
    a = M.CO.annual_costs(ok, x).sum(axis=1)
    cap = ok.capex_usd if capex is None else capex
    a = a + cap * (M.crf(x["discount_rate"], 15) + fom)
    c = a / ok.water_m3_year
    i = c.idxmin()
    edge = bool(ok.capacity[i] >= 2.5 or ok.kwp_per_m3d[i] >= 6.0 or ok.tank_h[i] >= 48)
    return float(c[i]), dict(capacity=float(ok.capacity[i]), kwp_per_m3d=float(ok.kwp_per_m3d[i]), tank_h=float(ok.tank_h[i]),
                             battery_kWh=float(ok[e_col][i]), at_edge_of_search=edge)


sens = []
for name, ch in SCEN.items():
    x = {**X0, **ch}
    deck = {f"{p:.2f}": float(M.CO.per_m3(g, x, p).sum(axis=1)[deck_i]) for p in PRICES_MAIN}
    wcost, wdes = water_best(x)
    bcost, bdes = batt_best(x)
    fixed = float(M.CO.annual_costs(g, x).sum(axis=1)[deck_i] / g.water_m3[deck_i])
    sens.append(dict(case=name, deck_usd_per_m3=deck, water_only_usd_per_m3=wcost, water_only_design=wdes,
                     battery_usd_per_m3=bcost, battery_design=bdes,
                     breakeven_water_only=breakeven(wcost, fixed), breakeven_battery=breakeven(bcost, fixed)))
ok4 = ok[ok.usd_per_m3_4h_2023.notna() & (ok.E_4h_kWh > 0.5)] if "usd_per_m3_4h_2023" in ok else ok.iloc[0:0]
if len(ok4):
    i4 = ok4.usd_per_m3_4h_2023.idxmin()
    c4 = float(ok4.usd_per_m3_4h_2023[i4])
    sens.append(dict(case="battery 482 USD/kWh, 4-hour, O&M 2.5 % (NREL 2023 update, behind ATB 2024; 2022 USD)",
                     deck_usd_per_m3={f"{p:.2f}": deck_cost[f"{p:.2f}"] for p in PRICES_MAIN},
                     water_only_usd_per_m3=wc["usd_per_m3"], battery_usd_per_m3=c4,
                     battery_design=dict(capacity=float(ok4.capacity[i4]), kwp_per_m3d=float(ok4.kwp_per_m3d[i4]),
                                         tank_h=float(ok4.tank_h[i4]), battery_kWh=float(ok4.E_4h_kWh[i4])),
                     breakeven_water_only=breakeven(wc["usd_per_m3"]), breakeven_battery=breakeven(c4)))

# ---------------------------------------------------------------- results.json: one deck page
named = {f"{o['capacity']:g}x, {o['kwp_per_m3d']:g} kWp, {o['tank_h']:g} h": (
    dict(off_grid_possible=False, reason="over the ten years the sun the plant cannot use, after the 15 % battery loss, "
                                          "is less than the grid energy to replace",
         ten_year_grid_without_battery_kWh=o["ten_year_grid_without_battery_kWh"],
         ten_year_store_if_all_charged_kWh=o["ten_year_store_if_all_charged_kWh"])
    if not o["off_grid_possible"] else
    dict(off_grid_possible=True, battery_kWh=o["check"]["battery_kWh"], battery_kW=o["check"]["battery_kW"],
         battery_smallest_kWh_power_not_limiting=o["E_min_kWh"], usd_per_m3=o["usd_per_m3"],
         unused_panel_energy_pct=o["unused_panel_energy_pct"], full_cycles_per_year=o["full_cycles_per_year"],
         critical_stretch=o["critical_stretch"]))
    for o in batt["named_designs"]}

results = dict(
    question="Can the solar RO plant (1,000 m3/day) run off-grid: zero grid kWh in every year 2015-2024, demand always met?",
    model="phase-2 scripts (schedule.py, design_grid.py, costs.py, cost_inputs.json); ten years as one series in a repeating state",
    units=dict(capacity="plant output at design flux / demand", panels="kWp per m3/day of demand", tank="hours of demand",
               battery="kWh per 1,000 m3/day of demand (deliverable energy)", cost="USD per m3, the part the design decides "
               "(RO plant capital and upkeep growing with capacity, panels, tank, battery, grid electricity)"),
    options=options,
    battery_on_named_designs=named,
    deck_design_usd_per_m3_by_grid_price=deck_cost,
    deck_design_usd_per_m3_parts_at_0_10=deck_parts_010,
    sensitivities=sens,
    sources=dict(battery="NREL/TP-6A40-93281 (Cole, Ramasamy & Turan, June 2025) p. 3, 6, 8, 11, 12",
                 battery_sensitivity="NREL/TP-6A40-85332 (Cole & Karmakar, June 2023) p. 2, 4, 7, 8, Table 2 p. 13",
                 other_costs="phase-2 cost_inputs.json _sources"),
)
(M.HERE / "results.json").write_text(json.dumps(results, indent=1, default=float, ensure_ascii=False))
(OUT / "compare.json").write_text(json.dumps(dict(sensitivities=sens, deck_fixed_usd_per_m3=deck_fixed,
                                                  deck_grid_kWh_per_m3=deck_grid_kwh_m3), indent=1, default=float))
tab = pd.DataFrame([dict(option=o["option"], capacity=o["design"]["capacity"], kwp=o["design"]["kwp_per_m3d"],
                         tank_h=o["design"]["tank_h"], sun_pct=o["sun_share_pct"], unused_pct=o["unused_panel_energy_pct"],
                         battery_kWh=o.get("battery_kWh"),
                         usd_m3=o.get("usd_per_m3", o.get("usd_per_m3_by_grid_price", {}).get("0.10"))) for o in options])
tab.to_csv(OUT / "compare_table.csv", index=False)
pd.set_option("display.width", 220)
print(tab.round(4).to_string(index=False))
print("deck design by grid price:", {k: round(v, 4) for k, v in deck_cost.items()})
for s in sens:
    print(s["case"], "| deck", {k: round(v, 3) for k, v in s["deck_usd_per_m3"].items()}, "| water", round(s["water_only_usd_per_m3"], 4),
          s.get("water_only_design"), "| battery", round(s["battery_usd_per_m3"], 4), s.get("battery_design"),
          "| breakeven w/b", round(s.get("breakeven_water_only", float("nan")), 3), round(s["breakeven_battery"], 3))
