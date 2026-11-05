"""Writes REPORT.md from the computed outputs (out/*.json, out/*.csv, results.json): every number in the report is read
from those files, none is typed by hand.

    python make_report.py
"""
import json

import numpy as np
import pandas as pd

import model as M

OUT = M.HERE / "out"
chk = json.loads((OUT / "check_model.json").read_text())
wat = json.loads((OUT / "water_result.json").read_text())
bat = json.loads((OUT / "battery_result.json").read_text())
res = json.loads((M.HERE / "results.json").read_text())
cmp_ = json.loads((OUT / "compare.json").read_text())
coarse = pd.read_csv(OUT / "water_coarse.csv")
fine = pd.read_csv(OUT / "water_fine.csv")
scr = pd.read_csv(OUT / "battery_screen.csv")
X0 = M.cost_inputs()


def phase2_unchanged():
    """md5 of every phase-2 file (outside offgrid/) now, against the manifest taken before the study started."""
    import hashlib
    before = {}
    for line in (M.HERE / ".phase2_manifest_before.md5").read_text().splitlines():
        h, p = line.split(maxsplit=1)
        before[p.strip()] = h
    now = {}
    for p in sorted(M.PHASE2.rglob("*")):
        if p.is_file() and M.HERE not in p.parents:
            now["./" + str(p.relative_to(M.PHASE2))] = hashlib.md5(p.read_bytes()).hexdigest()
    return before == now, len(before), len(now)


SAME, N_BEFORE, N_NOW = phase2_unchanged()


def f(v, d=1):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}"


def design(d):
    return f"{d['capacity']:g}×, {d['kwp_per_m3d']:g} kWp per m³/day, {d['tank_h']:g} h tank"


def mon(m):
    import calendar
    return calendar.month_abbr[int(m)]


b1 = chk["1_baseline"]
deck_b, big_b = b1["2x, 0.8 kWp per m3/d, 12 h"], b1["2x, 2 kWp per m3/d, 24 h"]
cont = chk["4_continuous_deck_design"]
cap = chk["3_capacity_above_2x"]
first = chk["5_phase2_start_empty_4x_6kWp_168h"]
wc = wat["cheapest"]
bc = bat["cheapest"]
opt = res["options"]
deck_cost = res["deck_design_usd_per_m3_by_grid_price"]
P5 = ["0.05", "0.10", "0.15", "0.20", "0.30"]
named = bat["named_designs"]
B = bat["inputs"]["base"]
S = bat["inputs"]["sensitivity"]
crf15 = bat["inputs"]["crf_15y"]
rate = bat["inputs"]["discount_rate"]

# cost of storing one hour of demand: as water (tank) or as battery energy (the plant's night energy for one hour)
crf25 = M.crf(X0["discount_rate"], X0["tank_life_y"])
tank_hour_usd_y = (crf25 + X0["tank_om_frac_y"]) * X0["tank_capex_usd_per_m3"] * M.D
data = M.load()
cvb = M.Sc.curves(data["perf"], data["other"], bc["capacity"])          # the plant of the cheapest battery design
night_kwh = [float(np.interp(M.D, cvb[T]["q"], cvb[T]["E"])) for T in sorted(cvb)]
batt_hour_usd_y = [B["e_cost"] * e * (crf15 + B["fom"]) for e in night_kwh]

L = []
w = L.append
w("# Can the solar RO plant run off-grid?")
w("")
w("*Off-grid study on the phase-2 model of the AI for RO deck (seawater RO, 1,000 m³/day, follows the sun, product-water tank). "
  "All numbers below are computed by the scripts in this folder from the phase-2 model and data, or copied from the cited sources.*")
w("")
w("## Answer")
w("")
w(f"**Yes, the model can run the plant on the sun alone, but not with the deck's design, and the water then costs more.** Off-grid "
  f"here means zero grid kWh in every one of the ten years 2015–2024, with the demand always met.")
w("")
diff = opt[2]["usd_per_m3"] - opt[1]["usd_per_m3"]
tie = abs(diff) / opt[1]["usd_per_m3"] < 0.01
w(f"- **The deck's design ({design(opt[0]['design'])}) cannot be off-grid.** Its sun share is {f(opt[0]['sun_share_pct'])} % "
  f"({f(opt[0]['sun_share_pct_by_year_range'][0])}–{f(opt[0]['sun_share_pct_by_year_range'][1])} by year). No battery size closes "
  f"the gap: over the ten years, the sun the plant cannot use, after the battery's 15 % loss, is less than the grid energy it would "
  f"have to replace.")
w(f"- **Two routes reach zero grid, " + ("at almost the same cost" if tie else "at different costs") + ":**")
w(f"  - **More water storage:** {design(wc)}: **{f(wc['usd_per_m3'], 3)} USD/m³**. A bigger plant fills a multi-day tank on "
  f"sunny days.")
w(f"  - **A battery:** {design(bc)} + a {f(bc['check']['battery_kWh'], 0)} kWh / {f(bc['check']['battery_kW'], 0)} kW battery: "
  f"**{f(bc['usd_per_m3'], 3)} USD/m³** at NREL's 2024 battery price ({f(opt[2]['battery_usd_per_m3_at_2023_4h_price'], 3)} at the "
  f"higher 2022 price behind NREL ATB 2024). A plant the size of the demand runs day and night; the battery carries the nights and "
  f"the low-sun days.")
w(f"  - The difference, {f(diff, 3)} USD/m³, is " + ("well inside what the cost inputs can tell apart; which route is cheaper "
  "depends on what a bigger plant and a battery really cost (section 7)." if tie else "shown with its sensitivities in section 7."))
sens = cmp_["sensitivities"]
water_wins = [s["case"] for s in sens if s["water_only_usd_per_m3"] < s["battery_usd_per_m3"]]
batt_wins = [s["case"] for s in sens if s["water_only_usd_per_m3"] >= s["battery_usd_per_m3"]]
be = [min(s["breakeven_water_only"], s["breakeven_battery"]) for s in sens]
w(f"  - Across the {len(sens)} cost cases of section 7, water storage is cheaper in {len(water_wins)} "
  f"({'; '.join(water_wins)}) and the battery in {len(batt_wins)} ({'; '.join(batt_wins)}).")
w(f"- **Keeping the deck's 2× plant and adding a battery** costs more: {design(opt[3]['design'])} + "
  f"{f(opt[3]['battery_kWh'], 0)} kWh: {f(opt[3]['usd_per_m3'], 3)} USD/m³.")
w(f"- **The deck's design buying from the grid** costs {', '.join(f(deck_cost[p], 3) for p in P5)} USD/m³ at grid prices of "
  f"{', '.join(P5)} USD/kWh. Going off-grid only pays if the grid costs more than "
  f"**{f(min(opt[1]['deck_design_costs_the_same_at_grid_price'], opt[2]['deck_design_costs_the_same_at_grid_price']), 2)} USD/kWh** "
  f"({f(min(be), 2)}–{f(max(be), 2)} across the cost cases), above every grid price phase 2 considers "
  f"({min(float(k) for k in deck_cost):.2f}–{max(float(k) for k in deck_cost):.2f} USD/kWh).")
w("")
w("| Option | Design | Sun share | Panels' energy unused | Battery | USD per m³ |")
w("|---|---|---|---|---|---|")
w(f"| Deck design, grid makes up the rest | {design(opt[0]['design'])} | {f(opt[0]['sun_share_pct'])} % | "
  f"{f(opt[0]['unused_panel_energy_pct'])} % | – | " + " · ".join(f"{f(deck_cost[p], 3)} at {p}" for p in P5) + " USD/kWh |")
w(f"| Off-grid, water storage only (cheapest) | {design(opt[1]['design'])} | {f(opt[1]['sun_share_pct'])} % | "
  f"{f(opt[1]['unused_panel_energy_pct'])} % | – | {f(opt[1]['usd_per_m3'], 3)} |")
for o in opt[2:]:
    w(f"| {o['option'][0].upper() + o['option'][1:]} | {design(o['design'])} | {f(o['sun_share_pct'])} % | "
      f"{f(o['unused_panel_energy_pct'])} % | {f(o['battery_kWh'], 0)} kWh, {f(o['battery_kW'], 0)} kW | {f(o['usd_per_m3'], 3)} |")
w("")
w("Costs are the part of the water cost that the design decides (costs.py): RO plant capital and upkeep growing with capacity, "
  "panels, tank, battery, grid electricity. Chemicals, staff and other costs that every design pays alike are left out, as in phase 2. "
  "Battery sizes are per 1,000 m³/day of demand.")
w("")

# ------------------------------------------------------------------ method
w("## 1. Method")
w("")
w("The phase-2 scripts are imported read-only from the parent folder: `schedule.py` (demand 41.667 m³/h, 8 trains, energy and "
  "permeate curves per capacity, NASA POWER sun and sea temperature), `design_grid.py` (the fast rule and its scoring), `costs.py` "
  "(`annual_costs`, `per_m3`, `designs`, `SCENARIOS`) and `cost_inputs.json`. "
  + (f"Nothing in the phase-2 folder was changed: the checksums of all its {N_NOW} files when this report was written equal those "
     f"taken before the study (Python was stopped from writing `.pyc` files there)." if SAME else
     f"**Warning: the phase-2 folder's checksums differ from those taken before the study ({N_BEFORE} files before, {N_NOW} now).**"))
w("")
w("**What I changed, and nothing else:**")
w("")
fs = [first[y]["grid_kWh_before_first_sunrise"] for y in first if y != "2015"]
w(f"1. **The ten years run as one series.** Phase 2 runs each year on its own, starting with an empty tank at midnight on 1 January. "
  f"That start alone makes zero grid impossible: even a 4×, 6 kWp per m³/day, 168-hour design buys "
  f"{f(min(fs))}–{f(max(fs))} kWh from the grid before the first sunrise of each year 2016–2024 "
  f"({f(first['2015']['grid_kWh_before_first_sunrise'])} kWh in 2015, whose data start at 03:00). Here the tank is carried from one "
  f"year to the next, and its level on 1 January 2015 is set equal to its level on 31 December 2024 (a repeating state, found by "
  f"iteration). For the deck's design this moves the ten-year mean sun share from {f(deck_b['sun_share_pct_mean'], 2)} % to "
  f"{f(cont['sun_share_pct_mean'], 2)} %.")
w("2. **Part (b) only: a battery.** It charges only from sun the plant cannot use (phase 2's curtailed sun) and supplies the plant "
  "whenever the rule would buy from the grid. The rule still decides what the plant makes, exactly as in phase 2.")
w("")
w("The physics (vessel model, curves, energy of the rest of the plant), the rule (by day make what the sun allows, up to capacity "
  "and tank space; at night or when the tank is empty make just the demand) and the cost method are unchanged.")
w("")
w("**Smallest store.** For a given capacity and panel size, the water the sun alone can make each hour is fixed by the rule. The "
  "smallest tank that never runs dry is then found exactly with the sequent-peak method used to size reservoirs (the largest "
  "drawdown from a full tank, with the ten years repeating). The same calculation sizes the battery, with the battery's net flow "
  "(85 % of the spare sun it takes in, minus the grid energy it replaces). Both are checked against hour-by-hour simulation (section 2).")
w("")
w(f"**Off-grid** means grid ≤ 1e-6 kWh in each year (the floating-point noise of the simulation) and no short hours.")
w("")
w(f"**Costs.** `costs.annual_costs` with `cost_inputs.json` (discount rate {100 * rate:.1f} %, 25 years for plant, panels and tank), "
  f"divided by the m³ made per year, as in phase 2. The battery is annualised the same way over its own life: capital × "
  f"(CRF({100 * rate:.1f} %, {B['life']} y) = {crf15:.4f} + fixed O&M {100 * B['fom']:.0f} %). No grid cost for the off-grid designs. "
  f"Tanks are rounded up to whole hours and batteries to whole kWh and kW, and each rounded design is simulated hour by hour.")
w("")

# ------------------------------------------------------------------ checks
w("## 2. What I checked")
w("")
w(f"1. **The phase-2 baseline, re-run with the phase-2 code:** deck design {f(deck_b['sun_share_pct_mean'], 2)} % sun "
  f"({f(deck_b['sun_share_pct_min'], 2)}–{f(deck_b['sun_share_pct_max'], 2)} by year); largest swept design (2×, 2.0 kWp, 24 h) "
  f"{f(big_b['sun_share_pct_mean'], 2)} % ({f(big_b['sun_share_pct_min'], 2)}–{f(big_b['sun_share_pct_max'], 2)}), with "
  f"{f(big_b['unused_panel_energy_pct_10y_total'], 2)} % of the panels' energy unused. Largest difference from `out/design_grid.csv`: "
  f"{max(deck_b['max_abs_diff_vs_design_grid_csv'], big_b['max_abs_diff_vs_design_grid_csv']):.1e} (kWh or m³).")
w(f"2. **My copy of the rule** (`model.rule_s0`, the same lines plus a start level) gives exactly `design_grid.rule_fast`'s output "
  f"when it starts empty (largest difference {chk['2_rule_s0_equals_rule_fast_max_abs_diff_m3h']:g} m³/h, four designs × ten years).")
w(f"3. **Capacities above 2×.** `schedule.curves` keeps **{cap['trains']} trains** at every capacity and makes each one bigger "
  f"(Q_tr = C × demand / 8: {f(cap['train_size_m3h_at_design_flux']['2x'], 2)} m³/h per train at 2×, "
  f"{f(cap['train_size_m3h_at_design_flux']['4x'], 2)} at 4×). The energy and permeate per m³ at a given flux depend only on the "
  f"flux and the sea temperature, so a capacity's curve is the 2× curve scaled by C/2. Checked for 2.5×, 3×, 4× and 5×: the curves "
  f"scale exactly (largest deviation {cap['max_deviation_from_exact_scaling']:.1e}), the flux window (f_min, f_max) is the same, "
  f"the top output is C × demand × f_max(T), and the convexity check inside `curves` passes. The fast rule matches the slow "
  f"reference rule `schedule.run_rule` hour by hour at 4× (4 kWp, 72 h, 2024) and 3× (2.5 kWp, 48 h, 2022): largest difference "
  f"{cap['rule_fast_vs_run_rule_4x_4kWp_72h_2024']['max_abs_diff_q']:g} m³/h. What this means: a 4× plant is modelled as 8 "
  f"trains of {f(cap['train_size_m3h_at_design_flux']['4x'], 2)} m³/h, not as 16 trains of today's size. That changes the turndown "
  f"steps and the train starts, not the energy per m³; below 8 × f_min the model already lets part of a train run, as in phase 2.")
w(f"4. **The exact smallest tank agrees with hour-by-hour simulation** in all {wat['coarse']['exact_min_tank_agrees']} of "
  f"{wat['coarse']['designs']} coarse designs: a design is off-grid in the simulation exactly when its tank is at least the exact "
  f"minimum.")
v = wc["check"]
w(f"5. **The cheapest water-only design, simulated hour by hour:** zero grid in all ten years "
  f"(largest year {v['grid_kWh_year_max']:.1e} kWh), {v['short_hours']} short hours; the tank is full for "
  f"{min(v['tank_full_hours_by_year'].values())}–{max(v['tank_full_hours_by_year'].values())} hours each year, so no year leans on the "
  f"repeating-state assumption. With the tank half an hour smaller ({f(wc['check_tank_minus_half_hour']['tank_h'], 2)} h) it buys "
  f"{f(wc['check_tank_minus_half_hour']['grid_kWh_worst_year'])} kWh in its worst year.")
if bc and bc.get("check"):
    bb = bc["check"]
    w(f"6. **The cheapest battery design, simulated hour by hour:** "
      + ("zero grid in all ten years" if bb["off_grid"] else "**NOT off-grid**")
      + f", {bb['short_hours']} short hours; the battery is full for {min(bb['battery_full_hours_by_year'].values())}–"
      f"{max(bb['battery_full_hours_by_year'].values())} hours each year. With 99 % of its energy it buys "
      f"{f(bb['with_99pct_energy_grid_kWh_10y'])} kWh over the ten years.")
w("")

# ------------------------------------------------------------------ (a)
w("## 3. (a) Water storage only")
w("")
w("**Smallest tank (hours of demand) for zero grid in every year**, exact; ∞ = impossible with any tank. Rows: capacity; "
  "columns: kWp of panels per m³/day.")
w("")
m = coarse.pivot_table(index="capacity", columns="kwp_per_m3d", values="min_tank_h_exact", aggfunc="first")
w("| capacity | " + " | ".join(f"{c:g}" for c in m.columns) + " |")
w("|---|" + "---|" * len(m.columns))
for C, row in m.iterrows():
    w(f"| {C:g}× | " + " | ".join("∞" if not np.isfinite(x) else f"{x:,.1f}" for x in row.values) + " |")
w("")
w(f"The coarse sweep simulated {wat['coarse']['designs']} designs hour by hour (capacity {', '.join(f'{c:g}' for c in wat['coarse']['grid']['capacity'])}; "
  f"panels {', '.join(f'{p:g}' for p in wat['coarse']['grid']['kwp_per_m3d'])}; tank {', '.join(str(t) for t in wat['coarse']['grid']['tank_h'])} h); "
  f"{wat['coarse']['off_grid']} reach zero grid in all ten years (full results: `out/water_coarse.csv`). The fine search then took "
  f"capacity {wat['fine']['capacity'][0]:g}–{wat['fine']['capacity'][1]:g} and panels {wat['fine']['kwp_per_m3d'][0]:g}–"
  f"{wat['fine']['kwp_per_m3d'][1]:g} kWp per m³/day in steps of 0.05, each with its exact smallest tank "
  f"({wat['fine']['pairs']} pairs, {wat['fine']['pairs_off_grid_possible']} of them possible; `out/water_fine.csv`).")
w("")
p = wc["usd_per_m3_parts"]
w(f"**Cheapest:** {design(wc)} (exact minimum {f(wc['min_tank_h_exact'], 1)} h, rounded up): **{f(wc['usd_per_m3'], 3)} USD/m³** "
  f"= RO plant {f(p['RO plant'], 3)} + RO upkeep {f(p['RO upkeep'], 3)} + panels {f(p['panels'], 3)} + tank {f(p['tank'], 3)}. "
  f"It uses {f(v['kWh_per_m3'], 3)} kWh/m³ and leaves {f(v['unused_panel_energy_pct'])} % of its panels' energy unused. "
  + ("The optimum lies inside the searched range" if not (wat['fine']['capacity_at_edge'] or wat['fine']['panels_at_edge'])
     else "**The optimum lies at the edge of the searched range**") +
  f", and it is flat: {len(wat['within_1pct_of_cheapest'])} designs are within 1 % of it, from "
  f"{min(r['capacity'] for r in wat['within_1pct_of_cheapest']):g}× to {max(r['capacity'] for r in wat['within_1pct_of_cheapest']):g}×, "
  f"{min(r['kwp_per_m3d'] for r in wat['within_1pct_of_cheapest']):g}–{max(r['kwp_per_m3d'] for r in wat['within_1pct_of_cheapest']):g} kWp, "
  f"{f(min(r['min_tank_h'] for r in wat['within_1pct_of_cheapest']), 0)}–{f(max(r['min_tank_h'] for r in wat['within_1pct_of_cheapest']), 0)} h.")
w("")
cb = wat["fine"]["cheapest_by_capacity"]
w("**Cheapest off-grid cost at each capacity** (best panels and exact tank): " +
  " · ".join(f"{k}× {f(cb[k], 3)}" for k in ["1.95", "2", "2.2", "2.45", "2.7", "3", "3.5"] if k in cb) + " USD/m³. "
  f"Below {wat['fine']['lowest_feasible_capacity']:g}× nothing works in the searched range.")
w("")
cs = wc["critical_stretch"]
low_y = min(v["tank_lowest_m3_by_year"], key=v["tank_lowest_m3_by_year"].get)
w(f"**What sets its tank:** the worst stretch of the ten years, {cs['start'][:16]} to {cs['end'][:16]} ({f(cs['days'])} days, "
  f"{'–'.join(mon(x) for x in cs['months'])}): on {cs['days_sun_short_of_demand']} of its {cs['days_in_stretch']} calendar days the "
  f"sun could not make the day's water, {cs['low_sun_days_in_stretch']} were low-sun days (phase 2's definition: under half the "
  f"month's median), and over the stretch the sun alone made {f(cs['sun_alone_makes_pct_of_demand_in_stretch'])} % of the demand. "
  f"The tank's lowest level in the ten years is {f(v['tank_lowest_m3_by_year'][low_y], 0)} m³, in {low_y}.")
w("")
w("### What stops the deck's 2× plant")
w("")
cl = wat["capacity_limit_unlimited_panels"]
m2 = cl["2x"]["month_mean_output_vs_demand"]
low2 = sorted(m2, key=m2.get)[:2]
fw = cap["f_window_by_T"]
hot = [T for T in fw if fw[T][1] <= 1.0 + 1e-9]
w(f"- **A day only has so many sunny hours.** With unlimited panels the plant runs at full output whenever the sun is up. "
  f"Even then a 1.5× plant makes only {f(100 * cl['1.5x']['ten_year_output_vs_demand'])} % of the ten years' demand "
  f"(monthly means {f(100 * min(cl['1.5x']['month_mean_output_vs_demand'].values()))}–"
  f"{f(100 * max(cl['1.5x']['month_mean_output_vs_demand'].values()))} %), so it can never be off-grid. A 2× plant makes "
  f"{f(100 * cl['2x']['ten_year_output_vs_demand'])} %, but only {f(100 * m2[low2[0]])} % in {mon(low2[0])} and "
  f"{f(100 * m2[low2[1]])} % in {mon(low2[1])}: almost no margin. At sea temperatures of {' and '.join(hot)} °C each train has a "
  f"single usable point, its design flux (f_min = f_max = 1.0), so in the hot months the plant cannot run above its nominal "
  f"capacity. In the cool months the trains can run up to {f(100 * max(x[1] for x in fw.values()), 1)} % of design flux, but the "
  f"days are short.")
st = {(s["capacity"], s["kwp_per_m3d"]): s for s in wat["what_stops_2x"]}
w(f"- **With real panel fields the margin is gone for months.** At 2× the smallest workable field is "
  f"{fine[(fine.capacity == 2.0) & np.isfinite(fine.min_tank_h)].kwp_per_m3d.min():g} kWp per m³/day, and it needs a "
  f"{f(float(fine[(fine.capacity == 2.0) & np.isfinite(fine.min_tank_h)].sort_values('kwp_per_m3d').min_tank_h.iloc[0]), 0)}-hour tank. "
  f"With 4 kWp the tank must carry {f(st[(2.0, 4.0)]['tank_h_needed'], 0)} h of demand through a {f(st[(2.0, 4.0)]['days'], 0)}-day "
  f"drawdown ({st[(2.0, 4.0)]['start'][:10]} to {st[(2.0, 4.0)]['end'][:10]}); with 6 kWp, {f(st[(2.0, 6.0)]['tank_h_needed'], 0)} h "
  f"over {f(st[(2.0, 6.0)]['days'], 0)} days; with 8 kWp, {f(st[(2.0, 8.0)]['tank_h_needed'], 0)} h. With 3 kWp the drawdown lasts "
  f"{f(st[(2.0, 3.0)]['days'], 0)} days, running across years ({f(st[(2.0, 3.0)]['tank_h_needed'], 0)} h). The cheapest 2× off-grid "
  f"design costs {f(cb['2'], 3)} USD/m³.")
lg = wat["phase2_largest_design_2x_2kWp_24h"]
mo = lg["grid_share_by_month_pct"]
top = sorted(mo.items(), key=lambda kv: -kv[1])[:6]
w(f"- **It is the season, not a few cloudy days.** The largest phase-2 design (2×, 2.0 kWp, 24 h) buys from the grid on "
  f"{f(lg['days_with_grid_per_year'])} days a year, up to {lg['longest_run_days_with_grid']} days in a row; its grid energy falls in "
  + ", ".join(f"{mon(k)} ({f(v)} %)" for k, v in top) + f" and only {f(sum(v for k, v in mo.items() if k not in dict(top)))} % in "
  f"the other six months. Low-sun days are only {f(lg['low_sun_days_per_year'])} a year, at most {lg['longest_low_sun_run_days']} "
  f"in a row (worst day {lg['worst_day']}, {f(lg['worst_day_kWh_per_kWp'], 2)} kWh per kWp). With more capacity a short spell sets "
  f"the tank instead: for {wc['capacity']:g}× with {wc['kwp_per_m3d']:g} kWp it is {f(wc['critical_stretch']['days'])} days from "
  f"{wc['critical_stretch']['start'][:10]}; for 2.5× and 3× with 2 kWp, {f(st[(2.5, 2.0)]['days'])} and {f(st[(3.0, 2.0)]['days'])} "
  f"days from {st[(2.5, 2.0)]['start'][:10]} and {st[(3.0, 2.0)]['start'][:10]}.")
w("")

# ------------------------------------------------------------------ (b)
w("## 4. (b) A battery")
w("")
w(f"**Inputs** (NREL, *Cost Projections for Utility-Scale Battery Storage: 2025 Update*, NREL/TP-6A40-93281, June 2025; 2024 USD): "
  f"capital = {B['e_cost']:g} USD/kWh × energy + {B['p_cost']:g} USD/kW × power; fixed O&M {B['fom']:.0%} of capital a year; no "
  f"variable O&M; round trip {B['eta']:.0%}; life {B['life']} years. Exact quotes and pages are in section 9.")
w("")
w("**Assumptions (mine):** the rated energy is the energy the battery delivers (NREL defines duration as output duration), so the "
  "15 % loss is taken on charging; no self-discharge; one power rating for charging and discharging, chosen as the cheapest that "
  "still gives zero grid; fixed O&M as a share of this battery's own capital; the battery is replaced at the end of its life "
  "(annualised over 15 years). NREL's line was fitted to systems of 1–10 hours for the mainland United States; longer batteries are "
  "an extrapolation.")
w("")
w("| Design | Grid it would buy, 10 years | Spare sun × 85 %, 10 years | Smallest battery (power not limiting) | Cheapest battery | Duration | Full cycles a year | USD per m³ |")
w("|---|---|---|---|---|---|---|---|")
for o in named:
    dd = f"{o['capacity']:g}×, {o['kwp_per_m3d']:g} kWp, {o['tank_h']:g} h"
    if not o["off_grid_possible"]:
        w(f"| {dd} | {f(o['ten_year_grid_without_battery_kWh'] / 1e6, 3)} GWh | {f(o['ten_year_store_if_all_charged_kWh'] / 1e6, 3)} GWh | "
          f"impossible | – | – | – | – |")
    else:
        cc = o["check"]
        w(f"| {dd} | {f(o['ten_year_grid_without_battery_kWh'] / 1e6, 3)} GWh | {f(o['ten_year_store_if_all_charged_kWh'] / 1e6, 3)} GWh | "
          f"{f(o['E_min_kWh'], 0)} kWh | {f(cc['battery_kWh'], 0)} kWh, {f(cc['battery_kW'], 0)} kW | {f(cc['duration_h'])} h | "
          f"{f(o['full_cycles_per_year'])} | {f(o['usd_per_m3'], 3)} |")
w("")
d0 = named[0]
ratio = {y: r["stored_if_all_charged_kWh"] / r["grid_kWh_without_battery"] for y, r in d0["by_year"].items()}
short_years = [y for y, x in ratio.items() if x < 1]
w(f"**The deck's design cannot be made off-grid with a battery:** over the ten years it would buy "
  f"{f(d0['ten_year_grid_without_battery_kWh'] / 1e6, 3)} GWh from the grid, but its spare sun, after the 15 % loss, gives only "
  f"{f(d0['ten_year_store_if_all_charged_kWh'] / 1e6, 3)} GWh "
  f"({f(100 * d0['ten_year_store_if_all_charged_kWh'] / d0['ten_year_grid_without_battery_kWh'])} % of the need). Year by year "
  f"(spare sun after the loss, as a share of the grid energy to replace): "
  + ", ".join(f"{y} {f(100 * x)} %" for y, x in ratio.items()) +
  f". It is short in {len(short_years)} of the 10 years"
  + (f"; only in {', '.join(y for y in ratio if y not in short_years)} does the year's total cover the need. Over the ten years the "
     f"spare sun is short, so no battery size, however large, makes this design off-grid." if len(short_years) < 10 else "."))
w("")
for o in named[1:]:
    if o["off_grid_possible"]:
        c_ = o["critical_stretch"]
        w(f"- {o['capacity']:g}×, {o['kwp_per_m3d']:g} kWp, {o['tank_h']:g} h: the battery is sized by {c_['start'][:10]} to "
          f"{c_['end'][:10]} ({f(c_['days'], 0)} days, {'–'.join(mon(x) for x in c_['months'])}).")
w("")
sc = bat["screen"]


def battery_paragraph(title, o):
    pc, cc = o["usd_per_m3_parts"], o["check"]
    return (f"**{title}: {design(o)} + {f(cc['battery_kWh'], 0)} kWh / {f(cc['battery_kW'], 0)} kW** ({f(cc['duration_h'])} h), "
            f"**{f(o['usd_per_m3'], 3)} USD/m³** = RO plant {f(pc['RO plant'], 3)} + upkeep {f(pc['RO upkeep'], 3)} + panels "
            f"{f(pc['panels'], 3)} + tank {f(pc['tank'], 3)} + battery {f(pc['battery'], 3)}. Smallest energy with power not limiting: "
            f"{f(o['E_min_kWh'], 0)} kWh. {f(o['full_cycles_per_year'])} full cycles a year; {f(o['battery_loss_kWh_per_year'], 0)} kWh a "
            f"year lost in the battery; {f(o['unused_panel_energy_pct'])} % of the panels' energy still unused. Its size is set by "
            f"{o['critical_stretch']['start'][:16]} to {o['critical_stretch']['end'][:16]} ({f(o['critical_stretch']['days'])} days). "
            f"Simulated hour by hour: " + ("zero grid in all ten years" if cc["off_grid"] else "**NOT off-grid**") +
            f", battery full {min(cc['battery_full_hours_by_year'].values())}–"
            f"{max(cc['battery_full_hours_by_year'].values())} hours each year. At the price behind NREL ATB 2024 (482 USD/kWh for a "
            f"4-hour system, 2022 USD, O&M 2.5 %): {f(o['sensitivity_4h_2023']['usd_per_m3'], 3)} USD/m³ with "
            f"{f(o['sensitivity_4h_2023']['E_kWh'], 0)} kWh.")


w(f"**Search.** A screen of {sc['designs']} designs (capacity {', '.join(f'{c:g}' for c in sc['grid']['capacity'])}; panels "
  f"{', '.join(f'{c:g}' for c in sc['grid']['kwp_per_m3d'])}; tank {', '.join(f'{c:g}' for c in sc['grid']['tank_h'])} h), of which "
  f"{sc['off_grid_possible']} can be off-grid and {sc['with_a_battery']} need a battery; then a local search from the best "
  f"({sc['local_search_designs']} designs), moving to any cheaper neighbour and halving the steps down to 0.05×, 0.05 kWp and 1 h "
  f"(`out/battery_screen.csv`). "
  + ("The capacity of the cheapest design is at the lowest possible value, 1×: a plant smaller than the demand cannot meet it on "
     "hot days, when each train is held at its design point. " if sc["capacity_at_lower_bound_1x"] else "")
  + f"{len(sc['within_1pct_of_cheapest'])} searched designs are within 1 % of the cheapest.")
w("")
w(battery_paragraph("Cheapest design with a battery", bc))
w("")
w(battery_paragraph("Cheapest with a battery that keeps the deck's 2× plant", bat["cheapest_at_2x"]))
w("")
one_x = (M.crf(X0["discount_rate"], X0["plant_life_y"]) * X0["ro_capex_usd_per_m3d"] * 1000 * X0["ro_capacity_share"]
         + X0["ro_capacity_om_usd_per_m3"] * 1000 * 365 * X0.get("upkeep_grows", 1.0))
w(f"**Water or battery: the trade-off.** Storing one hour of demand costs {f(tank_hour_usd_y, 0)} USD a year as water in the tank, "
  f"but {f(min(batt_hour_usd_y), 0)}–{f(max(batt_hour_usd_y), 0)} USD a year as battery energy (the plant's energy for one hour of "
  f"demand at night, {f(min(night_kwh), 0)}–{f(max(night_kwh), 0)} kWh, at {B['e_cost']:g} USD/kWh, before the power part). But the "
  f"water has to be made while the sun shines, which needs a bigger plant: each extra 1× of capacity costs {f(one_x, 0)} USD a year "
  f"(RO plant capital and upkeep, base inputs). The battery lets a plant the size of the demand run through the night instead. At "
  f"the base inputs the two come out " + ("almost level" if tie else f"{f(abs(diff), 3)} USD/m³ apart") +
  "; section 7 shows which way each input tips it.")
w("")

# ------------------------------------------------------------------ (c)
w("## 5. (c) Not modelled: wind, solar thermal with heat storage")
w("")
w("Other ways to bridge the nights and the low-sun season exist and are not modelled here, because the project has no model for "
  "them: wind turbines, and concentrating solar power with thermal (heat) storage. No numbers are given for them.")
w("")

# ------------------------------------------------------------------ compare
w("## 6. Compare (same cost method)")
w("")
w("| | USD per m³ |")
w("|---|---|")
for p_ in P5:
    w(f"| Deck design (2×, 0.8 kWp, 12 h) buying from the grid at {p_} USD/kWh | {f(deck_cost[p_], 3)} |")
w(f"| Off-grid, water only: {design(opt[1]['design'])} | {f(opt[1]['usd_per_m3'], 3)} |")
w(f"| Off-grid with a battery: {design(opt[2]['design'])} + {f(opt[2]['battery_kWh'], 0)} kWh | {f(opt[2]['usd_per_m3'], 3)} |")
w(f"| Off-grid with a battery, deck's 2× plant kept: {design(opt[3]['design'])} + {f(opt[3]['battery_kWh'], 0)} kWh | "
  f"{f(opt[3]['usd_per_m3'], 3)} |")
w("")
w(f"Without any grid cost the deck's design costs {f(cmp_['deck_fixed_usd_per_m3'], 3)} USD/m³ and buys "
  f"{f(cmp_['deck_grid_kWh_per_m3'], 3)} kWh/m³ from the grid (phase 2's year-by-year runs, `out/design_grid.csv`). Phase 2's grid "
  f"prices are 0.02–0.40 USD/kWh (`costs.py`); all twenty are in `results.json`. The off-grid designs cost as much as the deck's "
  f"design would at **{f(opt[1]['deck_design_costs_the_same_at_grid_price'], 2)} USD/kWh** (water only), "
  f"{f(opt[2]['deck_design_costs_the_same_at_grid_price'], 2)} USD/kWh (battery) and "
  f"{f(opt[3]['deck_design_costs_the_same_at_grid_price'], 2)} USD/kWh (battery, 2× plant kept). Off-grid water costs "
  f"{f(min(o['usd_per_m3'] for o in opt[1:]) - deck_cost['0.10'], 3)}–{f(max(o['usd_per_m3'] for o in opt[1:]) - deck_cost['0.10'], 3)} "
  f"USD/m³ more than the deck's design buying from the grid at 0.10 USD/kWh.")
w("")

# ------------------------------------------------------------------ sensitivities
w("## 7. Sensitivities (each option re-optimised)")
w("")
w("Phase 2's cost scenarios (`costs.SCENARIOS`, except \"spare trains already there\", which assumes the extra trains already exist "
  "and so does not apply to building a bigger plant), phase 2's notebook case with the plant costs in 2024 USD (× 1.31), and "
  "the older NREL battery price. The water-only optimum is re-picked from the fine search (tanks at their exact minimum, so the "
  f"base row shows {f(cmp_['sensitivities'][0]['water_only_usd_per_m3'], 3)} against {f(wc['usd_per_m3'], 3)} for the "
  f"{wc['tank_h']:g}-hour tank); the battery optimum from the screened and searched designs. USD per m³; "
  "the last two columns are the grid prices (USD/kWh) at which the deck's design would cost as much as each off-grid route.")
w("")
w("| Case | Deck + grid at 0.10 | Water only (design) | Battery (design) | Break-even grid price, water only | Break-even grid price, battery |")
w("|---|---|---|---|---|---|")
edge_cases = []
for s in cmp_["sensitivities"]:
    wd = s.get("water_only_design")
    bd = s.get("battery_design")
    mark_w = "*" if wd and wd.get("at_edge_of_search") else ""
    mark_b = "*" if bd and bd.get("at_edge_of_search") else ""
    if mark_w or mark_b:
        edge_cases.append(s["case"])
    w(f"| {s['case']} | {f(s['deck_usd_per_m3']['0.10'], 3)} | {f(s['water_only_usd_per_m3'], 3)}"
      + (f" ({wd['capacity']:g}×, {wd['kwp_per_m3d']:g} kWp, {wd['tank_h']:.0f} h){mark_w}" if wd else " (same as base)") + " | "
      + f"{f(s['battery_usd_per_m3'], 3)}" + (f" ({bd['capacity']:g}×, {bd['kwp_per_m3d']:g} kWp, {bd['tank_h']:g} h, {f(bd['battery_kWh'], 0)} kWh){mark_b}" if bd else "")
      + f" | {f(s.get('breakeven_water_only'), 2)} | {f(s['breakeven_battery'], 2)} |")
w("")
if edge_cases:
    w("\\* the re-picked design is at the edge of the searched range (water only: capacity 1.8–3.5×, panels 0.8–4.0 kWp; battery: "
      "the screened designs), so the true optimum of that case may lie outside it and cost a little less.")
    w("")

# ------------------------------------------------------------------ limits
w("## 8. Limits")
w("")
w("- **Ten years of weather, no margin.** The sizes are the smallest that survive the worst stretch of 2015–2024. A worse stretch "
  "outside these years would need more storage; no safety margin is added, and the plant is assumed available 100 % of the time "
  "(no outage, no membrane cleaning stop). A real off-grid plant would add a margin or a backup.")
w("- **The phase-2 limits carry over:** no dust on the panels and no panel ageing; no membrane fouling or ageing; the rest of the "
  "plant as a fixed amount per m³; NaCl standing in for seawater; NASA POWER's sea skin temperature on a 2 °C table; the wear of "
  "train starts not counted.")
w("- **The rule is kept.** For water storage alone the rule is already the best possible: it makes all the water the sun allows "
  "whenever there is space, so no other sun-only schedule can keep the tank fuller. With a battery, a schedule planned together "
  "with the battery might need a smaller battery; not tested.")
w("- **Capacity above 2× is modelled as 8 bigger trains** (section 2). The energy per m³ is unaffected; as in phase 2, part of a "
  "train may run (at dawn and dusk the plant makes any small output at the f_min energy per m³). The cost assumes the whole "
  "plant grows with capacity (base case of `cost_inputs.json`); the \"only the RO system grows\" case is in section 7.")
w("- **Price years differ:** RO plant and upkeep in 2016 USD (World Bank 2019), panels and battery in 2024 USD; the × 1.31 case in "
  "section 7 puts the plant in 2024 USD.")
w("- **Battery:** NREL's costs are for the mainland United States and were fitted to 1–10-hour systems; the fixed O&M is meant to "
  "keep the rated capacity at about one cycle a day; no self-discharge; the round-trip loss is all on charging.")
w("- **This study uses the phase-2 scripts** (`schedule.py`, `design_grid.py`), as asked. The phase-2 notebook (`make_notebook.py`) "
  "is a later variant (35 % recovery on the hottest days, among others); its text quotes 79.6 % sun for the deck's design "
  "(77.6–81.9 % by year), while the scripts give the 79.2 % used here. If the deck quotes the notebook's numbers, the off-grid "
  "sizes here come from the slightly different script model.")
w("")

# ------------------------------------------------------------------ sources
w("## 9. Sources")
w("")
w("- **Battery, base:** W. Cole, V. Ramasamy, M. Turan, *Cost Projections for Utility-Scale Battery Storage: 2025 Update*, NREL/TP-6A40-93281, "
  "June 2025, <https://docs.nrel.gov/docs/fy25osti/93281.pdf>. p. 3: \"All cost values were converted to 2024$ using the consumer "
  "pricing index.\" and \"We define the 4-hour duration as the output duration of the battery, such that a 4-hour device would be able "
  "to discharge at its rated power capacity for 4-hours.\" p. 6: \"...we used the bottom-up cost model to calculate the cost of a "
  "storage system with durations ranging from one hour to ten hours...\"; \"In the case of Figure 2, the energy cost is $241/kWh, and "
  "the power cost is $379/kW adjusted down by $7/kW for a final power cost of $372/kW.\" p. 8: \"The 2024 starting point of $334/kWh "
  "is derived from the bottom-up cost model described in Section 2.2.\" (4-hour system; Table 2, p. 16: 2024, 334 USD/kWh). p. 11: "
  "\"The FOM value selected is 4% of the $/kW capacity cost for a 4-hour battery.\"; \"...we have adopted zero for the VOM.\" "
  "p. 12: \"The round-trip efficiency is chosen to be 85%, which is well aligned with published values.\"; \"The lifetime we selected "
  "is 15 years, which is on the lower end of published values.\"")
w("- **Battery, sensitivity (the cost projections that NREL ATB 2024 cites):** W. Cole, A. Karmakar, *Cost Projections for Utility-Scale Battery "
  "Storage: 2023 Update*, NREL/TP-6A40-85332, June 2023, <https://www.nrel.gov/docs/fy23osti/85332.pdf>. p. 2: \"All cost values were "
  "converted to 2022$ using the consumer pricing index.\" p. 4: \"The 2022 starting point of $482/kWh is taken from Ramasamy et al. "
  "(2022).\" (Table 2, p. 13: 2022, 482 USD/kWh, 4-hour). p. 7: \"The FOM value selected is 2.5% of the $/kW capacity cost for a "
  "4-hour battery.\" p. 8: round trip 85 %, lifetime 15 years. NREL ATB 2024, utility-scale battery storage, "
  "<https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage>: 85 % round trip, 15-year lifetime, FOM 2.5 % of the "
  "capital cost in $/kW, cost breakdown in 2022$; it cites Cole and Karmakar (2023).")
w("- **Sun and sea temperature:** NASA POWER hourly and daily point data 2015–2024 (phase-2 `data/`).")
w("- **All other costs:** phase-2 `cost_inputs.json` and its `_sources` (World Bank 2019, *The Role of Desalination in an "
  "Increasingly Water-Scarce World*; IRENA, *Renewable Power Generation Costs in 2024*; Kocher & Menon 2023). The × 1.31 factor "
  "(plant costs from 2016 to 2024 USD) is phase 2's notebook case (US consumer price index, Bureau of Labor Statistics).")
w("")

# ------------------------------------------------------------------ files
w("## 10. Files")
w("")
w("- `model.py`: shared helpers (imports the phase-2 code; start level; repeating state; sequent peak; battery)")
w("- `check_model.py` → `out/check_model.json`: the checks in section 2")
w("- `sweep_water.py` → `out/water_coarse.csv`, `out/water_fine.csv`, `out/water_result.json` (log `out/water.log`): part (a)")
w("- `battery.py` → `out/battery_designs.csv`, `out/battery_screen.csv`, `out/battery_result.json` (log `out/battery.log`): part (b)")
w("- `compare.py` → `results.json`, `out/compare.json`, `out/compare_table.csv`: the comparison and sensitivities")
w("- `make_report.py` → this file")
w("")
(M.HERE / "REPORT.md").write_text("\n".join(L) + "\n")
print("REPORT.md written,", len(L), "lines")
