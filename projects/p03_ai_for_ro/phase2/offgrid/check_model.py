"""Checks before trusting the model for the off-grid study.

  1  the phase-2 baseline, re-run with the phase-2 code (design_grid.rule_fast + score_fast), equals out/design_grid.csv
  2  rule_s0 (the copy with a start level) gives exactly rule_fast's output when it starts empty
  3  capacities above 2x: how schedule.curves builds a capacity, and the fast rule against the slow reference rule
  4  the ten years as one series (repeating state) against phase 2's year-by-year runs, for the deck's design
  5  why the phase-2 set-up can never show zero grid: every year starts with an empty tank at midnight

    python check_model.py      -> out/check_model.json
"""
import json

import numpy as np
import pandas as pd

import model as M

OUT = M.HERE / "out"
OUT.mkdir(exist_ok=True)
data = M.load()
res = {}

# ---------------------------------------------------------------- 1  baseline
grid_csv = pd.read_csv(M.PHASE2 / "out" / "design_grid.csv")
base = {}
for (C, P, S_h) in ((2.0, 0.8, 12), (2.0, 2.0, 24)):
    rows = []
    for y in M.YEARS:
        pv, T = data["pv_y"][y], data["T_y"][y]
        cv = M.Sc.curves(data["perf"], data["other"], C)
        keys, k = M.DG.groups(cv, T)
        q = M.DG.rule_fast(pv * P * 1000.0, cv, S_h * M.D, keys, k)
        s = M.DG.score_fast(q, pv * P * 1000.0, cv, keys, k)
        s.update(year=y)
        rows.append(s)
    t = pd.DataFrame(rows).set_index("year")
    ref = grid_csv[(grid_csv.capacity == C) & (grid_csv.kwp_per_m3d == P) & (grid_csv.tank_h == S_h)].set_index("year")
    cols = ["water_m3", "energy_kWh", "grid_kWh", "solar_used_kWh", "pv_kWh", "short_hours"]
    max_rel = float((t[cols] - ref[cols]).abs().max().max())
    share = 100 * (1 - t.grid_kWh / t.energy_kWh)
    share_csv = 100 * (1 - ref.grid_kWh / ref.energy_kWh)
    unused = 100 * (1 - ref.solar_used_kWh.sum() / ref.pv_kWh.sum())
    unused_mean_of_years = float((100 * (1 - ref.solar_used_kWh / ref.pv_kWh)).mean())
    key = f"{C:g}x, {P:g} kWp per m3/d, {S_h} h"
    base[key] = dict(max_abs_diff_vs_design_grid_csv=max_rel,
                     sun_share_pct_mean=float(share_csv.mean()), sun_share_pct_min=float(share_csv.min()),
                     sun_share_pct_max=float(share_csv.max()), sun_share_pct_rerun_mean=float(share.mean()),
                     unused_panel_energy_pct_10y_total=float(unused), unused_panel_energy_pct_mean_of_years=unused_mean_of_years,
                     grid_kWh_per_year_mean=float(ref.grid_kWh.mean()), water_m3_per_year_mean=float(ref.water_m3.mean()),
                     energy_kWh_per_m3=float(ref.energy_kWh.sum() / ref.water_m3.sum()),
                     short_hours_total=int(ref.short_hours.sum()))
    print(key, json.dumps(base[key], indent=1))
res["1_baseline"] = base

# ---------------------------------------------------------------- 2  rule_s0 == rule_fast when starting empty
diffs = []
for (C, P, S_h) in ((2.0, 0.8, 12), (2.0, 2.0, 24), (4.0, 6.0, 168), (3.0, 3.0, 72)):
    for y in M.YEARS:
        pv, T = data["pv_y"][y], data["T_y"][y]
        cv = M.Sc.curves(data["perf"], data["other"], C)
        keys, k = M.DG.groups(cv, T)
        a = M.DG.rule_fast(pv * P * 1000.0, cv, S_h * M.D, keys, k)
        b, _ = M.rule_s0(pv * P * 1000.0, cv, S_h * M.D, keys, k, 0.0)
        diffs.append(float(np.abs(a - b).max()))
res["2_rule_s0_equals_rule_fast_max_abs_diff_m3h"] = max(diffs)
print("rule_s0 vs rule_fast, max |diff| (m3/h):", max(diffs))

# ---------------------------------------------------------------- 3  capacities above 2x
cv2 = M.Sc.curves(data["perf"], data["other"], 2.0)
chk = {"trains": M.Sc.TRAINS, "train_size_m3h_at_design_flux": {}}
worst = 0.0
for C in (2.5, 3.0, 4.0, 5.0):
    cvC = M.Sc.curves(data["perf"], data["other"], C)        # also runs the convexity assert inside schedule.curves
    chk["train_size_m3h_at_design_flux"][f"{C:g}x"] = float(cvC[25.0]["Q_tr"])
    for T in cv2:
        a, b = cv2[T], cvC[T]
        r = C / 2.0
        worst = max(worst, float(np.abs(b["q"] - r * a["q"]).max()), float(np.abs(b["E"] - r * a["E"]).max()),
                    float(np.abs(b["cp"] - a["cp"]).max()), abs(b["f_min"] - a["f_min"]), abs(b["f_max"] - a["f_max"]),
                    abs(b["q_max"] - C * M.D * b["f_max"]))
chk["train_size_m3h_at_design_flux"]["2x"] = float(cv2[25.0]["Q_tr"])
chk["max_deviation_from_exact_scaling"] = worst
chk["f_window_by_T"] = {f"{T:g}": [round(cv2[T]["f_min"], 4), round(cv2[T]["f_max"], 4)] for T in sorted(cv2)}
chk["kWh_per_m3_at_f_min_and_f_max_by_T"] = {f"{T:g}": [float(cv2[T]["E"][1] / cv2[T]["q"][1]), float(cv2[T]["E"][-1] / cv2[T]["q"][-1])]
                                            for T in sorted(cv2)}
print("capacity scaling: max deviation", worst)

# fast rule vs the slow reference rule (schedule.run_rule) and scores (schedule.score vs design_grid.score_fast), 4x
for (C, P, S_h, y) in ((4.0, 4.0, 72, 2024), (3.0, 2.5, 48, 2022)):
    pv, T = data["pv_y"][y], data["T_y"][y]
    cv = M.Sc.curves(data["perf"], data["other"], C)
    keys, k = M.DG.groups(cv, T)
    q_fast = M.DG.rule_fast(pv * P * 1000.0, cv, S_h * M.D, keys, k)
    q_ref = M.Sc.run_rule(pv * P * 1000.0, T, cv, S_h * M.D)
    s_ref = M.Sc.score(q_ref, pv, T, cv, P * 1000.0)
    s_fast = M.DG.score_fast(q_fast, pv * P * 1000.0, cv, keys, k)
    chk[f"rule_fast_vs_run_rule_{C:g}x_{P:g}kWp_{S_h}h_{y}"] = dict(
        max_abs_diff_q=float(np.abs(q_fast - q_ref).max()),
        grid_kWh_per_m3_ref=s_ref["grid_kWh_per_m3"], grid_kWh_per_m3_fast=s_fast["grid_kWh"] / s_fast["water_m3"],
        solar_share_ref=s_ref["solar_share_pct"], solar_share_fast=100 * s_fast["solar_used_kWh"] / s_fast["energy_kWh"],
        kWh_per_m3=s_ref["kWh_per_m3"], trains_on_max=int(max(M.Sc.trains_on(M.Sc.nearest(cv, T[t]), q_ref[t]) for t in range(len(q_ref)))),
        train_starts_per_day=s_fast["train_starts_per_day"])
    print(C, P, S_h, y, chk[f"rule_fast_vs_run_rule_{C:g}x_{P:g}kWp_{S_h}h_{y}"])
res["3_capacity_above_2x"] = chk

# ---------------------------------------------------------------- 4  ten years as one series, deck design
cv, keys, k = M.curves(data, 2.0)
pv_kw = data["pv"] * 0.8 * 1000.0
q, S0, it = M.run_cyclic(pv_kw, cv, 12 * M.D, keys, k)
ys = M.year_scores(q, pv_kw, cv, keys, k, data["year"], S0)
share = 100 * (1 - ys.grid_kWh / ys.energy_kWh)
res["4_continuous_deck_design"] = dict(start_level_m3=S0, iterations=it, sun_share_pct_mean=float(share.mean()),
                                       sun_share_pct_min=float(share.min()), sun_share_pct_max=float(share.max()),
                                       grid_kWh_per_year_mean=float(ys.grid_kWh.mean()),
                                       short_hours_total=int(ys.short_hours.sum()))
print("continuous deck design:", res["4_continuous_deck_design"])

# ---------------------------------------------------------------- 5  the phase-2 start: empty tank at midnight on 1 January
first = {}
for y in M.YEARS:
    pv, T = data["pv_y"][y], data["T_y"][y]
    cvy = M.Sc.curves(data["perf"], data["other"], 4.0)
    keys_y, k_y = M.DG.groups(cvy, T)
    qq = M.DG.rule_fast(pv * 6.0 * 1000.0, cvy, 168 * M.D, keys_y, k_y)       # a very large design
    E = M.energy_of(qq, cvy, keys_y, k_y)
    g = E - np.minimum(E, pv * 6000.0)
    first[str(y)] = dict(grid_kWh_year=float(g.sum()), grid_kWh_before_first_sunrise=float(g[: int(np.argmax(pv > 0))].sum()))
res["5_phase2_start_empty_4x_6kWp_168h"] = first
print("empty start, 4x / 6 kWp / 168 h:", first)

(OUT / "check_model.json").write_text(json.dumps(res, indent=1))
