"""Part (a): off-grid with water storage only. The phase-2 rule, physics and costs, unchanged; the ten years run as one
series in a repeating state (model.run_cyclic). Off-grid = zero grid kWh in every year 2015-2024 and no short hours.

  coarse   direct simulation of the rule over capacity x panels x tank (beyond phase 2's 2x / 2.0 kWp / 24 h)
  exact    for each capacity and panel size, the smallest tank that never runs dry (sequent peak on the water the sun
           alone can make, minus the demand) - checked against the coarse simulations
  fine     the cheapest off-grid design per m3 (costs.py's annual_costs and cost_inputs.json, no grid)
  verify   the cheapest design simulated hour by hour; the same design with a slightly smaller tank must buy from the grid
  limits   what stops the deck's capacity (2x): the critical stretch, by month and by run of low-sun days

    python sweep_water.py      -> out/water_coarse.csv, out/water_fine.csv, out/water_result.json   (log: out/water.log)
"""
import json
import time

import numpy as np
import pandas as pd

import model as M

OUT = M.HERE / "out"
OUT.mkdir(exist_ok=True)
ZERO = 1e-6                     # kWh per year: below this the grid use is floating-point noise, counted as zero
t0 = time.time()
data = M.load()
N = len(data["pv"])
WATER_Y = M.D * N / len(M.YEARS)          # m3 a year in the repeating state (production = demand over the ten years)


def log(*a):
    print(f"[{time.time() - t0:6.1f} s]", *a, flush=True)


def simulate(C, P, S_h, curves=None):
    cv, keys, k = curves or M.curves(data, C)
    pv_kw = data["pv"] * P * 1000.0
    q, S0, it = M.run_cyclic(pv_kw, cv, S_h * M.D, keys, k)
    ys = M.year_scores(q, pv_kw, cv, keys, k, data["year"], S0)
    return q, S0, it, ys


def summary(ys):
    share = 100 * (1 - ys.grid_kWh / ys.energy_kWh)
    return dict(grid_kWh_year_max=float(ys.grid_kWh.max()), grid_kWh_year_mean=float(ys.grid_kWh.mean()),
                years_with_grid=int((ys.grid_kWh > ZERO).sum()), short_hours=int(ys.short_hours.sum()),
                sun_share_pct_mean=float(share.mean()), sun_share_pct_min=float(share.min()),
                unused_panel_energy_pct=float(100 * (1 - ys.solar_used_kWh.sum() / ys.pv_kWh.sum())),
                kWh_per_m3=float(ys.energy_kWh.sum() / ys.water_m3.sum()), water_m3_year=float(ys.water_m3.mean()),
                off_grid=bool((ys.grid_kWh <= ZERO).all() and ys.short_hours.sum() == 0))


def min_tank_h(C, P, curves=None):
    cv, keys, k = curves or M.curves(data, C)
    q_sun, _ = M.q_sun_series(data["pv"] * P * 1000.0, cv, keys, k)
    return M.min_storage(q_sun - M.D) / M.D


def cost_row(C, P, S_h, water=WATER_Y, x=None):
    x = x or M.cost_inputs()
    g = pd.DataFrame({"capacity": [C], "kwp_per_m3d": [P], "tank_h": [S_h]})
    a = M.CO.annual_costs(g, x).iloc[0] / water
    return a


def critical_period(x):
    """The stretch that sets the smallest store: from the last moment it was full to its lowest point (repeating state).
    Returns the hour indices of the stretch (modulo the ten years, so it may run from Dec 2024 into Jan 2015) and the store."""
    y = -np.cumsum(np.r_[x, x])
    run_min = np.minimum.accumulate(np.r_[0.0, y])[1:]
    K = y - run_min
    end = int(np.argmax(K))
    full = np.flatnonzero(K[:end + 1] <= 1e-9)
    start = int(full[-1]) + 1 if len(full) else 0
    return np.arange(start, end + 1) % N, float(K[end])


# ---------------------------------------------------------------- coarse: direct simulation
log("coarse sweep")
CAP = (2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
PV = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
TANK = (24, 36, 48, 72, 96, 120, 144, 168)
rows = []
for C in CAP:
    cvk = M.curves(data, C)
    for P in PV:
        k_min = min_tank_h(C, P, cvk)
        for S_h in TANK:
            _, S0, it, ys = simulate(C, P, S_h, cvk)
            r = dict(capacity=C, kwp_per_m3d=P, tank_h=S_h, min_tank_h_exact=k_min, start_level_m3=S0, iterations=it, **summary(ys))
            rows.append(r)
    log(f"capacity {C:g} done")
coarse = pd.DataFrame(rows)
coarse.to_csv(OUT / "water_coarse.csv", index=False)
agree = ((coarse.tank_h >= coarse.min_tank_h_exact) == coarse.off_grid)
log(f"coarse: {len(coarse)} designs, {int(coarse.off_grid.sum())} off-grid; exact min-tank agrees with simulation in "
    f"{int(agree.sum())} of {len(coarse)}")

# ---------------------------------------------------------------- fine: exact smallest tank and the cheapest design
log("fine search")
CAP_F = np.round(np.arange(1.80, 3.501, 0.05), 2)
PV_F = np.round(np.arange(0.80, 4.001, 0.05), 2)
x0 = M.cost_inputs()
fine = []
for C in CAP_F:
    cvk = M.curves(data, C)
    for P in PV_F:
        h = min_tank_h(C, P, cvk)
        r = dict(capacity=C, kwp_per_m3d=P, min_tank_h=h)
        if np.isfinite(h):
            a = cost_row(C, P, h, x=x0)
            r.update({f"usd_m3_{k}": float(v) for k, v in a.items()}, usd_m3_total=float(a.sum()))
        fine.append(r)
fine = pd.DataFrame(fine)
fine.to_csv(OUT / "water_fine.csv", index=False)
feas = fine[np.isfinite(fine.min_tank_h)]
best = feas.loc[feas.usd_m3_total.idxmin()]
log(f"fine: {len(fine)} (capacity, panels) pairs, {len(feas)} can be off-grid; cheapest {best.capacity:g}x, "
    f"{best.kwp_per_m3d:g} kWp, tank {best.min_tank_h:.2f} h, {best.usd_m3_total:.4f} USD/m3")
on_edge = dict(capacity_at_edge=bool(best.capacity in (CAP_F.min(), CAP_F.max())),
               panels_at_edge=bool(best.kwp_per_m3d in (PV_F.min(), PV_F.max())))
near = feas[feas.usd_m3_total <= best.usd_m3_total * 1.01].sort_values("usd_m3_total")
smallest_cap = feas.groupby("capacity").usd_m3_total.min()
lowest_feasible_capacity = float(feas.capacity.min())

# ---------------------------------------------------------------- verify the cheapest design, tank rounded up to a whole hour
C_b, P_b = float(best.capacity), float(best.kwp_per_m3d)
S_b = float(np.ceil(best.min_tank_h))
q, S0, it, ys = simulate(C_b, P_b, S_b)
ver = summary(ys)
ver.update(start_level_m3=S0, iterations=it, tank_full_hours_per_year_min=None)
cvk = M.curves(data, C_b)
S = S0 + np.cumsum(q - M.D)
ver["tank_full_hours_by_year"] = {int(y): int((S[data["year"] == y] >= S_b * M.D - 1e-6).sum()) for y in M.YEARS}
ver["tank_lowest_m3_by_year"] = {int(y): float(S[data["year"] == y].min()) for y in M.YEARS}
ver.pop("tank_full_hours_per_year_min")
_, _, _, ys_small = simulate(C_b, P_b, float(best.min_tank_h) - 0.5)
ver_small = summary(ys_small)
cost_b = cost_row(C_b, P_b, S_b, water=float(ys.water_m3.mean()), x=x0)
log(f"verified {C_b:g}x {P_b:g} kWp {S_b:g} h: off-grid {ver['off_grid']}; with {best.min_tank_h - 0.5:.2f} h: "
    f"off-grid {ver_small['off_grid']} ({ver_small['grid_kWh_year_max']:.1f} kWh in the worst year)")

# the critical stretch for the cheapest design
tt = data["time"]
daily_pv = pd.Series(data["pv"], index=tt).resample("D").sum()
med = daily_pv.groupby(daily_pv.index.month).transform("median")
bad = daily_pv < 0.5 * med                          # phase 2, step 1: a low-sun day is under half the month's median
bad_dates = set(bad[bad].index.date)


def stretch_report(C, P):
    cvk = M.curves(data, C)
    q_sun, q_max = M.q_sun_series(data["pv"] * P * 1000.0, *cvk)
    idx, K = critical_period(q_sun - M.D)
    if not np.isfinite(K) or K <= 0:
        return dict(capacity=C, kwp_per_m3d=P, tank_h_needed=K / M.D)
    times = tt[idx]
    net = pd.Series((q_sun - M.D)[idx])
    dates = pd.Series(times.date)
    day = net.groupby(dates.values).sum() / M.D     # hours of demand gained (+) or lost (-) per day
    return dict(capacity=C, kwp_per_m3d=P, tank_h_needed=K / M.D, start=str(times[0]), end=str(times[-1]),
                days=float(len(idx) / 24), days_sun_short_of_demand=int((day < 0).sum()), days_in_stretch=int(len(day)),
                low_sun_days_in_stretch=int(sum(d in bad_dates for d in day.index)),
                months=sorted({int(m) for m in times.month}),
                sun_alone_makes_pct_of_demand_in_stretch=float(100 * (net.sum() + M.D * len(idx)) / (M.D * len(idx))))


crit_best = stretch_report(C_b, P_b)

# ---------------------------------------------------------------- what stops the deck's capacity (2x)

stops = [stretch_report(2.0, P) for P in (3.0, 4.0, 6.0, 8.0)] + [stretch_report(2.5, 2.0), stretch_report(3.0, 2.0)]

# 1.5x and below: even unlimited sun cannot make a day's water in daylight in some seasons
cap_check = {}
for C in (1.5, 1.75, 2.0):
    cv, keys, k = M.curves(data, C)
    q_sun, q_max = M.q_sun_series(data["pv"] * 1e9, cv, keys, k)          # unlimited panels: full output whenever pv > 0
    day = pd.Series(q_sun, index=tt).resample("D").sum() / (24 * M.D)
    month = day.groupby(day.index.month).mean()
    cap_check[f"{C:g}x"] = dict(days_below_demand_pct=float(100 * (day < 1).mean()),
                                ten_year_output_vs_demand=float(q_sun.sum() / (M.D * N)),
                                month_mean_output_vs_demand={int(m): round(float(v), 3) for m, v in month.items()},
                                min_tank_h_unlimited_panels=M.min_storage(q_sun - M.D) / M.D)

# the closest design inside phase 2's sizes (2x, 2.0 kWp, 24 h): where and when it still buys from the grid
cv2 = M.curves(data, 2.0)
pv_kw = data["pv"] * 2.0 * 1000.0
q2, S02, _ = M.run_cyclic(pv_kw, cv2[0], 24 * M.D, cv2[1], cv2[2])
E2 = M.energy_of(q2, *cv2)
g2 = pd.Series(E2 - np.minimum(E2, pv_kw), index=tt)
gd = g2.resample("D").sum()
gm = g2.groupby(g2.index.month).sum() / g2.sum()
runs = (gd > ZERO).astype(int)
run_len = runs.groupby((runs == 0).cumsum()).sum()
largest = dict(grid_share_by_month_pct={int(m): round(float(100 * v), 2) for m, v in gm.items()},
               days_with_grid_per_year=float((gd > ZERO).sum() / 10), longest_run_days_with_grid=int(run_len.max()),
               low_sun_days_per_year=float(bad.sum() / 10),
               longest_low_sun_run_days=int(bad.astype(int).groupby((~bad).cumsum()).sum().max()),
               worst_day=str(daily_pv.idxmin().date()), worst_day_kWh_per_kWp=float(daily_pv.min()))

res = dict(definition="zero grid kWh (<= 1e-6 kWh, floating-point noise) in every year 2015-2024 and no short hours; ten years "
                      "as one series, tank level on 1 Jan 2015 = level on 31 Dec 2024",
           coarse=dict(designs=len(coarse), off_grid=int(coarse.off_grid.sum()), exact_min_tank_agrees=int(agree.sum()),
                       grid=dict(capacity=CAP, kwp_per_m3d=PV, tank_h=TANK)),
           fine=dict(capacity=[float(CAP_F.min()), float(CAP_F.max()), 0.05], kwp_per_m3d=[float(PV_F.min()), float(PV_F.max()), 0.05],
                     pairs=len(fine), pairs_off_grid_possible=len(feas), lowest_feasible_capacity=lowest_feasible_capacity,
                     cheapest_by_capacity={f"{c:g}": float(v) for c, v in smallest_cap.items()}, **on_edge),
           cheapest=dict(capacity=C_b, kwp_per_m3d=P_b, min_tank_h_exact=float(best.min_tank_h), tank_h=S_b,
                         usd_per_m3=float(cost_b.sum()), usd_per_m3_parts={k: float(v) for k, v in cost_b.items()},
                         critical_stretch=crit_best, check=ver,
                         check_tank_minus_half_hour=dict(tank_h=float(best.min_tank_h) - 0.5, off_grid=ver_small["off_grid"],
                                                         grid_kWh_worst_year=ver_small["grid_kWh_year_max"],
                                                         years_with_grid=ver_small["years_with_grid"])),
           within_1pct_of_cheapest=near[["capacity", "kwp_per_m3d", "min_tank_h", "usd_m3_total"]].to_dict("records"),
           what_stops_2x=stops, capacity_limit_unlimited_panels=cap_check, phase2_largest_design_2x_2kWp_24h=largest,
           run_seconds=time.time() - t0)
(OUT / "water_result.json").write_text(json.dumps(res, indent=1, default=float))
log("done")
print(json.dumps({k: res[k] for k in ("coarse", "cheapest")}, indent=1, default=float))
