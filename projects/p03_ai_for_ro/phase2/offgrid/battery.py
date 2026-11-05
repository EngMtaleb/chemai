"""Part (b): a battery added to the phase-2 rule. The rule decides what the plant makes, exactly as in phase 2 (unchanged);
the battery charges only from sun the plant cannot use and supplies the plant whenever the rule would buy from the grid.
Ten years as one series, tank and battery in a repeating state (level on 1 Jan 2015 = level on 31 Dec 2024).

Battery inputs (public source, quoted in REPORT.md):
  base         NREL, Cole, Ramasamy & Turan (2025), Cost Projections for Utility-Scale Battery Storage: 2025 Update,
               NREL/TP-6A40-93281: energy cost 241 USD/kWh and power cost 372 USD/kW (p. 6), 2024 USD (p. 3);
               fixed O&M 4 % of the capital cost a year, no variable O&M (p. 11); round trip 85 %, life 15 years (p. 12).
  sensitivity  NREL, Cole & Karmakar (2023), ... 2023 Update, NREL/TP-6A40-85332 (the battery input of ATB 2024):
               4-hour system 482 USD/kWh in 2022 (p. 4, Table 2 p. 13), 2022 USD (p. 2); fixed O&M 2.5 % (p. 7);
               round trip 85 %, life 15 years (p. 8).
Assumptions (mine): the rated energy is the energy the battery delivers (NREL defines duration as output duration, p. 3),
so the 15 % round-trip loss is taken on charging; no self-discharge; one power rating for charge and discharge; fixed O&M
as a share of this battery's own capital cost; the battery is annualised over its own 15-year life at the phase-2
discount rate (same method as costs.py uses for each part).

    python battery.py      -> out/battery_designs.csv, out/battery_screen.csv, out/battery_result.json  (log: out/battery.log)
"""
import json
import time

import numpy as np
import pandas as pd

import model as M

OUT = M.HERE / "out"
OUT.mkdir(exist_ok=True)
ZERO = 1e-6
t0 = time.time()

BASE = dict(name="NREL 2025 update (2024 USD)", e_cost=241.0, p_cost=372.0, fom=0.04, eta=0.85, life=15)
SENS = dict(name="NREL 2023 update / ATB 2024 (2022 USD), 4-hour system", e_cost_4h=482.0, fom=0.025, eta=0.85, life=15)

data = M.load()
N = len(data["pv"])
X0 = M.cost_inputs()


def log(*a):
    print(f"[{time.time() - t0:6.1f} s]", *a, flush=True)


def flows(C, P, S_h):
    """The phase-2 rule (repeating state): what the plant makes, its energy, the sun it cannot use, the grid it would buy."""
    cv, keys, k = M.curves(data, C)
    pv_kw = data["pv"] * P * 1000.0
    q, S0, it = M.run_cyclic(pv_kw, cv, S_h * M.D, keys, k)
    E = M.energy_of(q, cv, keys, k)
    used = np.minimum(E, pv_kw)
    return dict(q=q, S0=S0, E=E, pv_kw=pv_kw, surplus=pv_kw - used, deficit=E - used, cv=(cv, keys, k))


def battery_annual(capex, fom, life, rate):
    return capex * (M.crf(rate, life) + fom)


def size_base(f, n_grid=80):
    """Smallest energy with power not limiting, and the cheapest (energy, power) pair at the NREL 2025 prices."""
    s, d, eta = f["surplus"], f["deficit"], BASE["eta"]
    if d.max() <= 0:                                   # already off-grid without a battery
        return dict(feasible=True, E_min_kWh=0.0, P_needed_kW=0.0, E_kWh=0.0, P_kW=0.0, capex_usd=0.0)
    if eta * s.sum() < d.sum():                        # over the ten years the spare sun, after losses, is too little
        return dict(feasible=False, E_min_kWh=np.inf)
    E_min = M.min_storage(eta * s - d)
    P_lo, P_hi = float(d.max()), float(max(s.max(), d.max()))
    Ps = np.unique(np.r_[P_lo, np.geomspace(P_lo, P_hi, n_grid)])
    E = np.array([M.battery_min_energy(s, d, p, eta) for p in Ps])
    capex = BASE["e_cost"] * E + BASE["p_cost"] * Ps
    j = int(np.nanargmin(np.where(np.isfinite(capex), capex, np.nan)))
    lo, hi = Ps[max(j - 1, 0)], Ps[min(j + 1, len(Ps) - 1)]           # refine between the neighbours
    Pf = np.linspace(lo, hi, 41)
    Ef = np.array([M.battery_min_energy(s, d, p, eta) for p in Pf])
    cf = BASE["e_cost"] * Ef + BASE["p_cost"] * Pf
    jf = int(np.nanargmin(np.where(np.isfinite(cf), cf, np.nan)))
    return dict(feasible=True, E_min_kWh=E_min, P_needed_kW=P_lo, E_kWh=float(Ef[jf]), P_kW=float(Pf[jf]),
                capex_usd=float(cf[jf]))


def size_4h(f):
    """NREL 2023 / ATB 2024 sensitivity: a 4-hour battery (power = energy / 4); smallest energy by bisection."""
    s, d, eta = f["surplus"], f["deficit"], SENS["eta"]
    if eta * s.sum() < d.sum():
        return np.inf
    lo, hi = 4 * float(d.max()), max(4 * float(d.max()), 1.0)
    while M.battery_min_energy(s, d, hi / 4, eta) > hi:
        hi *= 2
        if hi > 1e9:
            return np.inf
    if M.battery_min_energy(s, d, lo / 4, eta) <= lo:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if M.battery_min_energy(s, d, mid / 4, eta) <= mid else (mid, hi)
    return hi


def costs_per_m3(C, P, S_h, water, batt_capex, batt_fom, x=None, life=15):
    x = x or X0
    g = pd.DataFrame({"capacity": [C], "kwp_per_m3d": [P], "tank_h": [S_h]})
    a = M.CO.annual_costs(g, x).iloc[0]
    a["battery"] = battery_annual(batt_capex, batt_fom, life, x["discount_rate"])
    return a / water


def detail(C, P, S_h):
    """Full result for one design: sizes, hour-by-hour check, costs, the stretch that sets the size."""
    f = flows(C, P, S_h)
    s, d, eta = f["surplus"], f["deficit"], BASE["eta"]
    years = data["year"]
    by_year = {int(y): dict(grid_kWh_without_battery=float(d[years == y].sum()),
                            sun_it_cannot_use_kWh=float(s[years == y].sum()),
                            stored_if_all_charged_kWh=float(eta * s[years == y].sum())) for y in M.YEARS}
    out = dict(capacity=C, kwp_per_m3d=P, tank_h=S_h, tank_start_m3=f["S0"],
               ten_year_grid_without_battery_kWh=float(d.sum()), ten_year_sun_it_cannot_use_kWh=float(s.sum()),
               ten_year_store_if_all_charged_kWh=float(eta * s.sum()), by_year=by_year)
    sz = size_base(f)
    out.update(sz)
    if not sz["feasible"] or not np.isfinite(sz.get("E_min_kWh", np.inf)):
        out["off_grid_possible"] = False
        return out, f
    out["off_grid_possible"] = True
    E_b, P_b = float(np.ceil(sz["E_kWh"])), float(np.ceil(sz["P_kW"]))
    r = M.battery_cyclic(s, d, E_b, P_b, eta)
    ys = M.year_scores(f["q"], f["pv_kw"], *f["cv"], years, f["S0"], batt=r)
    small = M.battery_cyclic(s, d, 0.99 * sz["E_kWh"], P_b, eta)
    water = float(ys.water_m3.mean())
    capex = BASE["e_cost"] * E_b + BASE["p_cost"] * P_b
    c = costs_per_m3(C, P, S_h, water, capex, BASE["fom"])
    E4 = size_4h(f)
    c4 = costs_per_m3(C, P, S_h, water, SENS["e_cost_4h"] * np.ceil(E4), SENS["fom"]) if np.isfinite(E4) else None
    idx, K = critical_period(eta * np.minimum(s, P_b) - d)
    tt = data["time"][idx]
    out.update(check=dict(battery_kWh=E_b, battery_kW=P_b, duration_h=E_b / P_b,
                          grid_kWh_by_year={int(y): float(g) for y, g in zip(ys.year, ys.grid_kWh)},
                          off_grid=bool((ys.grid_kWh <= ZERO).all() and ys.short_hours.sum() == 0),
                          short_hours=int(ys.short_hours.sum()), battery_start_kWh=r["B0"],
                          battery_full_hours_by_year={int(y): int(v) for y, v in zip(ys.year, ys.batt_full_hours)},
                          battery_lowest_kWh_by_year={int(y): float(v) for y, v in zip(ys.year, ys.batt_min_kWh)},
                          with_99pct_energy_grid_kWh_10y=float(small["grid"].sum())),
               discharge_kWh_per_year=float(ys.batt_out_kWh.mean()), full_cycles_per_year=float(ys.batt_out_kWh.mean() / E_b),
               battery_loss_kWh_per_year=float((ys.batt_in_kWh - ys.batt_out_kWh).mean()),
               unused_panel_energy_pct=float(100 * ys.curtailed_kWh.sum() / ys.pv_kWh.sum()),
               unused_panel_energy_pct_without_battery=float(100 * s.sum() / f["pv_kw"].sum()),
               sun_share_pct=float(100 * (1 - ys.grid_kWh.sum() / ys.energy_kWh.sum())),
               sun_share_pct_without_battery=float(100 * (1 - d.sum() / f["E"].sum())),
               kWh_per_m3=float(ys.energy_kWh.sum() / ys.water_m3.sum()), water_m3_year=water,
               usd_per_m3=float(c.sum()), usd_per_m3_parts={k: float(v) for k, v in c.items()},
               sensitivity_4h_2023=dict(E_kWh=float(np.ceil(E4)) if np.isfinite(E4) else None,
                                        usd_per_m3=float(c4.sum()) if c4 is not None else None),
               critical_stretch=dict(start=str(tt[0]), end=str(tt[-1]), days=float(len(idx) / 24), kWh=K,
                                     months=sorted({int(m) for m in tt.month})))
    return out, f


def critical_period(x):
    y = -np.cumsum(np.r_[x, x])
    run_min = np.minimum.accumulate(np.r_[0.0, y])[1:]
    K = y - run_min
    end = int(np.argmax(K))
    full = np.flatnonzero(K[:end + 1] <= 1e-9)
    start = int(full[-1]) + 1 if len(full) else 0
    return np.arange(start, end + 1) % N, float(K[end])


# ---------------------------------------------------------------- the deck's design and three larger ones
log("named designs")
named = []
for (C, P, S_h) in ((2.0, 0.8, 12), (2.0, 1.2, 12), (2.0, 1.6, 24), (2.0, 2.0, 24)):
    o, _ = detail(C, P, S_h)
    named.append(o)
    log(f"{C:g}x {P:g} kWp {S_h} h: possible {o['off_grid_possible']}, E_min {o.get('E_min_kWh')}, "
        f"cost {o.get('usd_per_m3')}")
pd.DataFrame([{k: v for k, v in o.items() if not isinstance(v, dict)} for o in named]).to_csv(OUT / "battery_designs.csv", index=False)

# ---------------------------------------------------------------- screen: the cheapest design with a battery
log("screen")
CAP = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
PV = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
TANK = (0, 3, 6, 12, 18, 24, 36, 48)
HAS_BATTERY = 0.5          # kWh: a design counts as a battery design only if it needs more than this


def screen_one(C, P, S_h, with_4h=True):
    f = flows(C, P, S_h)
    sz = size_base(f, n_grid=40)
    row = dict(capacity=C, kwp_per_m3d=P, tank_h=S_h, **{k: v for k, v in sz.items()})
    if sz["feasible"] and np.isfinite(sz.get("E_min_kWh", np.inf)) and "E_kWh" in sz:
        water = float(f["q"].sum() / len(M.YEARS))
        c = costs_per_m3(C, P, S_h, water, sz["capex_usd"], BASE["fom"])
        row.update(water_m3_year=water, usd_per_m3=float(c.sum()), **{f"usd_m3_{k}": float(v) for k, v in c.items()})
        if with_4h:
            E4 = size_4h(f)
            row["E_4h_kWh"] = E4
            if np.isfinite(E4):
                row["usd_per_m3_4h_2023"] = float(costs_per_m3(C, P, S_h, water, SENS["e_cost_4h"] * E4, SENS["fom"]).sum())
    return row


rows = []
for C in CAP:
    for P in PV:
        for S_h in TANK:
            rows.append(screen_one(C, P, S_h))
    log(f"capacity {C:g} done")
scr = pd.DataFrame(rows)
scr["stage"] = "screen"
ok = scr[scr.usd_per_m3.notna() & (scr.E_kWh > HAS_BATTERY)]
b0 = ok.loc[ok.usd_per_m3.idxmin()]
log(f"screen best: {b0.capacity:g}x {b0.kwp_per_m3d:g} kWp {b0.tank_h:g} h, {b0.usd_per_m3:.4f} USD/m3")

# local search from the best of the screen: move to a cheaper neighbour (capacity, panels, tank), halving the steps
# when no neighbour is cheaper, down to 0.05x, 0.05 kWp and 1 h; only designs that need a battery count
LEVELS = ((0.2, 0.4, 6.0), (0.1, 0.2, 3.0), (0.05, 0.1, 1.0), (0.05, 0.05, 1.0))


def local_search(start, stage, fix_capacity=False):
    cache = {}

    def cost(x):
        key = (round(x[0], 4), round(x[1], 4), round(x[2], 4))
        if key not in cache:
            if key[0] < 1.0 or key[1] <= 0 or key[2] < 0:
                cache[key] = (np.inf, None)
            else:
                r = screen_one(*key, with_4h=False)
                good = r.get("usd_per_m3") is not None and np.isfinite(r.get("usd_per_m3", np.nan)) and r.get("E_kWh", 0) > HAS_BATTERY
                cache[key] = (r["usd_per_m3"] if good else np.inf, r)
        return cache[key][0]

    x = [float(v) for v in start]
    fx = cost(x)
    for lv in LEVELS:
        while True:
            moved = False
            for i in ((1, 2) if fix_capacity else (0, 1, 2)):
                for sgn in (1, -1):
                    y = list(x)
                    y[i] = round(y[i] + sgn * lv[i], 4)
                    fy = cost(y)
                    if fy < fx - 1e-12:
                        x, fx, moved = y, fy, True
            if not moved:
                break
    rows_ = [dict(r, stage=stage) for (c_, r) in cache.values() if r is not None]
    return x, fx, pd.DataFrame(rows_)


log("local search")
x_best, f_best, fr = local_search((b0.capacity, b0.kwp_per_m3d, b0.tank_h), "local search")
log(f"local search: {x_best} -> {f_best:.4f} USD/m3 ({len(fr)} designs)")
ok2 = ok[ok.capacity == 2.0]
b2 = ok2.loc[ok2.usd_per_m3.idxmin()]
x2, f2, fr2 = local_search((2.0, b2.kwp_per_m3d, b2.tank_h), "local search at 2x", fix_capacity=True)
log(f"local search at 2x: {x2} -> {f2:.4f} USD/m3 ({len(fr2)} designs)")
allr = pd.concat([scr, fr, fr2], ignore_index=True)
allr.to_csv(OUT / "battery_screen.csv", index=False)
best, _ = detail(*x_best)
log(f"cheapest with a battery: {x_best} -> {best['usd_per_m3']:.4f} USD/m3, battery {best['check']['battery_kWh']:.0f} kWh / "
    f"{best['check']['battery_kW']:.0f} kW, off-grid {best['check']['off_grid']}")
best2, _ = detail(*x2)
log(f"cheapest with a battery at 2x: {x2} -> {best2['usd_per_m3']:.4f} USD/m3, battery {best2['check']['battery_kWh']:.0f} kWh, "
    f"off-grid {best2['check']['off_grid']}")
ok4 = ok[ok.E_4h_kWh.notna() & np.isfinite(ok.E_4h_kWh) & (ok.E_4h_kWh > HAS_BATTERY)] if "E_4h_kWh" in ok else ok.iloc[0:0]
best_4h = ok4.loc[ok4.usd_per_m3_4h_2023.idxmin()] if len(ok4) else None
near = allr[(allr.usd_per_m3 <= f_best * 1.01) & (allr.E_kWh > HAS_BATTERY)].drop_duplicates(["capacity", "kwp_per_m3d", "tank_h"])

res = dict(inputs=dict(base=BASE, sensitivity=SENS, discount_rate=X0["discount_rate"],
                       crf_15y=M.crf(X0["discount_rate"], 15)),
           named_designs=named,
           screen=dict(grid=dict(capacity=CAP, kwp_per_m3d=PV, tank_h=TANK), designs=int(len(scr)),
                       off_grid_possible=int(scr.usd_per_m3.notna().sum()), with_a_battery=int(len(ok)),
                       best_on_screen=dict(capacity=float(b0.capacity), kwp_per_m3d=float(b0.kwp_per_m3d), tank_h=float(b0.tank_h),
                                           usd_per_m3=float(b0.usd_per_m3)),
                       local_search_designs=int(len(fr)), local_search_steps=LEVELS,
                       capacity_at_lower_bound_1x=bool(abs(x_best[0] - 1.0) < 1e-9),
                       within_1pct_of_cheapest=near[["capacity", "kwp_per_m3d", "tank_h", "E_kWh", "P_kW", "usd_per_m3"]].to_dict("records")),
           cheapest=best,
           cheapest_at_2x=best2,
           cheapest_with_2023_4h_prices=None if best_4h is None else dict(
               capacity=float(best_4h.capacity), kwp_per_m3d=float(best_4h.kwp_per_m3d), tank_h=float(best_4h.tank_h),
               E_kWh=float(best_4h.E_4h_kWh), usd_per_m3=float(best_4h.usd_per_m3_4h_2023)),
           run_seconds=time.time() - t0)
(OUT / "battery_result.json").write_text(json.dumps(res, indent=1, default=float))
log("done")
