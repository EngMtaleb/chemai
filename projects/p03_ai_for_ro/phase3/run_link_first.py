"""Phase 4 - the link, as declared in declaration.md: fouling's energy on the phase-2 train under the sun, and when energy
alone would clean.

    python run_link.py <sun csv> <sea csv> <deposit daily csv> <out folder>

Project 5's cycle inputs and cleaning economics are imported from the repository's history: projects/p05_when_to_stop
at commit 0b12b39 (see project5.py).
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import project5  # noqa: E402
project5.add_to_path()  # Project 5 is kept at commit 0b12b39: see project5.py
import p4lib as L  # noqa: E402
from step4_ro_cip_economics import cycle_inputs, SMOOTH_DAYS, MIN_DAYS, MID, GRID  # noqa: E402
from chemai.data.ro_deposit import load_daily  # noqa: E402
from chemai.features.ro_physics import fouling_energy_penalty, specific_energy_per_bar  # noqa: E402
from chemai.models.stopping import interval_economics  # noqa: E402

t0 = time.time()
SUN, SEA, DAILY, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
res = {}
say = lambda *a: print(*a, f"[{time.time() - t0:.0f} s]", flush=True)  # noqa: E731

# ------------------------------------------------------------------ the phase-2 train, and the check that it is phase 2
sun, sea = L.load_sun_sea(SUN, SEA)
H = L.pv_per_kwp(sun)
q, rej = L.fit_membrane()
res["membrane"] = {"A20": round(L.MEM["A20"], 4), "B20": round(L.MEM["B20"], 4), "test_m3d": round(q, 2), "rejection": round(rej, 2)}
ref_design = {15.0: (0.40, 66.54, 2.38), 25.0: (0.40, 63.97, 2.29), 33.0: (0.40, 63.44, 2.27), 35.0: (0.35, 59.87, 2.23)}
for T, (R, pf, sec) in ref_design.items():
    v = L.operate([14.0], R, 43.0, T)[0]
    assert round(v["P_f"], 2) == pf and round(v["sec_ro"], 2) == sec, (T, v["P_f"], v["sec_ro"])
perf0, hot0 = L.perf_table()
assert len(perf0) == 132 and hot0 == [35.0], (len(perf0), hot0)
OTHER = L.other_energy(perf0)
assert round(OTHER, 2) == 1.23
CAP, KWP, TANK_H = 2.0, 0.8, 12
cv1, cvC = L.curves(perf0, 1.0, OTHER), L.curves(perf0, CAP, OTHER)
ten = []
for year in range(2015, 2025):
    pv, T, _ = L.sun_sea(H, sea, year); pv_kw = pv * KWP * 1000
    ten.append({"strategy": "steady", **L.score(L.run_steady(pv_kw, T, cv1, 0), pv_kw, T, cv1)})
    ten.append({"strategy": "rule", **L.score(L.run_rule(pv_kw, T, cvC, TANK_H * L.DEMAND), pv_kw, T, cvC)})
ten = pd.DataFrame(ten).groupby("strategy")[["kWh/m3", "solar share %", "grid kWh/m3"]].mean().round(2)
assert ten.loc["steady"].tolist() == [3.54, 41.47, 2.07] and ten.loc["rule"].tolist() == [3.53, 79.59, 0.72], ten
res["check: d = f = 0 is phase 2"] = "pass: membrane fit, design points, 132 + 12 table points, rest of plant 1.23, 10-year steady and rule"
say("phase 2 reproduced")

# ------------------------------------------------------------------ 1. the penalty map
rows = []
for T in (15.0, 25.0, 35.0):
    R = 0.35 if T in hot0 else 0.40
    for d in (0.0, 0.1, 0.2, 0.3, 0.4):
        for f in (0.0, 0.05, 0.10, 0.20):
            v = L.operate([14.0], R, 43.0, T, d=d, f=f)[0]
            rows.append({"T": T, "R": R, "d": d, "f": f, "P_f": v["P_f"] if v else np.nan, "sec_ro": v["sec_ro"] if v else np.nan,
                         "vessel_dP": (v["P_f"] - float(v["P_brine"])) if v else np.nan,
                         "lead_el_dP": float(v["lead_el_dp"]) if v else np.nan, "max_el_dP": float(v["max_el_dp"]) if v else np.nan,
                         "limits": L.flags(v) if v else "needs > 83 bar"})
pm = pd.DataFrame(rows)
clean = pm[(pm.d == 0) & (pm.f == 0)].set_index("T").sec_ro
pm["extra_kWh_m3"] = pm.sec_ro - pm["T"].map(clean)
pm["extra_pct"] = 100 * pm.extra_kWh_m3 / pm["T"].map(clean)
pm.to_csv(OUT / "penalty_map.csv", index=False)
say("penalty map done")

# ------------------------------------------------------------------ 2. this plant's fouling on our train, 25 C
DG, FG = np.round(np.arange(0.0, 0.601, 0.05), 3), np.round(np.arange(0.0, 0.301, 0.025), 3)
fine = np.array([[L.operate([14.0], 0.40, 43.0, 25.0, d=d, f=f)[0]["sec_ro"] for f in FG] for d in DG])
np.save(OUT / "fine_map_25C.npy", fine)
dep = pd.read_csv(DAILY)
dep = dep[dep.day_index >= 300]


def bilinear(d, f):
    d, f = np.clip(d, DG[0], DG[-1]), np.clip(f, FG[0], FG[-1])
    i = np.clip(np.searchsorted(DG, d, side="right") - 1, 0, len(DG) - 2)
    j = np.clip(np.searchsorted(FG, f, side="right") - 1, 0, len(FG) - 2)
    u, w = (d - DG[i]) / (DG[i + 1] - DG[i]), (f - FG[j]) / (FG[j + 1] - FG[j])
    return (fine[i, j] * (1 - u) * (1 - w) + fine[i + 1, j] * u * (1 - w) + fine[i, j + 1] * (1 - u) * w + fine[i + 1, j + 1] * u * w)


extra = bilinear(dep.dp_pct.to_numpy(), dep.flux_pct.to_numpy()) - fine[0, 0]
res["2 this plant's fouling on our train, 25 C"] = {
    "train_days": int(len(dep)), "clean_RO_kWh_m3": round(float(fine[0, 0]), 4),
    "extra_kWh_m3_median": round(float(np.median(extra)), 4), "extra_kWh_m3_p90": round(float(np.quantile(extra, 0.9)), 4),
    "extra_pct_median": round(float(100 * np.median(extra) / fine[0, 0]), 2),
    "extra_pct_p90": round(float(100 * np.quantile(extra, 0.9) / fine[0, 0]), 2),
    "dp_beyond_map_share": round(float((dep.dp_pct > DG[-1]).mean()), 4), "f_beyond_map_share": round(float((dep.flux_pct > FG[-1]).mean()), 4)}
say("plant's fouling done")

# ------------------------------------------------------------------ 3. a year under the sun, fouled
dep_all = pd.read_csv(DAILY)
legs = {}
for t, g in dep_all.groupby("train"):
    legs[t] = sorted({int(round(v)) for v in pd.concat([(g.day_index - g.days_since_cip).dropna(), (g.day_index + g.days_to_next_cip).dropna()])})
pre_f = []
for t, ls in legs.items():                     # f on the 7 days before each campaign's first leg (legs within 7 days = one campaign)
    camps, last = [], None
    for x in ls:
        if last is None or x - last > 7:
            camps.append(x)
        last = x
    g = dep_all[dep_all.train == t].set_index("day_index")
    for a in camps:
        w = g[(g.index < a) & (g.index >= a - 7)]
        if len(w) >= 3:
            pre_f.append(float(w.flux_pct.median()))
STATES = {"clean": (0.0, 0.0),
          "median day": (float(dep.dp_pct.median()), float(dep.flux_pct.median())),
          "operators' cleaning level": (0.228, float(np.median(pre_f))),
          "both at 90th percentile": (float(dep.dp_pct.quantile(0.9)), float(dep.flux_pct.quantile(0.9)))}
pv, T24, _ = L.sun_sea(H, sea, 2024); pv_kw = pv * KWP * 1000
year = []
for name, (d, f) in STATES.items():
    perf, hot = (perf0, hot0) if name == "clean" else L.perf_table(d, f)
    c1, cC = L.curves(perf, 1.0, OTHER), L.curves(perf, CAP, OTHER)
    for strat, fn, cv in (("steady", lambda: L.run_steady(pv_kw, T24, c1, 0), c1),
                          ("rule", lambda: L.run_rule(pv_kw, T24, cC, TANK_H * L.DEMAND), cC)):
        s = L.score(fn(), pv_kw, T24, cv)
        year.append({"state": name, "d": d, "f": f, "35 % recovery at": hot, "strategy": strat, **s})
    say(f"year at {name} (d {d:.3f}, f {f:.3f}) done; 35 % recovery at {hot}")
year = pd.DataFrame(year)
base = year[year.state == "clean"].set_index("strategy")
year["extra kWh/m3"] = year["kWh/m3"] - year.strategy.map(base["kWh/m3"])
year["extra grid kWh/m3"] = year["grid kWh/m3"] - year.strategy.map(base["grid kWh/m3"])
year["grid share of the penalty"] = year["extra grid kWh/m3"] / year["extra kWh/m3"]
year.to_csv(OUT / "fouled_years_2024.csv", index=False)

# ------------------------------------------------------------------ 4. when would energy alone clean, with the sun?
A25 = L.MEM["A20"] * L.tcf(25.0) / L.tcf(20.0)
v0 = L.operate([14.0], 0.40, 43.0, 25.0)[0]
NDP0, DP0 = 14.0 / A25, v0["P_f"] - float(v0["P_brine"])
dlt = 0.5
KWH_BAR = (L.sec_ro(v0["P_f"] + dlt, v0["Q_f"], float(v0["Q_p"]), float(v0["P_brine"]) + dlt) - v0["sec_ro"]) / dlt
fouled = year[year.state != "clean"]
SHARE = {"no sun (Project 5)": 1.0,
         "steady, with panels": float(fouled[fouled.strategy == "steady"]["grid share of the penalty"].max()),
         "rule, with panels": float(fouled[fouled.strategy == "rule"]["grid share of the penalty"].max())}
daily = load_daily()
cycles = cycle_inputs(daily)


def count(price, F, kwh_bar, ndp0, dp0):
    n = 0
    for c in cycles:
        m = price * fouling_energy_penalty(c["ndp"], c["d_dp"], ndp0, dp0, kwh_bar)
        e = interval_economics(c["day"], m, F, smooth=SMOOTH_DAYS, min_t=MIN_DAYS)
        n += int(e.rule_time is not None)
    return n


p5_mid = count(MID["price"], MID["F"], specific_energy_per_bar(erd_eff=0.95), MID["ndp0"], MID["dp0"])
assert p5_mid == 0, p5_mid
res["check: Project 5's own constants"] = f"pass: mid case, {p5_mid} of {len(cycles)} cycles, as published"
grid = []
for price, F in itertools.product(GRID["price"], GRID["F"]):
    row = {"price": price, "F": F}
    for name, share in SHARE.items():
        row[name] = count(price * share, F, KWH_BAR, NDP0, DP0)
    grid.append(row)
grid = pd.DataFrame(grid)
grid.to_csv(OUT / "cleaning_by_energy_with_sun.csv", index=False)
say("cleaning economics done")

# ------------------------------------------------------------------ the expectations
med = pm[(pm["T"] == 25.0)]
p90d, p90f = STATES["both at 90th percentile"]
v90 = L.operate([14.0], 0.40, 43.0, 25.0, d=p90d, f=p90f)[0]
fy = year[year.state != "clean"].set_index(["state", "strategy"])["grid share of the penalty"].unstack()
res["constants from the model"] = {"NDP0_bar": round(NDP0, 2), "DP0_bar": round(DP0, 3), "kWh_per_bar": round(KWH_BAR, 4),
                                   "Project 5 mid case: NDP0, DP0, kWh/bar": [MID["ndp0"], MID["dp0"], round(specific_energy_per_bar(), 4)]}
res["states"] = {k: [round(a, 4), round(b, 4)] for k, (a, b) in STATES.items()}
res["grid share used in 4 (the largest over the three fouled states)"] = {k: round(v, 3) for k, v in SHARE.items()}
res["E1"] = {"extra_pct_median_day": res["2 this plant's fouling on our train, 25 C"]["extra_pct_median"],
             "pass": bool(res["2 this plant's fouling on our train, 25 C"]["extra_pct_median"] < 1.0)}
res["E2"] = {"grid share, by state": {s: {k: round(float(x), 3) for k, x in r.items()} for s, r in fy.iterrows()},
             "pass": bool((fy["rule"] < fy["steady"]).all())}
res["E3"] = {"max cycles, no sun": int(grid["no sun (Project 5)"].max()), "max cycles, rule": int(grid["rule, with panels"].max()),
             "pass": bool((grid["rule, with panels"] <= grid["no sun (Project 5)"]).all())}
res["E4"] = {"state": [round(p90d, 4), round(p90f, 4)], "max_element_dP_bar": round(float(v90["max_el_dp"]), 3),
             "lead_element_dP_bar": round(float(v90["lead_el_dp"]), 3), "limits": L.flags(v90),
             "pass": bool(float(v90["max_el_dp"]) <= 1.0)}
res["seconds"] = round(time.time() - t0)
(OUT / "link_results.json").write_text(json.dumps(res, indent=1, default=str))
pd.set_option("display.width", 220)
print(json.dumps(res, indent=1, default=str))
print(pm.round(3).to_string(index=False))
print(year[["state", "strategy", "d", "f", "kWh/m3", "grid kWh/m3", "solar share %", "extra kWh/m3", "extra grid kWh/m3",
            "grid share of the penalty", "35 % recovery at"]].round(3).to_string(index=False))
print(grid.to_string(index=False))
