"""Phase 2, step 3 - a year of solar-powered RO operation, hour by hour (Kuwait sun and sea, 2015-2024).

Plant     per 1,000 m3/day of demand, delivered evenly round the clock (41.7 m3/h). 8 identical trains;
          capacity C = the plant's output at design flux (14 LMH) divided by demand.
          Each train runs between f_min(T) and f_max(T) of its design output - the range inside every FilmTec limit
          with permeate <= 500 mg/L, from the vessel model (7 elements, 40 % recovery, high-rejection membrane, 43 g/L).
Energy    RO energy from the vessel model + the rest of the plant (intake, pretreatment, product pumping), taken as a
          fixed amount per m3: RO is ~65 % of a SWRO plant's energy (Schunke et al. 2020), applied at the design point.
          At any output the trains share the load equally (lowest energy); below 8 x f_min some trains stop.
Sun       PV output per kWp, hour by hour, from step 1. No batteries: sun not used in its hour is lost (curtailed).
Water     a product tank of S hours of demand; the city is never short (every strategy keeps the tank >= 0).

Strategies
  steady     produce the demand every hour; sun when it shines, grid otherwise
  rule       no forecast: by day use all the sun the tank can take; by night empty the tank first, then produce
  plan-fcst  each midnight, plan the next 48 h by linear programming with tomorrow = today's sun (persistence);
             run the first 24 h of the plan with the sun that actually comes
  plan-known the same plan, knowing the next 48 h of sun exactly (an upper bound, not achievable)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack, eye

HERE = Path(__file__).parent
DEMAND = 1000.0 / 24.0                  # m3/h
TRAINS = 8
C_P_MAX = 500.0                         # mg/L
RO_SHARE = 0.65                         # Schunke et al. 2020


# ------------------------------------------------------------------ inputs
def load_perf(paths):
    t = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    t["f"] = t.flux / 14.0
    t["ok"] = (t.note == "ok") & (t.c_p <= C_P_MAX)
    return t


def other_energy(perf):
    """kWh/m3 outside the RO section: the RO is 65 % of the plant at the design point (25 C, design flux)."""
    g = perf[(perf.f.round(4) == 1.0)]
    g = g.sort_values("T")
    sec25 = float(np.interp(25.0, g["T"], g["sec_ro"]))
    return sec25 * (1 - RO_SHARE) / RO_SHARE


def curves(perf, other, C):
    """For each table temperature: points (q, E) of the plant's lowest-energy curve, its max output, f_min, f_max."""
    Q_tr = C * DEMAND / TRAINS
    out = {}
    for T, g in perf.groupby("T"):
        ok = g[g.ok].sort_values("f")
        if ok.empty:                                      # no feasible turndown: design flux only
            ok = g[g.f.round(4) == 1.0]
        q = [0.0] + list(TRAINS * Q_tr * ok.f)
        E = [0.0] + list(TRAINS * Q_tr * ok.f * (ok.sec_ro + other))
        cp = [float(ok.c_p.iloc[0])] + list(ok.c_p)
        q, E = np.array(q), np.array(E)
        sl = np.diff(E) / np.diff(q)
        assert np.all(np.diff(sl) > -1e-9), f"energy curve not convex at {T} C"
        out[float(T)] = dict(q=q, E=E, cp=np.array(cp), f=np.r_[ok.f.iloc[0], ok.f.values], q_max=q[-1],
                             f_min=float(ok.f.iloc[0]), f_max=float(ok.f.iloc[-1]), Q_tr=Q_tr)
    return out


def load_sun_sea(year):
    pv = pd.read_csv(HERE / "out" / "pv_hourly_per_kwp.csv", index_col=0)
    pv.index = pd.to_datetime(pv.index, utc=True).tz_convert("Asia/Kuwait")
    pv = pv[pv.index.year == year]["pv"]
    f = HERE / "data" / "POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv"
    lines = f.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("-END HEADER-"))
    sst = pd.read_csv(f, skiprows=i + 1)
    sst.index = pd.to_datetime(dict(year=sst.YEAR, month=sst.MO, day=sst.DY))
    T_hour = pd.Series(pv.index.tz_localize(None).normalize(), index=pv.index).map(sst.TS)
    return pv.to_numpy(float), T_hour.to_numpy(float)


# ------------------------------------------------------------------ the plant in one hour
def nearest(cv, T):
    keys = np.array(sorted(cv))
    return cv[float(keys[np.argmin(np.abs(keys - T))])]


def energy(c, q):
    return np.interp(q, c["q"], c["E"])


def permeate(c, q):
    return np.interp(q, c["q"], c["cp"])


def trains_on(c, q):
    if q <= 1e-9:
        return 0
    if q >= TRAINS * c["f_min"] * c["Q_tr"] - 1e-9:
        return TRAINS
    return int(np.ceil(q / (c["f_min"] * c["Q_tr"]) - 1e-9))


# ------------------------------------------------------------------ strategies
def run_steady(pv, T, cv, S_max):
    return np.full(len(pv), DEMAND)


def run_rule(pv, T, cv, S_max):
    q, S = np.zeros(len(pv)), 0.0
    for t in range(len(pv)):
        c = nearest(cv, T[t])
        if pv[t] > 0:
            q_sun = np.interp(pv[t], c["E"], c["q"])           # the output the sun can power on its own
            q[t] = min(max(q_sun, DEMAND - S), c["q_max"], S_max - S + DEMAND)
        else:
            q[t] = max(0.0, DEMAND - S)                           # night: tank first
        S += q[t] - DEMAND
    return q


RAMP_COST = 0.02          # kWh-equivalent per m3/h of change in output: stops the plan from switching trains every hour


def plan_lp(pv_h, T_h, cv, S0, S_max, q_prev):
    """min grid energy (+ a small cost on changing output) over the horizon.
    Variables per hour: output q, energy e (above the convex curve), sun used u, grid g, tank S, output up/down."""
    H = len(pv_h)
    cs = [nearest(cv, T) for T in T_h]
    n = 7 * H
    iq, ie, iu, ig, iS, iup, idn = (np.arange(H) + k * H for k in range(7))
    cost = np.zeros(n); cost[ig] = 1.0; cost[iq] = 1e-6; cost[iup] = RAMP_COST; cost[idn] = RAMP_COST
    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    for t, c in enumerate(cs):
        sl = np.diff(c["E"]) / np.diff(c["q"])
        ic = c["E"][:-1] - sl * c["q"][:-1]
        for s_, i0 in zip(sl, ic):                                # s q - e <= -i0
            rows += [r, r]; cols += [iq[t], ie[t]]; vals += [s_, -1.0]; b_ub.append(-i0); r += 1
    rows.append(r); cols.append(iS[H - 1]); vals.append(-1.0); b_ub.append(-S0); r += 1   # end tank >= start
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, n))
    Z = coo_matrix((H, H)); I = eye(H)
    D_sh = eye(H) - eye(H, k=-1)
    m1 = hstack([Z, I, -I, -I, Z, Z, Z])                          # e - u - g = 0
    m2 = hstack([-I, Z, Z, Z, D_sh, Z, Z])                        # S_t - S_{t-1} - q_t = -D
    m3 = hstack([D_sh, Z, Z, Z, Z, -I, I])                        # q_t - q_{t-1} - up + dn = 0
    A_eq = vstack([m1, m2, m3])
    be = [0.0] * H + [-DEMAND + S0] + [-DEMAND] * (H - 1) + [q_prev] + [0.0] * (H - 1)
    bounds = ([(0, c["q_max"]) for c in cs] + [(0, None)] * H + [(0, p) for p in pv_h] + [(0, None)] * H
              + [(0, S_max)] * H + [(0, None)] * (2 * H))
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=be, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[iq]


def run_plan(pv, T, cv, S_max, known=True, horizon=48):
    n = len(pv)
    q, S = np.zeros(n), 0.0
    for a in range(0, n, 24):                                     # every day, and a last part-day if the year has one
        b = min(a + horizon, n)
        if known:
            pv_h = pv[a:b]
        else:
            last = pv[a - 24:a] if a >= 24 else pv[a:a + 24]       # persistence: tomorrow looks like today
            pv_h = np.tile(last, int(np.ceil((b - a) / len(last))))[: b - a]
        x = plan_lp(pv_h, T[a:b], cv, S, S_max, q[a - 1] if a > 0 else DEMAND)
        for t in range(min(24, n - a)):
            q[a + t] = x[t]
            S += x[t] - DEMAND
    return q


# ------------------------------------------------------------------ scoring
def score(q, pv, T, cv, kwp):
    E = np.array([energy(nearest(cv, T[t]), q[t]) for t in range(len(q))])
    cp = np.array([permeate(nearest(cv, T[t]), q[t]) for t in range(len(q))])
    sun = pv * kwp
    used = np.minimum(E, sun)
    S = np.cumsum(q - DEMAND)
    n_on = np.array([trains_on(nearest(cv, T[t]), q[t]) for t in range(len(q))])
    water = q.sum()
    return {"water_m3": float(water), "short_hours": int((S < -1e-6).sum()),
            "kWh_per_m3": float(E.sum() / water), "solar_share_pct": float(100 * used.sum() / E.sum()),
            "grid_kWh_per_m3": float((E - used).sum() / water),
            "curtailed_pct": float(100 * (sun.sum() - used.sum()) / sun.sum()) if sun.sum() > 0 else 0.0,
            "permeate_mgL_avg": float((cp * q).sum() / water), "permeate_mgL_max": float(cp[q > 0].max()),
            "train_starts_per_day": float(np.maximum(np.diff(n_on), 0).sum() / (len(q) / 24))}


def main():
    perf_files = sys.argv[1].split(",")
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    perf = load_perf([HERE / "out" / p for p in perf_files])
    other = other_energy(perf)
    pv, T = load_sun_sea(year)
    print(f"year {year}: {len(pv)} hours, sea {T.min():.1f}-{T.max():.1f} C, other plant energy {other:.2f} kWh/m3")
    rows = []
    for C in (1.0, 1.5, 2.0):
        cv = curves(perf, other, C)
        for kwp_per_m3d in (0.4, 0.8, 1.2, 1.6):
            kwp = kwp_per_m3d * 1000.0
            for S_h in (0, 6, 12, 24):
                S_max = S_h * DEMAND
                for name, fn in (("steady", lambda: run_steady(pv, T, cv, S_max)),
                                 ("rule", lambda: run_rule(pv * kwp, T, cv, S_max)),
                                 ("plan-fcst", lambda: run_plan(pv * kwp, T, cv, S_max, known=False)),
                                 ("plan-known", lambda: run_plan(pv * kwp, T, cv, S_max, known=True))):
                    if name == "steady" and (S_h > 0 or C > 1.0):
                        continue                                  # steady needs neither tank nor spare capacity
                    if C == 1.0 and name != "steady":
                        continue                                  # no spare capacity: nothing to shift
                    q = fn()
                    s = score(q, pv, T, cv, kwp)
                    s.update(strategy=name, capacity=C, kwp_per_m3d=kwp_per_m3d, tank_h=S_h)
                    rows.append(s)
                    print(f"C {C:.1f} | PV {kwp_per_m3d:.1f} kWp/(m3/d) | tank {S_h:2d} h | {name:10s} | "
                          f"{s['kWh_per_m3']:.2f} kWh/m3 | solar {s['solar_share_pct']:5.1f} % | grid {s['grid_kWh_per_m3']:.2f} | "
                          f"curtailed {s['curtailed_pct']:5.1f} % | permeate {s['permeate_mgL_avg']:.0f} mg/L | "
                          f"starts/day {s['train_starts_per_day']:.1f} | short h {s['short_hours']}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "out" / f"schedule_{year}.csv", index=False)
    (HERE / "out" / f"schedule_{year}_meta.json").write_text(json.dumps({"other_kWh_per_m3": other, "year": year}))


if __name__ == "__main__":
    main()
