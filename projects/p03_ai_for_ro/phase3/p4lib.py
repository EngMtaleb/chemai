"""Phase 4 - the phase-2 train and year, with fouling. The phase-2 notebook's code (solar_ro_phase2.ipynb, sections 2-5),
copied without change except two fouling multipliers on the train:

    every element's feed-channel pressure drop x (1 + d)        the membrane's water permeability x (1 - f)

With d = f = 0 it is phase 2 exactly (checked in run_link.py).
"""
import numpy as np
import pandas as pd
import pvlib

# ------------------------------------------------------------------ section 2: what one kWp of panels gives
LAT, LON, TILT, AZ, ALBEDO = 29.37, 47.98, 29.0, 180.0, 0.25
GAMMA, LOSSES, ETA_INV, DC_AC = -0.004, 0.14, 0.96, 1.1


def read_power(f):
    lines = f.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("-END HEADER-")) + 1
    return pd.read_csv(f, skiprows=head)


def load_sun_sea(sun_file, sea_file):
    sun = read_power(sun_file)
    sun.index = pd.to_datetime(dict(year=sun.YEAR, month=sun.MO, day=sun.DY, hour=sun.HR), utc=True)
    sun = sun.rename(columns={"ALLSKY_SFC_SW_DWN": "ghi", "ALLSKY_SFC_SW_DNI": "dni", "ALLSKY_SFC_SW_DIFF": "dhi",
                              "CLRSKY_SFC_SW_DWN": "ghi_clear", "T2M": "t_air", "WS10M": "wind"})
    sea = read_power(sea_file)
    sea.index = pd.to_datetime(dict(year=sea.YEAR, month=sea.MO, day=sea.DY))
    return sun, sea


def pv_per_kwp(d):
    sp = pvlib.solarposition.get_solarposition(d.index + pd.Timedelta(minutes=30), LAT, LON)
    sp.index = d.index
    poa = pvlib.irradiance.get_total_irradiance(TILT, AZ, sp.apparent_zenith, sp.azimuth, d.dni, d.ghi, d.dhi,
                                                dni_extra=pvlib.irradiance.get_extra_radiation(d.index),
                                                albedo=ALBEDO, model="haydavies")
    poa_g = poa["poa_global"].fillna(0).clip(lower=0)
    tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = pvlib.temperature.sapm_cell(poa_g, d.t_air, d.wind, **tp)
    pdc = pvlib.pvsystem.pvwatts_dc(poa_g, t_cell, pdc0=1.0, gamma_pdc=GAMMA) * (1 - LOSSES)
    pac = pvlib.inverter.pvwatts(pdc, pdc0=1.0 / DC_AC / ETA_INV, eta_inv_nom=ETA_INV).clip(lower=0).fillna(0)
    pdc_25 = pvlib.pvsystem.pvwatts_dc(poa_g, pd.Series(25.0, index=d.index), pdc0=1.0, gamma_pdc=GAMMA) * (1 - LOSSES)
    h = pd.DataFrame({"ghi": d.ghi / 1000, "ghi_clear": d.ghi_clear / 1000, "pv": pac, "pv_dc": pdc, "pv_dc_25C": pdc_25,
                      "t_air": d.t_air})
    h.index = h.index.tz_convert("Asia/Kuwait")
    return h


# ------------------------------------------------------------------ section 3: one pressure vessel
AREA_EL, GPM_PER_M3H, PSI_PER_BAR = 41.0, 4.402868, 14.5038
ETA_HP, ETA_ERD, ETA_BP, ETA_MOTOR = 0.90, 0.96, 0.90, 0.95
LIM = dict(el_recovery=0.13, el_flux=32.0, q_feed_max=14.0, q_brine_min=3.4, p_max=83.0, el_dp_max=1.0)
MEM = {"A20": 1.097, "B20": 0.0773}                    # phase 1 values; refitted to the datasheet by fit_membrane()


def tcf(T):
    return float(np.exp((2640.0 if T >= 25 else 3020.0) * (1 / 298.15 - 1 / (273.15 + T))))


def pi_bar(c, T):
    return (0.741829 * c + 0.00111004 * c ** 2) * (273.15 + T) / 298.15


def solve_flux(P_loc, cm, A, B, T, iters=50):
    lo, hi = np.zeros_like(P_loc), np.maximum(A * P_loc, 1e-9)
    for _ in range(iters):
        J = (lo + hi) / 2
        f = J - A * (P_loc - pi_bar(cm, T) + pi_bar(B * cm / (J + B), T))
        hi, lo = np.where(f > 0, J, hi), np.where(f > 0, lo, J)
    J = (lo + hi) / 2
    return J, B * cm / (J + B)


def vessel(P, Qf, c_f, T, n=7, slices=20, y_iters=12, d=0.0, f=0.0):
    """Phase 2's vessel; d and f: the fouling multipliers (1 + d) on every element's pressure drop, (1 - f) on A."""
    k = tcf(T) / tcf(20.0)
    A, B = MEM["A20"] * k * (1.0 - f), MEM["B20"] * k
    P = np.asarray(P, float).copy(); Q = np.asarray(Qf, float) * np.ones_like(P); Q_f = Q.copy()
    c = np.full_like(P, float(c_f)); dA = AREA_EL / slices
    rec, flux, perm, cpm, dps = [], [], [], [], []
    Y = np.full_like(P, 0.08)
    for e in range(n):
        for _ in range(y_iters):
            pf = np.exp(0.7 * Y)
            dP = (1.0 + d) * 0.01 * (Q * (1 - Y / 2) * GPM_PER_M3H) ** 1.7 / PSI_PER_BAR
            Qi, ci = Q.copy(), c.copy()
            qp_el, salt_el = np.zeros_like(P), np.zeros_like(P)
            for s_ in range(slices):
                P_loc = P - dP * (s_ + 0.5) / slices
                J, cp = solve_flux(P_loc, ci * pf, A, B, T)
                qp = np.minimum(J * dA / 1000.0, 0.95 * Qi)
                ci = (Qi * ci - qp * cp) / (Qi - qp); Qi = Qi - qp
                qp_el += qp; salt_el += qp * cp
            Y = qp_el / Q
        rec.append(Y.copy()); flux.append(1000 * qp_el / AREA_EL); perm.append(qp_el.copy())
        cpm.append(1000 * salt_el / np.maximum(qp_el, 1e-12)); dps.append(dP.copy())
        P, Q, c = P - dP, Qi, ci
        cp_last = cp
    perm, cpm = np.array(perm), np.array(cpm)
    Qp = perm.sum(0); pf_last = np.exp(0.7 * rec[-1])
    return dict(Q_p=Qp, recovery=Qp / Q_f, c_p_mgL=(perm * cpm).sum(0) / np.maximum(Qp, 1e-12), P_brine=P, Q_brine=Q,
                c_brine=c, ndp_tail=P - (pi_bar(c * pf_last, T) - pi_bar(cp_last, T)), lead_recovery=rec[0], lead_flux=flux[0],
                max_el_dp=np.max(dps, 0), lead_el_dp=dps[0])


def sec_ro(P_f, Q_f, Q_p, P_b):
    hp = Q_p * P_f / ETA_HP
    bp = (Q_f - Q_p) * max(P_f - ETA_ERD * P_b, 0) / ETA_BP
    return (hp + bp) / ETA_MOTOR / 36.0 / Q_p


def operate(fluxes, R, c_f, T, n=7, d=0.0, f=0.0):
    fluxes = np.asarray(fluxes, float); m = len(fluxes)
    Q_p = fluxes * n * AREA_EL / 1000.0; Q_f = Q_p / R
    lo, hi, alive = np.full(m, 0.5 * pi_bar(c_f, T)), np.full(m, LIM["p_max"]), np.ones(m, bool)
    for _ in range(3):
        grid = np.array([np.linspace(lo[i], hi[i], 41) for i in range(m)])
        q = vessel(grid, Q_f[:, None] * np.ones((1, 41)), c_f, T, n, d=d, f=f)["Q_p"]
        for i in range(m):
            if alive[i] and q[i, -1] < Q_p[i]:
                alive[i] = False
            elif alive[i]:
                j = int(np.argmax(q[i] >= Q_p[i])); lo[i], hi[i] = grid[i, max(j - 1, 0)], grid[i, j]
    q_lo, q_hi = vessel(lo, Q_f, c_f, T, n, d=d, f=f)["Q_p"], vessel(hi, Q_f, c_f, T, n, d=d, f=f)["Q_p"]
    P_f = np.array([float(np.interp(Q_p[i], [q_lo[i], q_hi[i]], [lo[i], hi[i]])) for i in range(m)])
    v = vessel(P_f, Q_f, c_f, T, n, d=d, f=f)
    out = []
    for i in range(m):
        if not alive[i]:
            out.append(None); continue
        dd = {k: x[i] for k, x in v.items()}
        dd.update(P_f=float(P_f[i]), Q_f=float(Q_f[i]), flux=float(fluxes[i]), T=T, R=R,
                  sec_ro=sec_ro(float(P_f[i]), float(Q_f[i]), float(dd["Q_p"]), float(dd["P_brine"])))
        out.append(dd)
    return out


def flags(v):
    f = []
    if v["lead_recovery"] > LIM["el_recovery"]: f.append("lead recovery")
    if v["lead_flux"] > LIM["el_flux"]: f.append("lead flux")
    if v["Q_f"] > LIM["q_feed_max"]: f.append("feed flow")
    if v["Q_brine"] < LIM["q_brine_min"]: f.append("brine flow")
    if v["max_el_dp"] > LIM["el_dp_max"]: f.append("element dP")
    if v["ndp_tail"] < 0: f.append("tail NDP < 0")
    return ", ".join(f) or "ok"


def element_at_test():
    lo, hi = 1.0, 40.0
    for _ in range(40):
        q = (lo + hi) / 2
        v = vessel(np.array([55.16]), np.array([q]), 32.0, 25.0, n=1)
        lo, hi = (lo, q) if v["recovery"][0] < 0.08 else (q, hi)
    v = vessel(np.array([55.16]), np.array([(lo + hi) / 2]), 32.0, 25.0, n=1)
    return 24 * float(v["Q_p"][0]), 100 * (1 - float(v["c_p_mgL"][0]) / 1000 / 32.0)


def fit_membrane():
    for _ in range(12):
        q, rej = element_at_test()
        MEM["A20"] *= 30.2 / q
        MEM["B20"] *= (100 - 99.80) / (100 - rej)
    return element_at_test()


# ------------------------------------------------------------------ section 4: the operating window
FLUXES = list(range(6, 18))
TEMPS = [float(T) for T in np.arange(15.0, 35.1, 2.0)]


def table(R, temps, d=0.0, f=0.0):
    rows = []
    for T in temps:
        for flux, v in zip(FLUXES, operate(FLUXES, R, 43.0, float(T), d=d, f=f)):
            if v is None:
                rows.append(dict(T=T, R=R, flux=flux, note="needs > 83 bar")); continue
            rows.append(dict(T=T, R=R, flux=flux, P_f=v["P_f"], sec_ro=v["sec_ro"], c_p=float(v["c_p_mgL"]),
                             lead_flux=float(v["lead_flux"]), lead_rec=100 * float(v["lead_recovery"]),
                             Q_brine=float(v["Q_brine"]), note=flags(v)))
    t = pd.DataFrame(rows)
    t["ok"] = (t.note == "ok") & (t.c_p <= 500)
    return t


def perf_table(d=0.0, f=0.0, hot=None):
    """Phase 2's table: 40 % recovery, 35 % at the temperatures where the design point is not usable at 40 %.
    `hot`: use the clean train's hot temperatures (the plant keeps its recovery plan when it fouls)."""
    p40 = table(0.40, TEMPS, d, f)
    if hot is None:
        design_ok = p40[p40.flux == 14].set_index("T").ok
        hot = [T for T in TEMPS if not design_ok[T]]
    if hot:
        perf = pd.concat([p40[~p40["T"].isin(hot)], table(0.35, hot, d, f)], ignore_index=True)
    else:                                         # every temperature usable at 40 %: no 35 % table
        perf = p40.copy()
    perf["f"] = perf.flux / 14.0
    return perf, hot


# ------------------------------------------------------------------ section 5: a year, hour by hour
DEMAND, TRAINS, C_P_MAX, RO_SHARE = 1000.0 / 24.0, 8, 500.0, 0.65


def other_energy(perf_clean):
    g = perf_clean[(perf_clean.f.round(4) == 1.0) & (perf_clean.R == 0.40)].sort_values("T")
    return float(np.interp(25.0, g["T"], g["sec_ro"])) * (1 - RO_SHARE) / RO_SHARE


def curves(perf, C, other):
    Q_tr = C * DEMAND / TRAINS
    out = {}
    for T, gT in perf.groupby("T"):
        R = float(gT.R.iloc[0]); oth = other * 0.40 / R
        ok = gT[gT.ok].sort_values("f")
        assert not ok.empty, f"no usable point at {T} C"
        q = np.array([0.0] + list(TRAINS * Q_tr * ok.f))
        E = np.array([0.0] + list(TRAINS * Q_tr * ok.f * (ok.sec_ro + oth)))
        cp = np.array([float(ok.c_p.iloc[0])] + list(ok.c_p))
        assert np.all(np.diff(np.diff(E) / np.diff(q)) > -1e-9), "energy curve not convex"
        out[float(T)] = dict(q=q, E=E, cp=cp, q_max=q[-1], f_min=float(ok.f.iloc[0]), Q_tr=Q_tr, R=R)
    return out


def sun_sea(H, sea, year):
    pv = H.pv[H.index.year == year]
    T = pd.Series(pv.index.tz_localize(None).normalize(), index=pv.index).map(sea.TS)
    return pv.to_numpy(float), T.to_numpy(float), pv.index


def groups(cv, T):
    keys = np.array(sorted(cv))
    return keys, np.abs(T[:, None] - keys[None, :]).argmin(1)


def run_steady(pv_kw, T, cv, S_max):
    return np.full(len(pv_kw), DEMAND)


def run_rule(pv_kw, T, cv, S_max):
    keys, k = groups(cv, T)
    n = len(pv_kw); q_sun, q_max = np.zeros(n), np.zeros(n)
    for j, key in enumerate(keys):
        m = k == j; c = cv[float(key)]
        q_sun[m] = np.interp(pv_kw[m], c["E"], c["q"]); q_max[m] = c["q_max"]
    q, S = np.zeros(n), 0.0
    for t in range(n):
        qt = min(max(q_sun[t], DEMAND - S), q_max[t], S_max - S + DEMAND) if pv_kw[t] > 0 else max(0.0, DEMAND - S)
        q[t] = qt; S += qt - DEMAND
    return q


def hourly(q, T, cv):
    keys, k = groups(cv, T)
    E, cp, n_on = np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q), int)
    for j, key in enumerate(keys):
        m = k == j; c = cv[float(key)]
        E[m] = np.interp(q[m], c["q"], c["E"]); cp[m] = np.interp(q[m], c["q"], c["cp"])
        unit = c["f_min"] * c["Q_tr"]
        n_on[m] = np.where(q[m] <= 1e-9, 0, np.where(q[m] >= TRAINS * unit - 1e-9, TRAINS, np.ceil(q[m] / unit - 1e-9))).astype(int)
    return E, cp, n_on


def score(q, pv_kw, T, cv):
    E, cp, n_on = hourly(q, T, cv)
    used = np.minimum(E, pv_kw); S = np.cumsum(q - DEMAND); water = q.sum()
    return {"kWh/m3": E.sum() / water, "solar share %": 100 * used.sum() / E.sum(), "grid kWh/m3": (E - used).sum() / water,
            "sun thrown away %": 100 * (pv_kw.sum() - used.sum()) / pv_kw.sum() if pv_kw.sum() > 0 else 0.0,
            "permeate mg/L": (cp * q).sum() / water, "permeate max mg/L": cp[q > 0].max(),
            "train starts/day": np.maximum(np.diff(n_on), 0).sum() / (len(q) / 24), "hours short": int((S < -1e-6).sum()),
            "water m3": water, "energy kWh": E.sum(), "grid kWh": (E - used).sum()}
