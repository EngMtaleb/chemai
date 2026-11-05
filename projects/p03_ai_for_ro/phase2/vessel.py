"""Phase 2, step 2 - one pressure vessel of 8-inch SW30-type elements: the pressure, recovery and energy a design needs.

Membrane      A = 1.097 LMH/bar, B = 0.0773 LMH at 20 C - fitted in phase 1 (SW30 chemistry, NaCl); 41 m2 per element
              (8-inch, 440 ft2 - FilmTec SW30HRLE-440 / SW30XLE-440 datasheets).
Temperature   FilmTec TCF = exp(K (1/298.15 - 1/(273.15 + T))), K = 2640 at or above 25 C, 3020 below (FilmTec design
              equations); A and B are both scaled by TCF(T)/TCF(20).
Osmotic       the NaCl relation of phase 1 (Sitterley et al. 2025, eq 3), bar.
Polarisation  FilmTec: pf = exp(0.7 Y_i), Y_i = that element's recovery.
Pressure drop FilmTec eq 27c: dP (psi) = 0.01 q^1.7 per element, q = the element's average feed-side flow (gpm).
Limits        FilmTec 8-inch design guidelines (45-D01695-en), seawater open intake SDI < 5: max element recovery 13 %,
              max element flux 32 LMH, max feed 14 m3/h per vessel, min concentrate 3.4 m3/h per vessel;
              datasheet: max 83 bar, max 1.0 bar drop per element.
Energy        isobaric energy recovery: the high-pressure pump lifts the permeate-equivalent flow to the feed pressure;
              the device returns eta_ERD of the brine pressure to an equal flow of seawater; a booster tops it up.
              eta_HP 0.90 (Lin & Veerapaneni 2021, ESI Table S1), eta_ERD 0.96 (Schunke et al. 2020: 95-97 %),
              booster 0.90 and motor 0.95 are assumptions to confirm. The low-pressure supply (intake, pretreatment) is
              outside this number.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
AREA_EL = 41.0
A20, B20 = 1.097, 0.0773
GPM_PER_M3H, PSI_PER_BAR = 4.402868, 14.5038
ETA_HP, ETA_ERD, ETA_BP, ETA_MOTOR = 0.90, 0.96, 0.90, 0.95
LIM = dict(el_recovery=0.13, el_flux=32.0, q_feed_max=14.0, q_brine_min=3.4, p_max=83.0, el_dp_max=1.0)


def tcf(T):
    return float(np.exp((2640.0 if T >= 25 else 3020.0) * (1 / 298.15 - 1 / (273.15 + T))))


def pi_bar(c):
    return 0.741829 * c + 0.00111004 * c ** 2


def solve_flux(P_loc, cm, A, B, iters=50):
    """J (LMH) from J = A (P - pi(c_m) + pi(c_p)), c_p = B c_m / (J + B). The root is unique (f rises with J), so
    bisect on [0, A P]: a fixed-point loop can settle on the wrong side when P is near the osmotic pressure."""
    lo, hi = np.zeros_like(P_loc), np.maximum(A * P_loc, 1e-9)
    for _ in range(iters):
        J = (lo + hi) / 2
        f = J - A * (P_loc - pi_bar(cm) + pi_bar(B * cm / (J + B)))
        hi, lo = np.where(f > 0, J, hi), np.where(f > 0, lo, J)
    J = (lo + hi) / 2
    return J, B * cm / (J + B)


def vessel(P_f, Q_f, c_f, T, n=7, slices=20, y_iters=12):
    """Feed pressure (bar, scalar or array), feed flow (m3/h), salinity (g/L), temperature (C) -> one vessel's output."""
    k = tcf(T) / tcf(20.0)
    A, B = A20 * k, B20 * k
    P = np.atleast_1d(np.asarray(P_f, float)).copy()
    Q = np.full_like(P, float(Q_f)); c = np.full_like(P, float(c_f))
    dA = AREA_EL / slices
    rec, flux, perm, cpm, dps = [], [], [], [], []
    Y = np.full_like(P, 0.08)
    for e in range(n):
        for _ in range(y_iters):                                 # the element's recovery sets its polarisation
            pf = np.exp(0.7 * Y)
            dP = 0.01 * (Q * (1 - Y / 2) * GPM_PER_M3H) ** 1.7 / PSI_PER_BAR
            Qi, ci = Q.copy(), c.copy()
            qp_el, salt_el = np.zeros_like(P), np.zeros_like(P)
            for s_ in range(slices):
                P_loc = P - dP * (s_ + 0.5) / slices
                J, cp = solve_flux(P_loc, ci * pf, A, B)
                qp = np.minimum(J * dA / 1000.0, 0.95 * Qi)
                ci = (Qi * ci - qp * cp) / (Qi - qp)
                Qi = Qi - qp
                qp_el += qp; salt_el += qp * cp
            Y = qp_el / Q
        rec.append(Y.copy()); flux.append(1000 * qp_el / AREA_EL); perm.append(qp_el.copy())
        cpm.append(1000 * salt_el / np.maximum(qp_el, 1e-12)); dps.append(dP.copy())
        P, Q, c = P - dP, Qi, ci
        cp_last = cp
    perm, cpm = np.array(perm), np.array(cpm)
    Qp = perm.sum(0)
    pf_last = np.exp(0.7 * rec[-1])
    out = dict(Q_p=Qp, recovery=Qp / Q_f, c_p_mgL=(perm * cpm).sum(0) / np.maximum(Qp, 1e-12),
               P_brine=P, Q_brine=Q, c_brine=c, ndp_tail=P - (pi_bar(c * pf_last) - pi_bar(cp_last)),
               lead_recovery=rec[0], lead_flux=flux[0], max_el_dp=np.max(dps, 0),
               el_recovery=np.array(rec), el_flux=np.array(flux), el_cp=cpm)
    return out


def sec_ro(P_f, Q_f, Q_p, P_b):
    """kWh per m3 of permeate: high-pressure pump on Q_p, booster on the brine-equivalent flow after the isobaric device."""
    Q_b = Q_f - Q_p
    hp = Q_p * P_f / ETA_HP
    bp = Q_b * max(P_f - ETA_ERD * P_b, 0) / ETA_BP
    return (hp + bp) / ETA_MOTOR / 36.0 / Q_p                  # bar x m3/h -> kW: / 36


def pick(v, i):
    return {k: (x[..., i] if isinstance(x, np.ndarray) else x) for k, x in v.items()}


def operate(flux, R, c_f, T, n=7):
    """The feed pressure giving an average flux (LMH) at recovery R; None if it would need more than 83 bar."""
    Q_p = flux * n * AREA_EL / 1000.0
    Q_f = Q_p / R
    lo, hi = 0.5 * pi_bar(c_f), LIM["p_max"]
    for _ in range(3):                                           # scan, then zoom in twice
        grid = np.linspace(lo, hi, 41)
        q = vessel(grid, Q_f, c_f, T, n)["Q_p"]
        if q[-1] < Q_p:
            return None
        j = int(np.argmax(q >= Q_p))
        lo, hi = grid[max(j - 1, 0)], grid[j]
    P_f = float(np.interp(Q_p, [vessel(lo, Q_f, c_f, T, n)["Q_p"][0], vessel(hi, Q_f, c_f, T, n)["Q_p"][0]], [lo, hi]))
    v = pick(vessel(P_f, Q_f, c_f, T, n), 0)
    v.update(P_f=P_f, Q_f=Q_f, flux=flux, n=n, T=T, R=R, c_f=c_f,
             sec_ro=sec_ro(P_f, Q_f, float(v["Q_p"]), float(v["P_brine"])))
    return v


def flags(v):
    f = []
    if v["lead_recovery"] > LIM["el_recovery"]: f.append("lead recovery")
    if v["lead_flux"] > LIM["el_flux"]: f.append("lead flux")
    if v["Q_f"] > LIM["q_feed_max"]: f.append("feed flow")
    if v["Q_brine"] < LIM["q_brine_min"]: f.append("brine flow")
    if v["max_el_dp"] > LIM["el_dp_max"]: f.append("element dP")
    if v["ndp_tail"] < 0: f.append("tail NDP < 0")
    return ", ".join(f) or "ok"


def main():
    C_F, FLUX = 43.0, 14.0
    rows = []
    for n, R, T in product((6, 7), (0.35, 0.40, 0.45, 0.50), (11.0, 25.0, 36.0)):
        v = operate(FLUX, R, C_F, T, n)
        if v is None:
            rows.append(dict(n=n, R=R, T=T, P_f=np.nan, note="needs > 83 bar"))
            continue
        rows.append(dict(n=n, R=R, T=T, P_f=v["P_f"], Q_f=v["Q_f"], Q_brine=v["Q_brine"], c_brine=v["c_brine"],
                         ndp_tail=v["ndp_tail"], lead_flux=v["lead_flux"], lead_rec=100 * v["lead_recovery"],
                         max_el_dp=v["max_el_dp"], c_p=v["c_p_mgL"], sec_ro=v["sec_ro"], note=flags(v)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    t.to_csv(HERE / "out" / "vessel_design_sweep.csv", index=False)
    # check against the datasheet: SW30HRLE-440 at 32 g/L NaCl, 55.2 bar, 25 C, 8 % recovery -> 30.2 m3/d, 99.80 % rejection
    lo, hi = 1.0, 30.0
    for _ in range(40):
        q = (lo + hi) / 2
        v = pick(vessel(55.16, q, 32.0, 25.0, n=1), 0)
        lo, hi = (lo, q) if v["recovery"] < 0.08 else (q, hi)
    v = pick(vessel(55.16, (lo + hi) / 2, 32.0, 25.0, n=1), 0)
    check = {"permeate_m3d": float(24 * v["Q_p"]), "rejection_pct": float(100 * (1 - v["c_p_mgL"] / 1000 / 32.0)),
             "datasheet_SW30HRLE_440": {"permeate_m3d": 30.2, "rejection_pct": 99.80},
             "datasheet_SW30XLE_440": {"permeate_m3d": 37.4, "rejection_pct": 99.8}}
    print("\nsingle-element check at the datasheet test:", json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in check.items()}))
    (HERE / "out" / "vessel_check.json").write_text(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
