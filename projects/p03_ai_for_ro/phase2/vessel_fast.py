"""The vessel model of vessel.py, batched: many production levels at one temperature in one call.
Same equations and the same arithmetic, element by element, so the results should match vessel.operate exactly.
Used by the notebook to build the performance table in minutes instead of a quarter of an hour."""
import numpy as np

import vessel as V


def solve_flux(P_loc, cm, A, B, iters=50):
    lo, hi = np.zeros_like(P_loc), np.maximum(A * P_loc, 1e-9)
    for _ in range(iters):
        J = (lo + hi) / 2
        f = J - A * (P_loc - V.pi_bar(cm) + V.pi_bar(B * cm / (J + B)))
        hi, lo = np.where(f > 0, J, hi), np.where(f > 0, lo, J)
    J = (lo + hi) / 2
    return J, B * cm / (J + B)


def vessel(P, Qf, c_f, T, n=7, slices=20, y_iters=12):
    """P and Qf: arrays of the same shape (bar, m3/h)."""
    k = V.tcf(T) / V.tcf(20.0)
    A, B = V.A20 * k, V.B20 * k
    P = np.asarray(P, float).copy()
    Q = np.asarray(Qf, float) * np.ones_like(P)
    Q_f = Q.copy()
    c = np.full_like(P, float(c_f))
    dA = V.AREA_EL / slices
    rec, flux, perm, cpm, dps = [], [], [], [], []
    Y = np.full_like(P, 0.08)
    for e in range(n):
        for _ in range(y_iters):
            pf = np.exp(0.7 * Y)
            dP = 0.01 * (Q * (1 - Y / 2) * V.GPM_PER_M3H) ** 1.7 / V.PSI_PER_BAR
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
        rec.append(Y.copy()); flux.append(1000 * qp_el / V.AREA_EL); perm.append(qp_el.copy())
        cpm.append(1000 * salt_el / np.maximum(qp_el, 1e-12)); dps.append(dP.copy())
        P, Q, c = P - dP, Qi, ci
        cp_last = cp
    perm, cpm = np.array(perm), np.array(cpm)
    Qp = perm.sum(0)
    pf_last = np.exp(0.7 * rec[-1])
    return dict(Q_p=Qp, recovery=Qp / Q_f, c_p_mgL=(perm * cpm).sum(0) / np.maximum(Qp, 1e-12),
                P_brine=P, Q_brine=Q, c_brine=c, ndp_tail=P - (V.pi_bar(c * pf_last) - V.pi_bar(cp_last)),
                lead_recovery=rec[0], lead_flux=flux[0], max_el_dp=np.max(dps, 0))


def operate_many(fluxes, R, c_f, T, n=7):
    """vessel.operate for several average fluxes at once; returns one dict per flux (None if > 83 bar)."""
    fluxes = np.asarray(fluxes, float)
    Q_p = fluxes * n * V.AREA_EL / 1000.0
    Q_f = Q_p / R
    m = len(fluxes)
    lo = np.full(m, 0.5 * V.pi_bar(c_f)); hi = np.full(m, V.LIM["p_max"])
    alive = np.ones(m, bool)
    for _ in range(3):
        grid = np.array([np.linspace(lo[i], hi[i], 41) for i in range(m)])
        q = vessel(grid, Q_f[:, None] * np.ones((1, 41)), c_f, T, n)["Q_p"]
        for i in range(m):
            if not alive[i]:
                continue
            if q[i, -1] < Q_p[i]:
                alive[i] = False
                continue
            j = int(np.argmax(q[i] >= Q_p[i]))
            lo[i], hi[i] = grid[i, max(j - 1, 0)], grid[i, j]
    q_lo = vessel(lo, Q_f, c_f, T, n)["Q_p"]
    q_hi = vessel(hi, Q_f, c_f, T, n)["Q_p"]
    P_f = np.array([float(np.interp(Q_p[i], [q_lo[i], q_hi[i]], [lo[i], hi[i]])) for i in range(m)])
    v = vessel(P_f, Q_f, c_f, T, n)
    out = []
    for i in range(m):
        if not alive[i]:
            out.append(None)
            continue
        d = {k: x[i] for k, x in v.items()}
        d.update(P_f=float(P_f[i]), Q_f=float(Q_f[i]), flux=float(fluxes[i]), n=n, T=T, R=R, c_f=c_f,
                 sec_ro=V.sec_ro(float(P_f[i]), float(Q_f[i]), float(d["Q_p"]), float(d["P_brine"])))
        out.append(d)
    return out
