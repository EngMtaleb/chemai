"""Part C: from the first 20 minutes of a closed-circuit RO batch, forecast its live
cost for the rest of the batch - and from that forecast, where the batch should end.

Project 5's square-root law draws a straight line through the live cost between
minutes 5 and 20 (m = m0 + s V) and solves V* = sqrt(2 F_eff / s). On the test
batches it was 11 % short at b = 10: the live cost bends, and the further out the
best end, the more an early slope misleads. Every method here forecasts the curve
m(V) beyond minute 20; the same economics then picks the end:

    J(V) = (F_b + E20 + ∫_{V20}^{V} m dV) / V,     V* = argmin J

so the methods differ only in the curve they believe. Volumes are expressed as
r = V / V20, live costs as m / m20 (both from minute 20).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.models.stopping import rolling_median

EARLY = (5.0, 20.0)
MIN_RUN = 25.0                                       # minutes, as the declared T4
GRID_R = np.round(np.arange(1.0, 10.0001, 0.02), 4)  # where a forecast curve is evaluated
POINT_STEP = 0.02                                    # training points every 2 % of V20

PLAIN = ["m20", "slope_rel", "curv_rel", "resid_rel", "V20_l", "E20_kwh", "Vfs_l", "c5", "c20",
         "flux", "q_recirc", "temp_c", "prior_drain_kwh", "prior_brine_l"]
PHYSICS = ["dc_rel", "c_ratio", "line_ratio_at_r", "conc_at_r"]


def early_and_points(curve: pd.DataFrame, row: pd.Series, cfg: CCROConfig | None = None):
    """(features, future points) of one batch, or None when it is too short to forecast.

    Features use samples up to minute 20 only. Future points: the smoothed live cost
    after minute 20, every POINT_STEP of r, to the end of the batch."""
    cfg = cfg or CCROConfig()
    ph = curve["phase"].to_numpy()
    filt = ph == "filt"
    el = curve["elapsed_min"].to_numpy()
    live = curve["live_kwh_m3"].to_numpy()
    V, E, c, t = (curve[k].to_numpy() for k in ("V_m3", "E_kwh", "c_ms_cm", "t_s"))
    w = filt & (el >= EARLY[0]) & (el <= EARLY[1]) & np.isfinite(live)
    if w.sum() < 50 or not filt.any() or el[filt][-1] < MIN_RUN:
        return None
    s, m0 = np.polyfit(V[w], live[w], 1)
    q2 = np.polyfit(V[w], live[w], 2)[0]
    ms = rolling_median(t, live, cfg.live_window_s)
    i20 = int(np.flatnonzero(filt & (el <= EARLY[1]))[-1])
    ifs = int(np.flatnonzero(filt)[0])
    V20, E20, m20 = float(V[i20]), float(E[i20]), float(ms[i20])
    i5 = int(np.flatnonzero(filt & (el >= EARLY[0]))[0])
    dcdv = np.polyfit(V[w], c[w], 1)[0] if np.isfinite(c[w]).all() else np.nan
    f = dict(m20=m20, slope_rel=s * V20 / m20, curv_rel=q2 * V20 ** 2 / m20,
             resid_rel=float(np.std(live[w] - (m0 + s * V[w]))) / m20, V20_l=V20 * 1000, E20_kwh=E20,
             Vfs_l=float(V[ifs]) * 1000, c5=float(c[i5]), c20=float(c[i20]),
             flux=float(row["flux"]), q_recirc=float(row["q_recirc"]), temp_c=float(row["temp_c"]),
             prior_drain_kwh=float(row["prior_drain_kwh"]), prior_brine_l=float(row["prior_brine_l"]),
             dc_rel=dcdv * V20 / float(c[i20]) if np.isfinite(dcdv) else np.nan, m0=float(m0), s=float(s),
             c_ratio=float(c[i20] / c[i5]) if c[i5] > 0 else np.nan, end_r=float(V[filt][-1] / V20))
    after = filt & (el > EARLY[1]) & np.isfinite(ms)
    r_after, m_after = V[after] / V20, ms[after]
    pts = pd.DataFrame()
    if len(r_after) > 5:
        grid = np.arange(1.0 + POINT_STEP, r_after.max() + 1e-9, POINT_STEP)
        order = np.argsort(r_after)
        mg = np.interp(grid, r_after[order], m_after[order])
        pts = pd.DataFrame({"r": grid, "ratio": mg / m20,
                            "resid": (mg - (m0 + s * grid * V20)) / m20})
    return f, pts


def physics_at(feat: pd.DataFrame, r: np.ndarray) -> pd.DataFrame:
    """The physics features that depend on r: the straight line's ratio, and the loop
    conductivity a constant salt build-up would reach (c grows with permeate removed)."""
    out = pd.DataFrame(index=feat.index)
    out["line_ratio_at_r"] = (feat["m0"] + feat["s"] * r * feat["V20_l"] / 1000) / feat["m20"]
    out["conc_at_r"] = 1 + feat["dc_rel"] * (r - 1)
    return out


def best_stop(ratio_curve: np.ndarray, feat: dict, b: float) -> float:
    """V* (litres) from a forecast m/m20 on GRID_R, with the batch's prior drain as the fixed cost."""
    V20, E20, m20 = feat["V20_l"] / 1000, feat["E20_kwh"], feat["m20"]
    Vg = GRID_R * V20
    m = np.asarray(ratio_curve, float) * m20
    E = E20 + np.r_[0.0, np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(Vg))]
    F = feat["prior_drain_kwh"] + b * feat["prior_brine_l"] / 1000.0
    J = (F + E) / Vg
    return float(Vg[int(np.argmin(J))] * 1000)
