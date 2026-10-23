"""RO physics: the forms that decide what a fouling signal looks like.

Two questions shape this project before any model is fitted.

1. **What does the pressure drop depend on besides fouling?** Flow and
   temperature. In a spacer-filled feed channel the friction factor falls with
   Reynolds number, f ~ Re^-m, so at fixed geometry

        dP ~ Q_avg^(2-m) * mu^m                      m = 0.19-0.30

   Schock & Miquel (1987) give m = 0.3; CFD on spacer channels gives ~0.2.
   The flow exponent is therefore 1.7-1.8 - not the 1.0-1.5 found in vendor
   normalisation sheets - and viscosity enters as mu^m, which is how the
   seasons reach the pressure drop. Normalise with the wrong form and a
   production change reads as fouling (VALIDATION.md section 6).

2. **How does the pressure drop grow as the channel closes?** At a fixed
   flow, the velocity scales as 1/gap and the hydraulic diameter as gap, so
   Re stays put and

        dP ~ gap^-n                                  n = 3 (slit), ~2.2 (spacer CFD)

   A foulant layer that grows at a steady rate makes dP ACCELERATE. Written as
   DP_norm^(-1/n), the same signal is linear in the blocked fraction - the
   way 1/T straightens Antoine's equation.

Permeate flow and salt passage are normalised the ASTM D4516 way: actual
values scaled to reference net driving pressure, temperature and
concentration. ASTM does not standardise the pressure drop; that part is
physics, above.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.config import ROConfig

R_GAS = 0.083145          # L bar / (mol K)
NACL_G_MOL = 58.44


def _celsius(temp_c) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    if np.any(t < -5) or np.any(t > 60):
        raise ValueError("temperature must be in degrees Celsius, between -5 and 60 "
                         "(a value near 300 is Kelvin)")
    return t


def water_viscosity(temp_c) -> np.ndarray:
    """Dynamic viscosity of water, Pa s (Vogel equation; 0.89 mPa s at 25 C).

    Seawater is ~8 % more viscous at the same temperature. Normalisation uses
    the RATIO mu(T_ref)/mu(T), so that constant factor cancels.
    """
    t = _celsius(temp_c) + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (t - 140.0))


def tcf(temp_c, e: float = 2640.0, t_ref_c: float = 25.0) -> np.ndarray:
    """Temperature correction factor: permeability at T relative to T_ref.

    TCF = exp(E (1/T_ref - 1/T)), E in kelvin. Above T_ref the membrane is more
    permeable (TCF > 1). The constant is a vendor convention - 2640 K for water
    is a common one; take the installed membrane's own value where known.
    """
    t = _celsius(temp_c) + 273.15
    return np.exp(e * (1.0 / (t_ref_c + 273.15) - 1.0 / t))


def osmotic_pressure(tds_g_l, temp_c, osmotic_coefficient: float = 0.9) -> np.ndarray:
    """Osmotic pressure, bar, of a NaCl-equivalent solution (van 't Hoff).

    pi = phi * 2 * (TDS / 58.44) * R * T. Seawater at 35 g/L and 25 C gives
    ~27 bar. Normalisation uses differences and ratios, so the approximation
    matters less than using the same one every day.
    """
    c = np.asarray(tds_g_l, dtype=float)
    if np.any(c < 0):
        raise ValueError("TDS cannot be negative")
    return osmotic_coefficient * 2.0 * (c / NACL_G_MOL) * R_GAS * (_celsius(temp_c) + 273.15)


def log_mean_concentration_factor(recovery) -> np.ndarray:
    """Feed-to-concentrate average concentration over feed concentration.

    For a concentration that rises exponentially along the vessel, the mean is
    -ln(1 - R) / R (1.277 at 40 % recovery). ASTM D4516's arithmetic feed-brine
    average gives a similar number; either is fine if used consistently.
    """
    r = np.asarray(recovery, dtype=float)
    if np.any((r <= 0) | (r >= 0.95)):
        raise ValueError("recovery must be a fraction between 0 and 0.95")
    return -np.log1p(-r) / r


def net_driving_pressure(p_feed, p_conc, p_perm, tds_feed, recovery, temp_c,
                         tds_perm=0.0, polarisation: float = 1.0) -> np.ndarray:
    """NDP, bar = mean feed-side pressure - permeate pressure - osmotic difference."""
    cf = log_mean_concentration_factor(recovery)
    pi_fc = osmotic_pressure(np.asarray(tds_feed, float) * cf, temp_c) * polarisation
    pi_p = osmotic_pressure(tds_perm, temp_c)
    return (np.asarray(p_feed, float) + np.asarray(p_conc, float)) / 2.0 \
        - np.asarray(p_perm, float) - (pi_fc - pi_p)


def normalised_permeate_flow(q_perm, ndp, temp_c, ndp_ref: float, cfg: ROConfig | None = None):
    """ASTM D4516: permeate flow at reference NDP and temperature."""
    cfg = cfg or ROConfig()
    ndp = np.asarray(ndp, dtype=float)
    if np.any(ndp <= 0):
        raise ValueError("net driving pressure must be positive")
    return np.asarray(q_perm, float) * (ndp_ref / ndp) / tcf(temp_c, cfg.tcf_e_water, cfg.t_ref_c)


def normalised_salt_passage(salt_passage, flux, flux_ref: float, cf, cf_ref: float, temp_c,
                            cfg: ROConfig | None = None):
    """ASTM D4516-style: salt passage at reference flux, concentration and temperature.

    Salt flux J_s = B * C_fc does not depend on pressure, so salt passage scales
    as B(T) * C_fc / (J_w * C_f): more water dilutes the same salt.
    """
    cfg = cfg or ROConfig()
    sp = np.asarray(salt_passage, float)
    return sp * (np.asarray(flux, float) / flux_ref) * (cf_ref / np.asarray(cf, float)) \
        / tcf(temp_c, cfg.tcf_e_salt, cfg.t_ref_c)


def normalised_dp(dp, q_avg, temp_c, q_avg_ref: float, flow_exponent: float | None = None,
                  visc_exponent: float | None = None, cfg: ROConfig | None = None):
    """Feed-channel pressure drop at reference flow and temperature.

    dP_ref = dP * (Q_ref / Q)^(2-m) * (mu_ref / mu)^m. The defaults are the
    physics (m = 0.25); pass `flow_exponent=1.5, visc_exponent=0` to reproduce a
    typical vendor sheet, or `flow_exponent=1.0` a linear one.
    """
    cfg = cfg or ROConfig()
    n = cfg.dp_flow_exponent if flow_exponent is None else flow_exponent
    m = cfg.dp_visc_exponent if visc_exponent is None else visc_exponent
    q = np.asarray(q_avg, dtype=float)
    if np.any(q <= 0) or q_avg_ref <= 0:
        raise ValueError("flows must be positive")
    mu_ratio = water_viscosity(cfg.t_ref_c) / water_viscosity(temp_c)
    return np.asarray(dp, float) * (q_avg_ref / q) ** n * mu_ratio ** m


# ------------------------------------------------------------ the gap law

def dp_from_blockage(phi, n: float | None = None, lead_share: float = 1.0,
                     rest_ratio: float = 1.0, cfg: ROConfig | None = None):
    """DP_norm for a blocked gap fraction phi of the lead element.

    With the whole vessel fouled evenly (lead_share = rest_ratio = 1) this is
    (1 - phi)^-n. Biofouling collects in the lead element: its share of the
    clean pressure drop is f, and the other elements block rest_ratio * phi.
    """
    cfg = cfg or ROConfig()
    n = cfg.gap_exponent if n is None else n
    phi = np.asarray(phi, dtype=float)
    if np.any(phi < 0) or np.any(phi >= 1):
        raise ValueError("blocked fraction must be in [0, 1)")
    f = lead_share
    return f * (1 - phi) ** -n + (1 - f) * (1 - rest_ratio * phi) ** -n


def blockage_fraction(dp_norm, n: float | None = None, cfg: ROConfig | None = None):
    """Inverse of the even-fouling gap law: phi = 1 - DP_norm^(-1/n)."""
    cfg = cfg or ROConfig()
    n = cfg.gap_exponent if n is None else n
    x = np.asarray(dp_norm, dtype=float)
    if np.any(x <= 0):
        raise ValueError("DP_norm must be positive")
    return 1.0 - x ** (-1.0 / n)


def blockage_transform(dp_norm, n: float | None = None, cfg: ROConfig | None = None):
    """DP_norm^(-1/n): straight in time when the foulant layer grows steadily."""
    return 1.0 - blockage_fraction(dp_norm, n, cfg)


# ------------------------------------------------------------ whole records

NORMALISE_COLUMNS = ("feed_flow", "conc_flow", "perm_flow", "feed_pressure", "conc_pressure",
                     "perm_pressure", "temperature_c", "feed_tds", "perm_tds")


def normalise(df: pd.DataFrame, reference: dict | None = None, baseline_rows: int = 30,
              flow_exponent: float | None = None, visc_exponent: float | None = None,
              cfg: ROConfig | None = None) -> pd.DataFrame:
    """Normalise a record of RO operation. One row per hour or per day.

    Columns (flows m3/h, pressures bar, TDS g/L): feed_flow, conc_flow, perm_flow,
    feed_pressure, conc_pressure, perm_pressure, temperature_c, feed_tds, perm_tds.
    A `dp` column, if present, is used instead of feed - concentrate pressure:
    a differential transmitter across the vessel is the better measurement.

    The reference (start-up) state is the median of the first `baseline_rows`
    rows unless given. Returns NDP, normalised permeate flow, normalised salt
    passage and DP_norm - the pressure drop at reference flow and temperature
    over its own baseline value, so 1.15 means 15 % above start-up.
    """
    cfg = cfg or ROConfig()
    missing = [c for c in NORMALISE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    d = df.reset_index(drop=True)
    q_f, q_c, q_p = (d[c].to_numpy(float) for c in ("feed_flow", "conc_flow", "perm_flow"))
    if np.any(q_p <= 0) or np.any(q_c <= 0):
        raise ValueError("flows must be positive")
    if np.any(np.abs(q_f - q_c - q_p) > 0.05 * q_f):
        raise ValueError("feed flow differs from permeate + concentrate by more than 5 %: "
                         "check the flow meters before normalising")
    t = d["temperature_c"].to_numpy(float)
    rec = q_p / q_f
    dp = d["dp"].to_numpy(float) if "dp" in d.columns else \
        d["feed_pressure"].to_numpy(float) - d["conc_pressure"].to_numpy(float)
    ndp = net_driving_pressure(d["feed_pressure"], d["conc_pressure"], d["perm_pressure"],
                               d["feed_tds"], rec, t, d["perm_tds"])
    cf = log_mean_concentration_factor(rec)
    flux = q_p  # proportional to flux for a fixed membrane area
    sp = d["perm_tds"].to_numpy(float) / d["feed_tds"].to_numpy(float)
    q_avg = (q_f + q_c) / 2.0

    k = min(baseline_rows, len(d))
    ref = reference or {
        "ndp": float(np.median(ndp[:k])), "flux": float(np.median(flux[:k])),
        "cf": float(np.median(cf[:k])), "q_avg": float(np.median(q_avg[:k])),
    }
    dp_ref_flow = normalised_dp(dp, q_avg, t, ref["q_avg"], flow_exponent, visc_exponent, cfg)
    base = ref.get("dp") or float(np.median(dp_ref_flow[:k]))
    npf = normalised_permeate_flow(q_p, ndp, t, ref["ndp"], cfg)
    nsp = normalised_salt_passage(sp, flux, ref["flux"], cf, ref["cf"], t, cfg)
    npf_base = ref.get("npf") or float(np.median(npf[:k]))
    nsp_base = ref.get("nsp") or float(np.median(nsp[:k]))
    return pd.DataFrame({
        "ndp": ndp, "dp": dp, "dp_ref_flow": dp_ref_flow, "dp_norm": dp_ref_flow / base,
        "npf": npf, "npf_ratio": npf / npf_base, "nsp": nsp, "nsp_ratio": nsp / nsp_base,
    }, index=df.index)


# ------------------------------------------------------------ the plant's own exponent

def fit_flow_exponent(df: pd.DataFrame, max_gap_days: int = 2, min_steps: int = 15,
                      min_flow_spread: float = 0.03) -> dict:
    """Measure the pressure-drop flow exponent from the plant's own day-to-day changes.

    Fouling moves slowly; production can move in a day (a train starts, demand
    steps). So between consecutive days the change in ln dP is almost all flow
    and temperature. Writing both exponents from one friction factor,
    dP ~ Q_avg^n * mu^(2-n), turns this into a one-parameter regression:

        d ln dP - 2 d ln mu = n * (d ln Q_avg - d ln mu)

    Needs columns day, feed_flow, conc_flow, temperature_c and either dp or
    feed_pressure and conc_pressure. Returns the exponent with its standard
    error, or ok=False with the reason when production is too steady to tell.
    """
    need = ["day", "feed_flow", "conc_flow", "temperature_c"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    d = df.sort_values("day").reset_index(drop=True)
    dp = d["dp"].to_numpy(float) if "dp" in d.columns else \
        d["feed_pressure"].to_numpy(float) - d["conc_pressure"].to_numpy(float)
    q = (d["feed_flow"].to_numpy(float) + d["conc_flow"].to_numpy(float)) / 2.0
    mu = water_viscosity(d["temperature_c"].to_numpy(float))
    day = d["day"].to_numpy(float)
    ok = (dp > 0) & (q > 0)
    pair = ok[:-1] & ok[1:] & (np.diff(day) <= max_gap_days)
    y = np.diff(np.log(np.where(ok, dp, 1.0)))[pair] - 2.0 * np.diff(np.log(mu))[pair]
    x = np.diff(np.log(np.where(ok, q, 1.0)))[pair] - np.diff(np.log(mu))[pair]
    out = dict(n=None, se=None, steps=int(pair.sum()), flow_spread=round(float(np.std(x)), 4)
               if len(x) else 0.0, ok=False, note="")
    if len(x) < min_steps or np.std(x) < min_flow_spread:
        out["note"] = (f"production too steady to measure the exponent ({len(x)} day pairs, "
                       f"spread {out['flow_spread']:.3f}); use the physics value 1.75")
        return out
    keep = np.ones(len(x), bool)
    for _ in range(3):                         # trim outliers: one bad day is not a flow step
        xs, ys = x[keep], y[keep]
        b0 = float(np.sum(xs * ys) / np.sum(xs * xs))
        r = y - b0 * x
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) <= 3.5 * max(mad, 1e-6)
    xs, ys = x[keep], y[keep]
    n_hat = float(np.sum(xs * ys) / np.sum(xs * xs))
    resid = ys - n_hat * xs
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(xs) - 1, 1) / np.sum(xs * xs)))
    out.update(n=round(n_hat, 3), se=round(se, 3), steps=int(keep.sum()), ok=True)
    if not 0.8 <= n_hat <= 2.2:
        out["ok"] = False
        out["note"] = (f"exponent {n_hat:.2f} is outside 0.8-2.2: a flow meter or the dP "
                       "measurement is suspect before the membranes are")
    return out
