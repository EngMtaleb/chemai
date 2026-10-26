"""CCRO batches: phases, energy, permeate, brine, and the cost curve of each batch.

A batch is start-up + stabilisation + filtration of one file. Its brine is
drained at the start of the next file of the same experiment, provided that
file follows within `max_gap_s` and is numbered next. Energy is the sum of
the two pumps' power draw; permeate and brine are integrated flows.

    specific energy  SEC = E / V_permeate                     kWh/m3
    brine ratio      SBC = V_brine / V_permeate               m3 brine per m3 permeate
    cost             J   = SEC + b * SBC                      b = brine price in kWh per m3 of brine

J is the objective the testbed's own optimisers minimised (with b = 1 or 10).
Here start-up and stabilisation are counted too (the testbed's objective left
them out; VALIDATION.md 2 shows both).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.models.stopping import average_cost, cumulative, marginal_equals_average, rolling_median, sqrt_law

COLUMNS = [
    "dataset", "exp", "file", "seq", "row", "ts", "runtime_hr", "state",
    "reject_conductivity_ms_per_cm", "reject_flow_l_per_min", "feed_flow_l_per_min",
    "permeate_flow_l_per_min", "feed_pressure_psi", "feed_pump_power_draw_kw",
    "recirculation_pump_power_draw_kw", "feed_temperature_c", "permeate_conductivity_us_per_cm",
    "flux_lmh", "conductivity_critical", "set_low_rej_cond_ms_per_cm",
    "feed_pump_set_flux_lmh", "recirculation_pump_set_flow_l_per_min",
]
PHASES = ("drain", "startup", "stab", "filt")


def phase_of(state) -> str:
    s = "" if state is None else str(state)
    if s.startswith("Discharge"):
        return "drain"
    if s.startswith(("Startup", "Initialize")):
        return "startup"
    if s.startswith("Stabil"):
        return "stab"
    if s.startswith("Experiment"):
        return "filt"
    return "other"


def _times(g: pd.DataFrame) -> np.ndarray:
    t = g["runtime_hr"].to_numpy(float) * 3600.0
    return np.maximum.accumulate(np.nan_to_num(t, nan=np.nanmin(t)))


def _power(g: pd.DataFrame) -> np.ndarray:
    return (g["feed_pump_power_draw_kw"].to_numpy(float)
            + g["recirculation_pump_power_draw_kw"].to_numpy(float))


def phase_totals(g: pd.DataFrame) -> dict:
    """Minutes, energy (kWh), permeate, reject-line and feed volumes (L) of one phase."""
    if len(g) < 2:
        return {"min": 0.0, "E_kwh": 0.0, "Vp_l": 0.0, "Vrej_l": 0.0, "Vfeed_l": 0.0}
    t = _times(g)
    return {
        "min": (t[-1] - t[0]) / 60.0,
        "E_kwh": cumulative(t, _power(g))[-1] / 3600.0,
        "Vp_l": cumulative(t, g["permeate_flow_l_per_min"].to_numpy(float))[-1] / 60.0,
        "Vrej_l": cumulative(t, g["reject_flow_l_per_min"].to_numpy(float))[-1] / 60.0,
        "Vfeed_l": cumulative(t, g["feed_flow_l_per_min"].to_numpy(float))[-1] / 60.0,
    }


def file_summary(g: pd.DataFrame) -> dict:
    """One file: the drain it opens with, then its own batch."""
    g = g.sort_values("row")
    ph = g["state"].map(phase_of).to_numpy()
    rec = {"dataset": g["dataset"].iloc[0], "exp": g["exp"].iloc[0], "file": g["file"].iloc[0],
           "seq": int(g["seq"].iloc[0]), "t_first": g["ts"].iloc[0], "t_last": g["ts"].iloc[-1]}
    for p in PHASES:
        for k, v in phase_totals(g[ph == p]).items():
            rec[f"{p}_{k}"] = v
    f = g[ph == "filt"]
    d = g[ph == "drain"]
    c = f["reject_conductivity_ms_per_cm"].to_numpy(float)
    rec.update({
        "c_start": float(np.nanmedian(c[:20])) if len(c) else np.nan,
        "c_end": float(np.nanmedian(c[-10:])) if len(c) else np.nan,
        "c_crit": float(np.nanmedian(f["conductivity_critical"])) if len(f) and f["conductivity_critical"].notna().any() else np.nan,
        "flux": float(np.nanmedian(f["flux_lmh"])) if len(f) else np.nan,
        "q_recirc": float(np.nanmedian(f["reject_flow_l_per_min"])) if len(f) else np.nan,
        "flux_setpoints": int(f["feed_pump_set_flux_lmh"].nunique()) if len(f) else 0,
        "recirc_setpoints": int(f["recirculation_pump_set_flow_l_per_min"].nunique()) if len(f) else 0,
        "p_start_psi": float(np.nanmedian(f["feed_pressure_psi"].to_numpy(float)[:20])) if len(f) else np.nan,
        "p_end_psi": float(np.nanmedian(f["feed_pressure_psi"].to_numpy(float)[-20:])) if len(f) else np.nan,
        "temp_c": float(np.nanmedian(f["feed_temperature_c"])) if len(f) else np.nan,
        "permeate_us_cm": float(np.nanmedian(f["permeate_conductivity_us_per_cm"])) if len(f) else np.nan,
        "drain_c_start": float(np.nanmedian(d["reject_conductivity_ms_per_cm"].to_numpy(float)[:20])) if len(d) else np.nan,
        "drain_c_end": float(np.nanmedian(d["reject_conductivity_ms_per_cm"].to_numpy(float)[-5:])) if len(d) else np.nan,
    })
    return rec


def batch_table(df: pd.DataFrame, cfg: CCROConfig | None = None) -> pd.DataFrame:
    """One row per file/batch, with the drain of its own brine (from the next file)."""
    cfg = cfg or CCROConfig()
    rows = [file_summary(g) for _, g in df.groupby(["dataset", "exp", "file"], observed=True, sort=False)]
    b = pd.DataFrame(rows).sort_values(["dataset", "exp", "seq"]).reset_index(drop=True)
    for col in ("brine_l", "brine_feed_l", "own_drain_kwh", "own_drain_min", "gap_next_s"):
        b[col] = np.nan
    for _, g in b.groupby(["dataset", "exp"], sort=False):
        idx = g.index.to_list()
        for a, n in zip(idx[:-1], idx[1:]):
            gap = (b.at[n, "t_first"] - b.at[a, "t_last"]).total_seconds()
            b.at[a, "gap_next_s"] = gap
            if (b.at[n, "seq"] == b.at[a, "seq"] + 1 and 0 <= gap < cfg.max_gap_s
                    and b.at[n, "drain_min"] > 0):
                b.at[a, "brine_l"] = b.at[n, "drain_Vrej_l"]
                b.at[a, "brine_feed_l"] = b.at[n, "drain_Vfeed_l"]
                b.at[a, "own_drain_kwh"] = b.at[n, "drain_E_kwh"]
                b.at[a, "own_drain_min"] = b.at[n, "drain_min"]
    b["V_l"] = b.startup_Vp_l + b.stab_Vp_l + b.filt_Vp_l
    b["E_kwh"] = b.startup_E_kwh + b.stab_E_kwh + b.filt_E_kwh + b.own_drain_kwh
    b["sec"] = b.E_kwh / b.V_l * 1000.0
    b["sbc"] = b.brine_l / b.V_l
    b["recovery"] = b.V_l / (b.V_l + b.brine_l)
    b["constant_setpoints"] = (b.flux_setpoints <= 1) & (b.recirc_setpoints <= 1)
    # The fixed cost the rule may use at decision time: the drain opening this file
    # (the previous batch's brine), or the development defaults for a run's first batch.
    has_prev = b.drain_min > 0
    b["prior_drain_kwh"] = np.where(has_prev, b.drain_E_kwh, cfg.default_drain_kwh)
    b["prior_brine_l"] = np.where(has_prev, b.drain_Vrej_l, cfg.default_brine_l)
    return b


def batch_curve(g: pd.DataFrame, cfg: CCROConfig | None = None) -> pd.DataFrame:
    """Sample-by-sample cost curve of one batch (start-up, stabilisation, filtration)."""
    cfg = cfg or CCROConfig()
    g = g.sort_values("row")
    ph = g["state"].map(phase_of).to_numpy()
    keep = np.isin(ph, ("startup", "stab", "filt"))
    g, ph = g[keep], ph[keep]
    t = _times(g)
    P = _power(g)
    Q = g["permeate_flow_l_per_min"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        live = np.where(Q > cfg.min_permeate_lpm, P / (Q * 60.0 / 1000.0), np.nan)   # kWh/m3
    c = (pd.Series(g["reject_conductivity_ms_per_cm"].to_numpy(float))
         .rolling(cfg.conductivity_smooth_samples, center=True, min_periods=1).median().to_numpy())
    return pd.DataFrame({
        "t_s": t, "elapsed_min": (t - t[0]) / 60.0, "phase": ph,
        "E_kwh": cumulative(t, P) / 3600.0,
        "V_m3": cumulative(t, Q) / 60.0 / 1000.0,
        "live_kwh_m3": live, "c_ms_cm": c,
    })


def cost_curve(curve: pd.DataFrame, drain_kwh: float, brine_l: float, b: float) -> np.ndarray:
    """J along the batch if it ended at each sample: (E + E_drain + b V_brine) / V, kWh/m3."""
    fixed = drain_kwh + b * brine_l / 1000.0
    return average_cost(fixed, curve["E_kwh"].to_numpy(), curve["V_m3"].to_numpy())


def sqrt_law_forecast(curve: pd.DataFrame, fixed_prior: float, fit_from_min: float = 5.0,
                      fit_to_min: float = 20.0) -> dict | None:
    """The square-root law as an early forecast of the stop.

    Fit live = m0 + s V on the filtration samples between `fit_from_min` and
    `fit_to_min` after start-up; the start-up and stabilisation energy that a
    straight line does not charge is added to the fixed cost. None if the batch
    is too short to fit (it must run 5 minutes past the fit window).
    """
    filt = curve["phase"].to_numpy() == "filt"
    el = curve["elapsed_min"].to_numpy()
    live = curve["live_kwh_m3"].to_numpy()
    V = curve["V_m3"].to_numpy()
    E = curve["E_kwh"].to_numpy()
    w = filt & (el >= fit_from_min) & (el <= fit_to_min) & np.isfinite(live)
    if w.sum() < 50 or not filt.any() or el[filt][-1] < fit_to_min + 5.0:
        return None
    s, m0 = np.polyfit(V[w], live[w], 1)
    i0 = int(np.where(filt)[0][0])
    f_eff = fixed_prior + E[i0] - m0 * V[i0]
    v_star, j_star = sqrt_law(f_eff, s, m0)
    return {"s_kwh_m6": float(s), "m0_kwh_m3": float(m0), "F_eff_kwh": float(f_eff),
            "V_star_l": float(v_star * 1000.0), "J_star": float(j_star)}


def score_batch(curve: pd.DataFrame, row: pd.Series, b: float, cfg: CCROConfig | None = None) -> dict:
    """The rule on one batch, scored against the batch's own best stopping point.

    The rule sees only the past samples and the previous drain; the score uses
    the drain of this batch's own brine. `inside` says whether the batch's own
    minimum lies far enough before its end for the regret to be exact.
    """
    cfg = cfg or CCROConfig()
    filt = curve["phase"].to_numpy() == "filt"
    t = curve["t_s"].to_numpy()
    J = cost_curve(curve, row["own_drain_kwh"], row["brine_l"], b)
    ok = filt & np.isfinite(J) & (curve["V_m3"].to_numpy() > 0)
    out = {"b": b, "evaluable": False}
    if ok.sum() < 50 or not np.isfinite(row["brine_l"]):
        return out
    i_ok = np.where(ok)[0]
    i_end = i_ok[-1]
    i_best = i_ok[np.nanargmin(J[i_ok])]
    fixed_prior = row["prior_drain_kwh"] + b * row["prior_brine_l"] / 1000.0
    dec = marginal_equals_average(t, curve["live_kwh_m3"].to_numpy(), curve["E_kwh"].to_numpy(),
                                  curve["V_m3"].to_numpy(), fixed_prior, cfg.live_window_s,
                                  min_elapsed_s=cfg.min_batch_min * 60.0, allowed=filt)
    i_rule = dec.index if dec.index is not None else i_end
    c = curve["c_ms_cm"].to_numpy()
    filt_min = (t[i_end] - t[i_ok[0]]) / 60.0
    # live cost against the average at the controller's own end: still falling if m < J
    m_s = rolling_median(t, curve["live_kwh_m3"].to_numpy(), cfg.live_window_s)
    out.update({
        "evaluable": True,
        "inside": bool((t[i_end] - t[i_best]) / 60.0 >= cfg.inside_margin_min and filt_min >= cfg.min_filtration_min),
        "rule_stopped": dec.index is not None,
        "c_rule": float(c[i_rule]), "c_best": float(c[i_best]), "c_end": float(c[i_end]),
        "V_rule_l": float(curve["V_m3"].iat[i_rule] * 1000), "V_best_l": float(curve["V_m3"].iat[i_best] * 1000),
        "V_end_l": float(curve["V_m3"].iat[i_end] * 1000),
        "J_rule": float(J[i_rule]), "J_best": float(J[i_best]), "J_end": float(J[i_end]),
        "live_end": float(m_s[i_end]), "falling_at_end": bool(m_s[i_end] < J[i_end]),
    })
    out["regret_rule_pct"] = 100.0 * (out["J_rule"] / out["J_best"] - 1.0)
    out["regret_end_pct"] = 100.0 * (out["J_end"] / out["J_best"] - 1.0)

    # the square-root law's early forecast, and the cost of stopping where it says
    V = curve["V_m3"].to_numpy()
    law = sqrt_law_forecast(curve, fixed_prior)
    out.update({"law_V_star_l": np.nan, "law_J": np.nan})
    if law is not None:
        out["law_V_star_l"] = law["V_star_l"]
        v_star = law["V_star_l"] / 1000.0
        if v_star <= V[i_end]:
            k = i_ok[min(np.searchsorted(V[i_ok], v_star), len(i_ok) - 1)]
            out["law_J"] = float(J[k])
    out["law_ratio"] = out["law_V_star_l"] / out["V_best_l"] if out["V_best_l"] > 0 else np.nan
    out["law_regret_pct"] = 100.0 * (out["law_J"] / out["J_best"] - 1.0)

    # fixed end points: the cost had the batch ended when the conductivity first reached c
    c_run = np.maximum.accumulate(np.nan_to_num(c[i_ok], nan=-np.inf))
    for cf in (10, 15, 20, 25):
        k = np.searchsorted(c_run, cf)
        out[f"J_c{cf}"] = float(J[i_ok[k]]) if k < len(i_ok) and i_ok[k] < i_end else np.nan
    return out
