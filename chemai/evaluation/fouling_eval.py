"""Scoring a fouling forecast - against the truth where it is known, and against
what was later measured where it is not.

The unit of every score is the CLEANING CYCLE. Days inside one cycle share one
fouling layer, one cleaning and one operator's decision; forecasts made a week
apart inside one cycle are not independent tests. Summaries report how many
cycles stand behind each number, and the split between development and test is
by PLANT in simulation, never by day or by cycle within a plant.

Two kinds of truth:

  * simulation: the counterfactual - the path the train would have followed had
    nobody cleaned it, so every forecast has an answer;
  * the real deposit: the measured 7-day median, which a cleaning cuts short.
    A crossing that never happened before the next cleaning is CENSORED: the
    truth is "later than the cleaning", and it is scored as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chemai.config import ROConfig, ROSimConfig
from chemai.data.ro_sim import PlantRun, counterfactual
from chemai.features.ro_physics import fit_flow_exponent, normalise
from chemai.models.fouling_forecast import forecast_cycle, level_at

PAD_TRUTH = 365


@dataclass
class ObservedCycle:
    """A cycle as a plant would see it: day since cleaning, DP_norm, censored flag."""
    plant: int
    train: str
    start: int                      # last cleaning day
    end: int                        # next cleaning day (or record end + 1)
    closed: bool                    # ended by a cleaning
    day: np.ndarray = field(repr=False)
    dp_norm: np.ndarray = field(repr=False)
    censored: np.ndarray = field(repr=False)
    law: str = ""

    @property
    def id(self) -> str:
        return f"{self.plant}:{self.train}:{self.start}"


def plant_view(run: PlantRun, flow_exponent: float | None = None, visc_exponent: float | None = None,
               ro: ROConfig | None = None, sim: ROSimConfig | None = None,
               sensor: str = "dpt") -> pd.DataFrame:
    """Recorded days of every train, normalised the way the plant does it, and
    censored at the baseline the way the deposit is (DP_norm <= 1 -> 1, flagged).

    sensor='dpt' reads a differential transmitter; 'pt' takes feed minus
    concentrate pressure from two transmitters that drift independently.
    """
    ro, sim = ro or ROConfig(), sim or ROSimConfig()
    out = []
    for tr, g in run.daily[run.daily["recorded"]].groupby("train"):
        g = g.sort_values("day").copy()
        if sensor == "pt":
            g = g.drop(columns=["dp"])
        elif sensor != "dpt":
            raise ValueError("sensor must be 'dpt' or 'pt'")
        if flow_exponent == "fit":
            # the plant's own exponent, from its first 120 recorded days only
            fit = fit_flow_exponent(g[g["day"] <= g["day"].min() + 120])
            fe = fit["n"] if fit["ok"] else ro.dp_flow_exponent
            ve = 2.0 - fe
        else:
            fe = ro.vendor_flow_exponent if flow_exponent is None else flow_exponent
            ve = ro.dp_visc_exponent if visc_exponent is None else visc_exponent
        n = normalise(g, baseline_rows=sim.baseline_days, flow_exponent=fe, visc_exponent=ve, cfg=ro)
        g["dp_norm_raw"] = n["dp_norm"].to_numpy()
        g["censored"] = g["dp_norm_raw"] <= 1.0
        g["dp_norm_obs"] = np.clip(g["dp_norm_raw"], 1.0, 2.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def sim_cycles(run: PlantRun, plant_id: int, view: pd.DataFrame | None = None, **kw) -> list[ObservedCycle]:
    view = plant_view(run, **kw) if view is None else view
    cycles = []
    for tr, g in view.groupby("train"):
        cips = sorted(run.cleanings.loc[run.cleanings["train"] == tr, "day"].astype(int))
        edges = [(c, n, True) for c, n in zip(cips[:-1], cips[1:])]
        if cips:
            edges.append((cips[-1], int(run.daily["day"].max()) + 1, False))
        for s, e, closed in edges:
            seg = g[(g["day"] > s) & (g["day"] < e)]
            cycles.append(ObservedCycle(plant=plant_id, train=str(tr), start=int(s), end=int(e),
                                        closed=closed, day=(seg["day"].to_numpy() - s).astype(float),
                                        dp_norm=seg["dp_norm_obs"].to_numpy(float),
                                        censored=seg["censored"].to_numpy(bool),
                                        law=run.params.law))
    return cycles


def _first_at_or_above(path: np.ndarray, level: float) -> float:
    hit = np.nonzero(path >= level)[0]
    return float(hit[0] + 1) if len(hit) else np.inf


def score_sim(runs: list[PlantRun], levels=(1.15, 1.20), law: str | None = None,
              cfg: ROConfig | None = None, every: int = 7, first: int = 14,
              plant_ids: list[int] | None = None, sensor: str = "dpt",
              flow_exponent: float | str | None = None, visc_exponent: float | None = None,
              cache: dict | None = None) -> pd.DataFrame:
    """Forecast at weekly decision days inside every simulated cycle; score each
    against the counterfactual crossing (the true path without cleaning)."""
    cfg = cfg or ROConfig()
    rows = []
    for k, run in enumerate(runs):
        pid = plant_ids[k] if plant_ids else k
        view = plant_view(run, flow_exponent, visc_exponent, cfg, sensor=sensor)
        true_now = run.daily.set_index(["train", "day"])["dp_true"]
        # The plant reads DP_norm against its OWN start-up measurement. Express the
        # truth on that scale (one constant per train), so the score measures the
        # forecast, not an offset in the baseline that no method could know.
        scale = {tr: float(np.median(g["dp_norm_raw"].to_numpy() /
                                     true_now.loc[list(zip(g["train"], g["day"]))].to_numpy()))
                 for tr, g in view.groupby("train")}
        lengths: dict[str, list[int]] = {}
        for cyc in sim_cycles(run, pid, view):
            prev = lengths.get(cyc.train, [])
            prev_median = float(np.median(prev)) if prev else np.nan
            if cyc.closed:
                lengths.setdefault(cyc.train, []).append(int(cyc.end - cyc.start))
            for d in range(first, int(cyc.end - cyc.start), every):
                m = cyc.day <= d
                if m.sum() < 3:
                    continue
                recent = cyc.dp_norm[m & (cyc.day > d - cfg.smooth_days)]
                obs_now = float(np.median(recent)) if len(recent) else np.nan
                abs_day = cyc.start + d
                k_sc = scale[cyc.train]
                key = (pid, cyc.train, abs_day)
                if cache is not None and key in cache:
                    raw = cache[key]
                else:
                    raw = counterfactual(run, cyc.train, abs_day, PAD_TRUTH)
                    if cache is not None:
                        cache[key] = raw
                path = raw * k_sc
                dp_now = float(true_now.get((cyc.train, abs_day), np.nan)) * k_sc
                for lv in levels:
                    f = forecast_cycle(cyc.day[m], cyc.dp_norm[m], lv, law, cyc.censored[m], cfg)
                    truth = 0.0 if (np.isfinite(dp_now) and dp_now >= lv) else _first_at_or_above(path, lv)
                    rows.append(dict(
                        plant=pid, train=cyc.train, cycle=cyc.id, law_true=cyc.law, level=lv,
                        decision_day=d, abs_day=abs_day, status=f.status, p50=f.days_to_level,
                        p10=f.days_lo, p90=f.days_hi, ratio=f.extrapolation_ratio,
                        in_envelope=f.in_envelope, n_points=f.n_points, span=f.span_days,
                        level_now=f.level_now, rate=f.rate_per_week, truth=truth,
                        true_level_now=dp_now, obs_level_now=obs_now, prev_cycle_median=prev_median,
                        scale=k_sc))
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame, horizon: float = 180.0) -> dict:
    """Headline numbers on GENUINE forecasts: decisions taken while the train was
    still below the level (truth > 0). A train already above the level needs no
    forecast, and counting those zeros would flatter every method.

    Errors are in days, on forecasts that gave a date and whose truth falls inside
    the horizon. Coverage counts an open upper bound (beyond the horizon) as
    covering any later truth.
    """
    s = scores[scores["status"].isin(["crossing", "not_in_sight", "falling"])
               & (scores["truth"] > 0)].copy()
    s["p50f"] = s["p50"].fillna(np.inf)
    inside = s[np.isfinite(s["truth"]) & (s["truth"] <= horizon)]
    pred = inside[np.isfinite(inside["p50f"])]
    err = (pred["p50f"] - pred["truth"]).abs()
    lo = s["p10"].fillna(horizon * 10)
    hi = s["p90"].fillna(np.inf)
    truth = s["truth"].where(s["truth"] <= horizon, np.inf)
    covered = ((truth >= lo) | (lo > horizon)) & (truth <= hi)
    out = dict(
        forecasts=int(len(s)), cycles=int(s["cycle"].nunique()),
        truth_inside_horizon=int(len(inside)), dated=int(len(pred)),
        missed_in_sight=int(len(inside) - len(pred)),
        mae_days=round(float(err.mean()), 1) if len(err) else None,
        median_abs_err_days=round(float(err.median()), 1) if len(err) else None,
        p90_abs_err_days=round(float(err.quantile(0.9)), 1) if len(err) else None,
        bias_days=round(float((pred["p50f"] - pred["truth"]).median()), 1) if len(pred) else None,
        coverage=round(float(covered.mean()), 3) if len(s) else None,
    )
    fl, ok = pred[~pred["in_envelope"]], pred[pred["in_envelope"]]
    if len(fl) and len(ok):
        e_ok = float((ok["p50f"] - ok["truth"]).abs().median())
        e_fl = float((fl["p50f"] - fl["truth"]).abs().median())
        out.update(flagged=int(len(fl)), median_err_inside=round(e_ok, 1),
                   median_err_flagged=round(e_fl, 1), flag_ratio=round(e_fl / max(e_ok, 0.5), 2))
    return out


# ------------------------------------------------------------------ real data

def score_real(cycles, levels=(1.15, 1.20), law: str | None = None, cfg: ROConfig | None = None,
               every: int = 7, first: int = 14, horizons=(14, 28),
               include_initial: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The external test on the deposit. Returns (crossing scores, level scores).

    Crossing truth: first day after the decision when the trailing 7-day median
    of the measured DP_norm reaches the level, before the next cleaning. If the
    cleaning came first, the truth is CENSORED: "at least until the cleaning".
    Level truth: the 7-day median measured `h` days later, where measured.
    The 'initial' stretch from start-up to the first cleaning is a cycle too.
    """
    cfg = cfg or ROConfig()
    cr, lv_rows = [], []
    for cyc in cycles:
        if cyc.kind == "initial" and not include_initial:
            continue
        s = pd.Series(cyc.dp_norm, index=cyc.day.astype(int))
        full = s.reindex(range(0, int(cyc.length) + 1))
        med = full.rolling(cfg.smooth_days, min_periods=3).median()
        for d in range(first, int(cyc.length), every):
            m = cyc.day <= d
            if m.sum() < 3:
                continue
            now = med[med.index <= d].dropna()
            obs_now = float(now.iloc[-1]) if len(now) and now.index[-1] >= d - cfg.smooth_days else np.nan
            for lv in levels:
                f = forecast_cycle(cyc.day[m], cyc.dp_norm[m], lv, law, cyc.censored[m], cfg)
                after = med[med.index > d]
                if np.isfinite(obs_now) and obs_now >= lv:
                    truth, censored = 0.0, False
                else:
                    hit = after[after >= lv]
                    if len(hit):
                        truth, censored = float(hit.index[0] - d), False
                    else:
                        truth, censored = float(cyc.length - d), True     # at least this long
                cr.append(dict(cycle=cyc.id, train=cyc.train, kind=cyc.kind, level=lv, decision_day=d,
                               abs_day=int(cyc.start + d), status=f.status, p50=f.days_to_level,
                               p10=f.days_lo, p90=f.days_hi, ratio=f.extrapolation_ratio,
                               in_envelope=f.in_envelope, level_now=f.level_now, obs_level_now=obs_now,
                               rate=f.rate_per_week, truth=truth, truth_censored=censored,
                               warnings="; ".join(f.warnings)))
            for h in horizons:
                pred = level_at(cyc.day[m], cyc.dp_norm[m], h, law, cyc.censored[m], cfg)
                tgt = med.get(d + h, np.nan) if (d + h) <= cyc.length else np.nan
                if pred is not None and np.isfinite(tgt) and np.isfinite(obs_now):
                    lv_rows.append(dict(cycle=cyc.id, train=cyc.train, kind=cyc.kind, decision_day=d,
                                        horizon=h, predicted=pred, measured=float(tgt),
                                        persistence=obs_now))
    return pd.DataFrame(cr), pd.DataFrame(lv_rows)


def real_decisions(cr: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
    """Decisions with a known answer on real data: crossed within `horizon` days
    (1), or ran at least `horizon` days without crossing (0). A cleaning before
    the horizon without a crossing leaves the answer unknown: excluded."""
    g = cr[(cr["truth"] > 0) & (cr["status"] != "insufficient")].copy()
    pos = (~g["truth_censored"]) & (g["truth"] <= horizon)
    neg = (g["truth"] > horizon)
    g = g[pos | neg].copy()
    g["y"] = pos[pos | neg].astype(int)
    return g
