"""Weekly cleaning decisions as a table a machine-learning model can learn from.

One row per (cycle, decision day, review level): what a plant could see on that
day - the cycle's normalised pressure drop so far - and the answer to one
question: **will the 7-day level reach the review level within the next 28 days?**

The same rows carry the two methods the models must beat, computed exactly as
Project 3 froze them: the forecast (P50 date within 28 days) and the margin rule
(the last 7 days within 0.03 of the level).

Two sources:

  * simulated plants - the answer is the counterfactual path, the one the train
    would have followed had nobody cleaned it (`ro_sim.counterfactual`);
  * the real deposit - the measured 7-day median, which a cleaning cuts short; a
    decision whose answer the cleaning hid is dropped (`fouling_eval.real_decisions`).

Features use the observed series only, the same way on both sources, so a model
trained on simulated plants can be run on the real one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.config import ROConfig
from chemai.data.ro_sim import PlantRun, counterfactual
from chemai.evaluation.fouling_eval import PAD_TRUTH, plant_view, sim_cycles
from chemai.models.fouling_forecast import forecast_cycle

HORIZON = 28                 # days: "plan a cleaning" horizon of the weekly plan
SEQ_DAYS = 42                # the trailing window the forecast fits, as a daily sequence
LEVELS = (1.15, 1.20)
STATUS = ("crossing", "not_in_sight", "falling", "at_or_above", "insufficient")

# Plain features: statistics of the observed series, nothing derived from a physical law.
PLAIN = ["level", "day", "now7", "gap", "last", "age_last", "n42", "cens42", "cens14",
         "mean42", "std42", "min42", "max42", "max_cycle", "chg7", "chg14", "chg28",
         "ols14", "ols42", "curv42"]
# Physics-informed: the gap to the level in units of the series' own noise, and what the
# two Project 3 forecasts (straight line, and the gap law ΔP ∝ gap^-n) say.
PHYSICS = ["gap_noise", "lin_p50", "lin_p10", "lin_p90", "lin_rate", "lin_ratio", "lin_env",
           "lin_crossing", "lin_falling", "lin_notinsight", "blk_p50", "blk_rate"]


def _median_in(t: np.ndarray, x: np.ndarray, lo: float, hi: float) -> float:
    v = x[(t > lo) & (t <= hi)]
    return float(np.median(v)) if len(v) else np.nan


def _ols(t: np.ndarray, x: np.ndarray, deg: int = 1) -> np.ndarray:
    if len(t) < deg + 2 or np.ptp(t) < 3:
        return np.full(deg + 1, np.nan)
    return np.polyfit(t - t[-1], x, deg)


def _days(v, cap: float = 365.0) -> float:
    return cap if v is None or not np.isfinite(v) else float(min(v, cap))


def decision_features(day, dp, cens, d: float, level: float, cfg: ROConfig | None = None) -> tuple[dict, np.ndarray]:
    """Everything a model may use at decision day `d`, and the 42-day daily sequence.

    `day`, `dp`, `cens`: the cycle's observed days since the cleaning, DP_norm
    (censored days held at 1.0) and the censored flags - only days <= d are read.
    """
    cfg = cfg or ROConfig()
    m = day <= d
    t, x, c = day[m].astype(float), dp[m].astype(float), cens[m].astype(bool)
    f: dict = {"level": level, "day": float(d)}
    now7 = _median_in(t, x, d - cfg.smooth_days, d)
    f["now7"], f["gap"] = now7, level - now7
    f["last"] = float(x[-1]) if len(x) else np.nan
    f["age_last"] = float(d - t[-1]) if len(t) else np.nan
    w = t > d - SEQ_DAYS
    tw, xw, cw = t[w], x[w], c[w]
    f["n42"] = int(len(xw))
    f["cens42"] = float(cw.mean()) if len(cw) else np.nan
    c14 = c[t > d - 14]
    f["cens14"] = float(c14.mean()) if len(c14) else np.nan
    for k, fn in (("mean42", np.mean), ("std42", np.std), ("min42", np.min), ("max42", np.max)):
        f[k] = float(fn(xw)) if len(xw) else np.nan
    f["max_cycle"] = float(x.max()) if len(x) else np.nan
    for lag in (7, 14, 28):
        f[f"chg{lag}"] = now7 - _median_in(t, x, d - lag - cfg.smooth_days, d - lag)
    w14 = t > d - 14
    f["ols14"] = float(_ols(t[w14], x[w14])[0] * 7)                 # per week
    p1, p2 = _ols(tw, xw), _ols(tw, xw, 2)
    f["ols42"] = float(p1[0] * 7)
    f["curv42"] = float(p2[0] * 49)                                   # per week^2
    resid = xw - np.polyval(p1, tw - tw[-1]) if np.all(np.isfinite(p1)) else np.array([])
    noise = float(np.std(resid)) if len(resid) > 2 else np.nan
    f["gap_noise"] = f["gap"] / max(noise, 0.003) if np.isfinite(noise) else np.nan

    lin = forecast_cycle(t, x, level, "linear", c, cfg)
    blk = forecast_cycle(t, x, level, "blockage", c, cfg)
    f["status"] = lin.status
    f["lin_p50"], f["lin_p10"], f["lin_p90"] = (_days(v) for v in (lin.days_to_level, lin.days_lo, lin.days_hi))
    f["lin_rate"] = lin.rate_per_week if lin.rate_per_week is not None else np.nan
    f["lin_ratio"] = lin.extrapolation_ratio if lin.extrapolation_ratio is not None else 0.0
    f["lin_env"] = float(lin.in_envelope)
    f["lin_crossing"] = float(lin.status == "crossing")
    f["lin_falling"] = float(lin.status == "falling")
    f["lin_notinsight"] = float(lin.status == "not_in_sight")
    f["blk_p50"] = _days(blk.days_to_level)
    f["blk_rate"] = blk.rate_per_week if blk.rate_per_week is not None else np.nan
    # the two methods to beat, as Project 3 froze them
    f["forecast_flag"] = bool(lin.status == "crossing" and lin.days_to_level is not None
                              and lin.days_to_level <= HORIZON)
    f["margin_flag"] = bool(np.isfinite(now7) and now7 >= level - cfg.plan_margin)

    seq = np.full((2, SEQ_DAYS), np.nan)                             # value, censored
    idx = (t[w] - (d - SEQ_DAYS + 1)).astype(int)
    ok = (idx >= 0) & (idx < SEQ_DAYS)
    seq[0, idx[ok]] = xw[ok]
    seq[1, idx[ok]] = cw[ok]
    return f, seq


def sim_decisions(runs: list[PlantRun], plant_ids: list[int], cfg: ROConfig | None = None,
                  flow_exponent: float | str = 1.5, visc_exponent: float | None = 0.25,
                  every: int = 7, first: int = 14) -> tuple[pd.DataFrame, np.ndarray]:
    """Decision rows of simulated plants, normalised the way the deposit is (vendor 1.5)
    unless told otherwise, with the counterfactual answer."""
    cfg = cfg or ROConfig()
    rows, seqs = [], []
    for run, pid in zip(runs, plant_ids):
        view = plant_view(run, flow_exponent, visc_exponent, cfg)
        true_now = run.daily.set_index(["train", "day"])["dp_true"]
        scale = {tr: float(np.median(g["dp_norm_raw"].to_numpy() /
                                     true_now.loc[list(zip(g["train"], g["day"]))].to_numpy()))
                 for tr, g in view.groupby("train")}
        for cyc in sim_cycles(run, pid, view):
            for d in range(first, int(cyc.end - cyc.start), every):
                m = cyc.day <= d
                if m.sum() < 3:
                    continue
                abs_day = cyc.start + d
                path = counterfactual(run, cyc.train, abs_day, PAD_TRUTH) * scale[cyc.train]
                dp_now = float(true_now.get((cyc.train, abs_day), np.nan)) * scale[cyc.train]
                for lv in LEVELS:
                    if np.isfinite(dp_now) and dp_now >= lv:
                        continue                                    # already there: no forecast needed
                    hit = np.nonzero(path >= lv)[0]
                    truth = float(hit[0] + 1) if len(hit) else np.inf
                    f, seq = decision_features(cyc.day, cyc.dp_norm, cyc.censored, d, lv, cfg)
                    f.update(source="sim", plant=pid, train=cyc.train, cycle=cyc.id, law_true=cyc.law,
                             decision_day=d, truth=truth, y=int(truth <= HORIZON))
                    rows.append(f)
                    seqs.append(seq)
    return pd.DataFrame(rows), np.asarray(seqs)


def real_decisions_table(cycles, cfg: ROConfig | None = None, every: int = 7, first: int = 14
                         ) -> tuple[pd.DataFrame, np.ndarray]:
    """Decision rows of the real deposit, with the answer where the record shows it:
    crossed within 28 days (1), ran 28 days without crossing (0); a cleaning that came
    first without a crossing hides the answer, and the row is dropped."""
    cfg = cfg or ROConfig()
    rows, seqs = [], []
    for cyc in cycles:
        s = pd.Series(cyc.dp_norm, index=cyc.day.astype(int))
        full = s.reindex(range(0, int(cyc.length) + 1))
        med = full.rolling(cfg.smooth_days, min_periods=3).median()
        for d in range(first, int(cyc.length), every):
            m = cyc.day <= d
            if m.sum() < 3:
                continue
            now = med[med.index <= d].dropna()
            obs_now = float(now.iloc[-1]) if len(now) and now.index[-1] >= d - cfg.smooth_days else np.nan
            for lv in LEVELS:
                if np.isfinite(obs_now) and obs_now >= lv:
                    continue
                after = med[med.index > d]
                hit = after[after >= lv]
                if len(hit):
                    truth, censored = float(hit.index[0] - d), False
                else:
                    truth, censored = float(cyc.length - d), True
                if censored and truth <= HORIZON:
                    continue                                        # the cleaning hid the answer
                f, seq = decision_features(cyc.day, cyc.dp_norm, cyc.censored, d, lv, cfg)
                # the margin rule exactly as Project 3 scored it on this record: the trailing
                # 7-day median of the daily grid, needing 3 recorded days
                f["margin_flag"] = bool(np.isfinite(obs_now) and obs_now >= lv - cfg.plan_margin)
                f.update(source="real", plant=-1, train=cyc.train, cycle=cyc.id, law_true="",
                         decision_day=d, truth=truth, y=int((not censored) and truth <= HORIZON),
                         obs_now_p3=obs_now)
                rows.append(f)
                seqs.append(seq)
    return pd.DataFrame(rows), np.asarray(seqs)
