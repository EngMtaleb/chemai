"""Part B: from the first five minutes of a UF filtration cycle, how far will the
membrane's resistance climb by minute eleven?

The cycle is a Project 4 filtration segment (steady flow, one set of elements
online). At minute 5 the plant has seen five binned points of ln R, the cycles
before this one, and the conditions it set (flow per element, temperature).
The target is the climb from minute ~4 to minute ~11, in percent of R:

    y = 100 * [ mean ln R over 10-12 min  -  mean ln R over 3-5 min ]

Hermia's blocking laws, written for constant flux (the pilot holds its flow),
give R(t)/R0 for four mechanisms:

    complete blocking      1 / (1 - K t)
    standard blocking      (1 - K t / 2)^-2
    intermediate blocking  exp(K t)            <- ln R straight in time: Project 4's slope
    cake filtration        1 + K t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR = (3.0, 5.0)          # minutes: where the climb is measured from (inside what the plant has seen)
TARGET = (10.0, 12.0)        # minutes: where it is measured to
SEEN = 5.0                   # minutes of the cycle the plant has seen at the decision
TAU0, TAU1 = 4.0, 11.0       # centres of the two windows
LAWS = ("complete", "standard", "intermediate", "cake")
CLEANINGS = pd.to_datetime(["2013-10-09", "2014-03-18", "2014-07-28"])   # Project 5, from the record

PLAIN = ["early_slope", "early_curv", "early_resid", "early_n", "flow_per_element", "temp_c", "n_online",
         "lnr0", "lnr_rel_episode", "hours_in_episode", "hours_in_run", "days_since_clean",
         "prev_rate", "prev_same_rate", "prev_rise", "med_rate_1h", "n_rate_1h", "med_rate_3h", "gap_prev_min"]
PHYSICS = ["viscosity", "k_complete", "k_standard", "k_intermediate", "k_cake", "best_law",
           "pred_line", "pred_hermia", "pred_history", "pred_history_flux", "flux_ratio"]
CLOCK = ["hour_sin", "hour_cos", "m2_sin", "m2_cos"]


def hermia_ln_ratio(law: str, k, t):
    """ln(R(t)/R0) under one law at constant flux; k per minute, t in minutes."""
    k, t = np.asarray(k, float), np.asarray(t, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if law == "complete":
            return -np.log(np.clip(1 - k * t, 1e-6, None))
        if law == "standard":
            return -2.0 * np.log(np.clip(1 - k * t / 2, 1e-6, None))
        if law == "intermediate":
            return k * t
        if law == "cake":
            return np.log1p(np.clip(k * t, -0.999, None))
    raise ValueError(law)


def fit_hermia(t: np.ndarray, z: np.ndarray) -> dict:
    """Least-squares K of each law on z = ln R(t) - ln R(t0), t from the cycle's first bin.

    A 1-D search on a log grid of K (plus K = 0) - robust, no starting values."""
    grid = np.r_[0.0, np.geomspace(1e-6, 0.05, 200)]
    out = {}
    for law in LAWS:
        pred = np.stack([hermia_ln_ratio(law, k, t) for k in grid])          # (k, n)
        off = (z[None, :] - pred).mean(1, keepdims=True)                      # free level: R0 is noisy too
        sse = ((z[None, :] - pred - off) ** 2).sum(1)
        i = int(np.argmin(sse))
        out[law] = (float(grid[i]), float(sse[i]), float(off[i, 0]))
    return out


def water_viscosity(temp_c) -> np.ndarray:
    t_k = np.asarray(temp_c, float) + 273.15
    return 2.414e-5 * 10 ** (247.8 / (t_k - 140.0))


def build_rows(segments: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """One row per segment that spans at least 12 minutes, with features known at minute 5."""
    seg = segments.sort_values("t_start").reset_index(drop=True).copy()
    seg["key"] = seg["run"] + "|" + seg["t_start"].astype("int64").astype(str)
    cur = curves.copy()
    cur["key"] = cur["run"] + "|" + cur["t_start"].astype("int64").astype(str)
    groups = dict(tuple(cur.groupby("key")))
    seg["flow_per_element"] = seg["q_gpm"] / seg["n_online"].clip(lower=1)
    ep0 = seg.groupby("episode")["t_start"].transform("min")
    run0 = seg.groupby("run")["t_start"].transform("min")
    seg["hours_in_episode"] = (seg["t_start"] - ep0).dt.total_seconds() / 3600
    seg["hours_in_run"] = (seg["t_start"] - run0).dt.total_seconds() / 3600
    first_lnr = seg.groupby("episode")["ln_r_start"].transform("first")
    seg["lnr_rel_episode"] = seg["ln_r_start"] - first_lnr
    rows = []
    hist: dict[int, list] = {}
    for i, s in seg.iterrows():
        c = groups.get(s["key"])
        prev = hist.get(s["episode"], [])
        r = dict(key=s["key"], run=s["run"], episode=int(s["episode"]), t_start=s["t_start"], t_std=s["t_std"],
                 config=int(s["config"]), n_online=int(s["n_online"]), flow_per_element=float(s["flow_per_element"]),
                 temp_c=float(s["temp_c"]), lnr0=float(s["ln_r_start"]), lnr_rel_episode=float(s["lnr_rel_episode"]),
                 hours_in_episode=float(s["hours_in_episode"]), hours_in_run=float(s["hours_in_run"]))
        past = [p for p in prev if (s["t_start"] - p["t_end"]).total_seconds() <= 3 * 3600]
        last = past[-1] if past else None
        same = [p for p in past if p["config"] == s["config"]]
        r["prev_rate"] = last["rate"] if last else np.nan
        r["prev_rise"] = last["rise"] if last else np.nan
        r["prev_flow"] = last["flow"] if last else np.nan
        r["gap_prev_min"] = (s["t_start"] - last["t_end"]).total_seconds() / 60 if last else np.nan
        r["prev_same_rate"] = same[-1]["rate"] if same else np.nan
        h1 = [p["rate"] for p in past if (s["t_start"] - p["t_end"]).total_seconds() <= 3600]
        r["med_rate_1h"] = float(np.median(h1)) if h1 else np.nan
        r["n_rate_1h"] = len(h1)
        r["med_rate_3h"] = float(np.median([p["rate"] for p in past])) if past else np.nan
        before = CLEANINGS[CLEANINGS <= s["t_start"]]
        r["days_since_clean"] = (s["t_start"] - before[-1]).total_seconds() / 86400 if len(before) else np.nan
        rise = np.nan
        if c is not None:
            t, z = c["minute"].to_numpy(float), c["lnr"].to_numpy(float)
            seen = t <= SEEN
            a = z[(t >= ANCHOR[0]) & (t <= ANCHOR[1])]
            b = z[(t >= TARGET[0]) & (t <= TARGET[1])]
            if t[-1] >= TARGET[0] and len(a) and len(b) and seen.sum() >= 3:
                ts, zs = t[seen], z[seen] - z[0]
                p1 = np.polyfit(ts, zs, 1)
                r["early_slope"] = p1[0] * 60 * 100                     # %/h, like Project 4's rate
                r["early_curv"] = np.polyfit(ts, zs, 2)[0] * 100 if seen.sum() >= 4 else np.nan
                r["early_resid"] = float(np.std(zs - np.polyval(p1, ts))) * 100
                r["early_n"] = int(seen.sum())
                h = fit_hermia(ts, zs)
                for law in LAWS:
                    r[f"k_{law}"] = h[law][0] * 60                     # per hour
                best = min(LAWS, key=lambda lw: h[lw][1])
                r["best_law"] = LAWS.index(best)
                r["pred_line"] = p1[0] * (TAU1 - TAU0) * 100
                k = h[best][0]
                r["pred_hermia"] = float(hermia_ln_ratio(best, k, TAU1) - hermia_ln_ratio(best, k, TAU0)) * 100
                r["y"] = float(b.mean() - a.mean()) * 100
                rise = r["y"]
                # what Project 4 measured for the whole segment - kept only for the history of later cycles
        prate = r["prev_same_rate"] if np.isfinite(r["prev_same_rate"]) else r["prev_rate"]
        r["pred_history"] = prate * (TAU1 - TAU0) / 60 if np.isfinite(prate) else np.nan
        r["flux_ratio"] = r["flow_per_element"] / r["prev_flow"] if np.isfinite(r["prev_flow"]) else np.nan
        r["pred_history_flux"] = r["pred_history"] * r["flux_ratio"] if np.isfinite(r["pred_history"]) else np.nan
        r["viscosity"] = float(water_viscosity(s["temp_c"]) * 1e3)
        hour = (s["t_std"].hour + s["t_std"].minute / 60)
        r["hour_sin"], r["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        hrs = (s["t_std"] - pd.Timestamp("2012-01-01")).total_seconds() / 3600
        r["m2_sin"], r["m2_cos"] = np.sin(2 * np.pi * hrs / 12.4206012), np.cos(2 * np.pi * hrs / 12.4206012)
        rows.append(r)
        hist.setdefault(s["episode"], []).append(dict(t_end=s["t_end"], rate=float(s["fouling_pct_h"]),
                                                      rise=rise, config=int(s["config"]),
                                                      flow=float(s["flow_per_element"])))
    return pd.DataFrame(rows)
