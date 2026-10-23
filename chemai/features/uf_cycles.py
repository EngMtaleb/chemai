"""UF backwash cycles and chemical-cleaning intervals, for the stopping economics (Project 5).

Two decisions on the same membrane, at two time scales:

* backwash (tens of minutes): each backwash spends filtrate and pump energy;
  between backwashes the transmembrane pressure (TMP) climbs, so every cubic
  metre costs a little more to push through;
* chemical cleaning (months): what the backwashes leave behind accumulates as
  a rising post-backwash resistance; a chemical clean resets most of it.

Both are cycles with a fixed cost to end them and a running cost that grows.
This module measures the pieces from the record; `chemai.models.stopping`
does the economics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.features.uf_segments import seconds

GAL_M3 = 3.785411784e-3
PSI_BAR = 0.0689476


def backwash_events(df: pd.DataFrame, on_gpm: float = 5.0, min_s: float = 20.0,
                    merge_s: float = 120.0) -> pd.DataFrame:
    """One row per backwash: start, duration, volume (m3) and pump energy (kWh).

    A backwash is backwash flow above `on_gpm`; pulses closer than `merge_s`
    are one backwash (the high-flow pulse and the lower rinse that follows);
    events shorter than `min_s` in total are logger noise.
    """
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "seconds", "volume_m3", "energy_kwh"])
    t = seconds(df.index)
    qb = df["q_bw"].clip(lower=0).to_numpy(float)
    pb = df["p_bw_filt"].clip(lower=0).to_numpy(float) * PSI_BAR * 100.0   # kPa
    on = np.nan_to_num(qb) > on_gpm
    dt = np.diff(t, append=t[-1])
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    spans = list(zip(edges[::2], edges[1::2] - 1))
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and t[a] - t[merged[-1][1]] <= merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rows = []
    for a, b in merged:
        sl = slice(a, b + 1)
        q_m3s = qb[sl] * GAL_M3 / 60.0
        dur = float(t[b] - t[a] + dt[b])
        if dur < min_s:
            continue
        rows.append({"start": df.index[a], "end": df.index[b], "seconds": dur,
                     "volume_m3": float(np.nansum(q_m3s * dt[sl])),
                     "energy_kwh": float(np.nansum(pb[sl] * q_m3s * dt[sl])) / 3600.0})
    return pd.DataFrame(rows)


def run_summary(df: pd.DataFrame) -> dict:
    """What one run filtered, how often it backwashed, and at what pressures."""
    t = seconds(df.index)
    dt = np.diff(t, append=t[-1])
    q = df["q_uf"].clip(lower=0).to_numpy(float)
    filtering = np.nan_to_num(q) > 1.0
    v_in = float(np.nansum(q * GAL_M3 / 60.0 * dt))
    ev = backwash_events(df)
    tmp = (df["p_uf_in"] - df["p_uf_out"]).to_numpy(float)
    return {
        "hours": float((t[-1] - t[0]) / 3600.0) if len(t) else 0.0,
        "v_in_m3": v_in,
        "backwashes": int(len(ev)),
        "v_bw_m3": float(ev.volume_m3.sum()) if len(ev) else 0.0,
        "bw_energy_kwh": float(ev.energy_kwh.sum()) if len(ev) else 0.0,
        "bw_seconds_median": float(ev.seconds.median()) if len(ev) else np.nan,
        "q_m3h": float(np.nanmedian(q[filtering])) * GAL_M3 * 60.0 if filtering.any() else np.nan,
        "p_in_bar": float(np.nanmedian(df["p_uf_in"].to_numpy(float)[filtering])) * PSI_BAR if filtering.any() else np.nan,
        "tmp_bar": float(np.nanmedian(tmp[filtering])) * PSI_BAR if filtering.any() else np.nan,
    }


def daily_baseline(segments: pd.DataFrame, min_segments: int = 5) -> pd.DataFrame:
    """Daily median of the per-element resistance at the start of each filtration segment.

    R is a resistance index of the elements online (TMP / viscosity / total
    flow); with n elements sharing the flow, one element's index is n times
    the unit's, so ln R_element = ln R_unit + ln n.
    """
    s = segments[segments["config"].isin([3, 5, 6, 7])].copy()
    s["lnr_element"] = s["ln_r_start"] + np.log(s["n_online"])
    s["day"] = pd.DatetimeIndex(s["t_start"]).normalize()
    d = s.groupby("day").agg(lnr=("lnr_element", "median"), segments=("lnr_element", "size"),
                             tmp_psi=("tmp_psi", "median"), q_gpm=("q_gpm", "median"),
                             temp_c=("temp_c", "median"))
    return d[d.segments >= min_segments]


def cleaning_days(daily: pd.DataFrame, drop: float = 0.4, max_gap_days: int = 30) -> list[pd.Timestamp]:
    """Days the element resistance fell by more than `drop` (in ln R) from the previous logged day.

    A fall that large (TMP roughly halved at the same flow and temperature) is a
    chemical clean, not a backwash; across a gap longer than `max_gap_days` the
    cause cannot be seen and the fall is not counted.
    """
    lnr = daily["lnr"]
    gap = daily.index.to_series().diff().dt.days
    hit = (lnr.diff() < -drop) & (gap <= max_gap_days)
    return list(daily.index[hit.to_numpy()])
