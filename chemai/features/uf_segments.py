"""Filtration segments and their fouling rate.

The UF unit has three elements. From March 2013 on, two filter while the third
is backwashed and rested, and the pair changes every 24-28 minutes; in 2012 all
three filter together and a backwash stops them all. Either way, between two
changes the online membrane is fixed, and its resistance climbs as the feed
deposits on it. That climb, per hour, is the fouling rate.

    R = TMP / (mu(T) * Q)        resistance index (membrane area is not in the record)
    fouling rate = d ln R / dt   per hour, fitted on 1-minute medians

Viscosity carries the temperature correction (Darcy's law): a cold morning
raises TMP without any fouling, and the index must not read that as fouling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chemai.config import UFSegmentConfig


def seconds(index) -> np.ndarray:
    """Seconds since the epoch, whatever resolution the index carries."""
    return pd.DatetimeIndex(index).as_unit("ns").asi8 / 1e9


def water_viscosity(temp_c) -> np.ndarray:
    """Vogel equation for water, Pa s. The salinity factor (~1.08) is constant and cancels."""
    t_k = np.asarray(temp_c, float) + 273.15
    return 2.414e-5 * 10 ** (247.8 / (t_k - 140.0))


def resistance_index(df: pd.DataFrame) -> pd.Series:
    tmp = df["p_uf_in"] - df["p_uf_out"]
    return tmp / (water_viscosity(df["temp_c"]) * df["q_uf"])


def configuration(df: pd.DataFrame, cfg: UFSegmentConfig | None = None) -> np.ndarray:
    """Which elements are filtering, as a code 0-7, debounced.

    A change counts only once the new code has held for `debounce_s`; a meter
    flickering around the threshold does not cut a segment.
    """
    cfg = cfg or UFSegmentConfig()
    on = df[["q_e1", "q_e2", "q_e3"]].to_numpy(float) > cfg.element_on_gpm
    raw = on[:, 0] * 1 + on[:, 1] * 2 + on[:, 2] * 4
    t = seconds(df.index)
    out = raw.copy()
    current, start_new, candidate = raw[0], None, None
    for i in range(len(raw)):
        if raw[i] == current:
            candidate, start_new = None, None
            out[i] = current
            continue
        if raw[i] != candidate:
            candidate, start_new = raw[i], i
        if t[i] - t[start_new] >= cfg.debounce_s:
            current = candidate
            out[start_new:i + 1] = current
            candidate, start_new = None, None
        else:
            out[i] = current
    return out


def steady_mask(df: pd.DataFrame, cfg: UFSegmentConfig | None = None) -> np.ndarray:
    cfg = cfg or UFSegmentConfig()
    q = df["q_uf"].to_numpy(float)
    q_med = np.nanmedian(q[q > 1.0]) if np.any(q > 1.0) else np.nan
    tmp = (df["p_uf_in"] - df["p_uf_out"]).to_numpy(float)
    ok = np.abs(q - q_med) <= cfg.flow_band * q_med
    ok &= tmp > cfg.min_tmp_psi
    ok &= np.isfinite(df["temp_c"].to_numpy(float))
    return ok


def cut_segments(df: pd.DataFrame, cfg: UFSegmentConfig | None = None) -> list[tuple[int, int]]:
    """(first, last) row positions of each segment, after trimming its transients."""
    cfg = cfg or UFSegmentConfig()
    if len(df) < 2:
        return []
    ok = steady_mask(df, cfg)
    code = configuration(df, cfg)
    t = seconds(df.index)
    dt = np.median(np.diff(t))
    max_gap = max(3 * dt, 5.0)
    breaks = np.zeros(len(df), bool)
    breaks[1:] = (code[1:] != code[:-1]) | (np.diff(t) > max_gap) | (ok[1:] != ok[:-1])
    starts = np.flatnonzero(breaks | (np.arange(len(df)) == 0))
    ends = np.r_[starts[1:] - 1, len(df) - 1]
    out = []
    for a, b in zip(starts, ends):
        if not ok[a]:
            continue
        a2 = np.searchsorted(t, t[a] + cfg.trim_start_s, "left")
        b2 = np.searchsorted(t, t[b] - cfg.trim_end_s, "right") - 1
        if b2 <= a2 or (t[b2] - t[a2]) < cfg.min_minutes * 60:
            continue
        out.append((int(a2), int(b2)))
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = np.sum((x - xm) * (y - ym)) / sxx
    resid = y - ym - b * (x - xm)
    se = np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1) / sxx)
    return float(b), float(se)


def segment_table(df: pd.DataFrame, run: str, cfg: UFSegmentConfig | None = None) -> pd.DataFrame:
    """One row per segment: when it ran, its fouling rate, and what the feed and pump did.

    `fouling_pct_h` = 100 * d ln R / dt on 1-minute medians (on the logging
    interval where that is coarser). Means of turbidity,
    pump speed and MF inlet pressure are kept for the mechanism and the
    positive control; nothing tidal is computed here.
    """
    cfg = cfg or UFSegmentConfig()
    if df.empty:
        return pd.DataFrame()
    code = configuration(df, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        lnr = np.log(resistance_index(df).to_numpy(float))
    dt = float(np.median(np.diff(seconds(df.index)))) if len(df) > 1 else cfg.bin_s
    bin_s = max(cfg.bin_s, round(dt))
    rows = []
    for a, b in cut_segments(df, cfg):
        seg = df.iloc[a:b + 1]
        s = pd.Series(lnr[a:b + 1], index=seg.index)
        binned = s.resample(f"{int(bin_s)}s").median().dropna()
        binned = binned[np.isfinite(binned)]
        if len(binned) < cfg.min_points:
            continue
        x = (seconds(binned.index) - seconds(binned.index)[0]) / 3600          # hours
        slope, se = _ols_slope(x, binned.to_numpy(float))
        tu = seg["turbidity"].to_numpy(float)
        rows.append(dict(
            run=run,
            t_start=seg.index[0], t_end=seg.index[-1],
            t_mid=seg.index[0] + (seg.index[-1] - seg.index[0]) / 2,
            minutes=(seg.index[-1] - seg.index[0]).total_seconds() / 60,
            n_bins=len(binned),
            config=int(np.bincount(code[a:b + 1]).argmax()),
            n_online=int(bin(int(np.bincount(code[a:b + 1]).argmax())).count("1")),
            fouling_pct_h=100 * slope,
            fouling_se_pct_h=100 * se,
            ln_r_start=float(binned.iloc[0]),
            q_gpm=float(seg["q_uf"].mean()),
            tmp_psi=float((seg["p_uf_in"] - seg["p_uf_out"]).mean()),
            temp_c=float(seg["temp_c"].mean()),
            turbidity=float(np.nanmean(tu)) if np.all(np.isfinite(tu)) and np.all(tu > -1) else np.nan,
            pump_speed=float(seg["pump_speed"].mean()),
            p_mf_in=float(seg["p_mf_in"].mean()),
        ))
    return pd.DataFrame(rows)


def segment_curves(df: pd.DataFrame, run: str, cfg: UFSegmentConfig | None = None) -> pd.DataFrame:
    """The binned ln R curve of every segment `segment_table` keeps, in long form:
    one row per (segment, bin) with minutes since the segment's first bin.

    Same cutting, same bins, same exclusions as `segment_table`, so a curve joins
    its segment row on (run, t_start)."""
    cfg = cfg or UFSegmentConfig()
    if df.empty:
        return pd.DataFrame()
    with np.errstate(invalid="ignore", divide="ignore"):
        lnr = np.log(resistance_index(df).to_numpy(float))
    dt = float(np.median(np.diff(seconds(df.index)))) if len(df) > 1 else cfg.bin_s
    bin_s = max(cfg.bin_s, round(dt))
    parts = []
    for a, b in cut_segments(df, cfg):
        seg = df.iloc[a:b + 1]
        s = pd.Series(lnr[a:b + 1], index=seg.index)
        binned = s.resample(f"{int(bin_s)}s").median().dropna()
        binned = binned[np.isfinite(binned)]
        if len(binned) < cfg.min_points:
            continue
        x = (seconds(binned.index) - seconds(binned.index)[0]) / 60.0          # minutes
        parts.append(pd.DataFrame({"run": run, "t_start": seg.index[0], "minute": x,
                                   "lnr": binned.to_numpy(float), "bin_s": bin_s}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
