"""When to stop a cycle: the marginal = average rule and the square-root law.

Every cycle in a desalination plant has the same economics. Running it costs
more and more per cubic metre - the loop concentrates, the membrane fouls -
and ending it costs a fixed amount: a drain and its brine, a backwash, a
chemical clean. Per cubic metre over the whole cycle,

    J(V) = (F + K(V)) / V          F = fixed cost of ending the cycle
                                   K(V) = running cost of the first V m3

and J is lowest where its derivative is zero:

    m(V) = dK/dV = J(V)            stop when the live cost of the next cubic
                                   metre reaches the average cost so far.

The rule needs no model of the process - only the live cost signal the plant
already logs, and the fixed cost of the last stop. If the live cost grows in
a straight line, m = m0 + s V, the stop has a closed form,

    V* = sqrt(2 F / s),   J* = m0 + sqrt(2 F s)

the square-root law of cleaning intervals (and of order quantities). A cycle
twice as costly to end should run sqrt(2) times longer, not twice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cumulative(t_s: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Trapezoid running integral of `rate` over time in seconds (NaN counted as 0)."""
    t = np.asarray(t_s, float)
    y = np.nan_to_num(np.asarray(rate, float))
    if len(t) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def average_cost(fixed: float, running: np.ndarray, output: np.ndarray) -> np.ndarray:
    """J = (F + K) / V, NaN where no output yet."""
    v = np.asarray(output, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, (fixed + np.asarray(running, float)) / v, np.nan)


def rolling_median(t_s: np.ndarray, x: np.ndarray, window_s: float, min_periods: int = 3) -> np.ndarray:
    """Trailing median over a time window (only the past - the rule runs live)."""
    idx = pd.to_datetime(np.asarray(t_s, float), unit="s")
    s = pd.Series(np.asarray(x, float), index=idx)
    return s.rolling(f"{int(round(window_s))}s", min_periods=min_periods).median().to_numpy()


@dataclass(frozen=True)
class StopDecision:
    index: int | None          # first sample where the rule says stop; None if it never does
    marginal: float            # smoothed live cost there (or at the end)
    average: float             # average cost there (or at the end)


def marginal_equals_average(t_s: np.ndarray, live_marginal: np.ndarray, running: np.ndarray,
                            output: np.ndarray, fixed: float, window_s: float,
                            min_elapsed_s: float = 0.0, allowed: np.ndarray | None = None,
                            min_periods: int = 3) -> StopDecision:
    """Online stop: the first sample where the smoothed live cost >= the average cost.

    t_s           sample times, seconds, increasing
    live_marginal the plant's live cost per unit output at each sample
    running       cumulative running cost K up to each sample
    output        cumulative output V up to each sample
    fixed         the fixed cost F of ending the cycle (known from the last stop)
    window_s      trailing median window for the live cost
    allowed       samples where a stop may be taken (e.g. the filtration phase)
    """
    t = np.asarray(t_s, float)
    m = rolling_median(t, live_marginal, window_s, min_periods)
    j = average_cost(fixed, running, output)
    ok = np.isfinite(m) & np.isfinite(j) & ((t - t[0]) >= min_elapsed_s)
    if allowed is not None:
        ok &= np.asarray(allowed, bool)
    hit = np.where(ok & (m >= j))[0]
    if len(hit):
        i = int(hit[0])
        return StopDecision(i, float(m[i]), float(j[i]))
    last = np.where(ok)[0]
    i = int(last[-1]) if len(last) else len(t) - 1
    return StopDecision(None, float(m[i]) if np.isfinite(m[i]) else float("nan"), float(j[i]))


def sqrt_law(fixed: float, slope: float, m0: float = 0.0) -> tuple[float, float]:
    """Linear growth m = m0 + s V: the best cycle V* = sqrt(2F/s) and its cost J* = m0 + sqrt(2Fs)."""
    if fixed <= 0 or slope <= 0:
        return float("inf"), float(m0)
    return float(np.sqrt(2.0 * fixed / slope)), float(m0 + np.sqrt(2.0 * fixed * slope))


def cost_of_stopping_at(v_stop: float, fixed: float, slope: float, m0: float = 0.0) -> float:
    """Average cost of a cycle stopped at V under linear growth: m0 + F/V + s V / 2."""
    return float(m0 + fixed / v_stop + slope * v_stop / 2.0)
