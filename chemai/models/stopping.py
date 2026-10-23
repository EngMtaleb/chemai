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


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median of pairwise slopes and the matching intercept - robust to the odd bad day."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan"), float("nan")
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    keep = dx != 0
    if not keep.any():
        return float("nan"), float("nan")
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    return slope, float(np.median(y - slope * x))


def implied_fixed_cost(slope: float, interval: float) -> float:
    """The fixed cost that would make `interval` the best one under linear growth: s T^2 / 2."""
    return float(slope * interval ** 2 / 2.0)


@dataclass(frozen=True)
class IntervalEconomics:
    """One cycle of a slow process (days between cleanings), judged by the rule and the law."""
    length: float              # the interval actually run
    slope: float               # growth of the live cost per unit time (Theil-Sen)
    m0: float                  # live cost at the start of the cycle (intercept)
    t_star: float              # the square-root law's best interval, sqrt(2 F / slope)
    rule_time: float | None    # first time the smoothed live cost reached the average cost
    implied_fixed: float       # the fixed cost that would make the actual interval the best one
    j_actual: float            # average cost of the cycle as run: F / T + mean live cost


def interval_economics(t: np.ndarray, live: np.ndarray, fixed: float, smooth: float,
                       min_t: float = 0.0) -> IntervalEconomics:
    """The rule and the square-root law on one cycle sampled in time (output constant per unit time).

    With output proportional to time, J(t) = (F + integral of m) / t, where F is
    the fixed cost per unit of output rate (for a plant: $ per m3/day of capacity)
    and m the live cost per unit of output ($/m3).
    """
    t = np.asarray(t, float)
    m = np.asarray(live, float)
    ok = np.isfinite(t) & np.isfinite(m)
    t, m = t[ok], m[ok]
    order = np.argsort(t)
    t, m = t[order], m[order]
    length = float(t[-1]) if len(t) else float("nan")
    slope, m0 = theil_sen(t, m)
    t_star = sqrt_law(fixed, slope, m0)[0] if np.isfinite(slope) else float("nan")
    run = cumulative(t, m)                       # integral of m dt, with t from the cycle start
    dec = marginal_equals_average(t * 86400.0, m, run, t, fixed, window_s=smooth * 86400.0,
                                  min_elapsed_s=min_t * 86400.0, min_periods=1)
    j = (fixed + run[-1]) / t[-1] if len(t) and t[-1] > 0 else float("nan")
    return IntervalEconomics(
        length=length, slope=slope, m0=m0, t_star=float(t_star),
        rule_time=float(t[dec.index]) if dec.index is not None else None,
        implied_fixed=implied_fixed_cost(slope, length) if np.isfinite(slope) else float("nan"),
        j_actual=float(j))
