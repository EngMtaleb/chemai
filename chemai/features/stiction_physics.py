"""Stiction physics read from CO and CV alone - the valve the historian does not keep.

Two estimators, both written before any test or plant result (PROTOCOL_physics.md):

`reversal_travel` - ADOPTED. The definition of S, measured on the trend. After a
reversal the command must travel S before a sticky valve moves (Choudhury,
Thornhill & Shah, 2005). So: find every reversal of CO, find the next reversal of
CV - the moment the valve finally moved - and measure how far CO travelled in
between. The median over the record is the estimate of S. A healthy valve also
shows some travel, because the process takes time to answer; the classifier
learns to tell the two apart from the other features. On the simulated
development set the estimate follows the true S with a rank correlation of 0.86
(0.94 where the slip stands clearly above the load noise); on new plants, touched
once, 0.72 with a median error of 0.5 % of travel (VALIDATION.md 23.2).

`fit_valve` - REJECTED at development, kept for the record. The Hammerstein
approach (e.g. Jelali, 2008): pass CO through the stiction model for a grid of
(S, J), fit a linear process model from the reconstructed valve position to CV,
keep the candidate that explains CV best. On closed-loop routine data it fails
for a reason worth knowing: CO already carries the disturbance through the
feedback, so even the TRUE valve position explains CV worse than CO does in
one-step prediction (VALIDATION.md 23.1). The valve has to be measured, or read
from the timing of the reversals - it cannot be fitted back from CV this way.

Units. CO is scaled by its standard deviation, so both estimators return S in CO
standard deviations, comparable across plants that store CO in different units;
`s_co` converts back to the units of CO (% of travel when CO is stored in %).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Candidate S, in CO standard deviations. A stiction limit cycle moves CO by
# about S between slips, so S reaches a few standard deviations of CO.
S_GRID: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.3, 2.9, 3.5, 4.2)
J_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)      # J / S; 0 is pure deadband
ARX_OUTPUT_LAGS = 2
ARX_INPUT_LAGS = 10
MAX_SAMPLES = 6000


def _valve_path_py(x: np.ndarray, s: float, j: float) -> np.ndarray:
    """MV for a whole record of commands x - the same logic as
    `chemai.data.loop_sim.StictionValve.step`, without the 0-100 % clip
    (x is scaled here)."""
    n = x.shape[0]
    out = np.empty(n)
    if s == 0.0:
        for k in range(n):
            out[k] = x[k]
        return out
    half = (s - j) / 2.0
    pos = x[0]
    x_prev = x[0]
    x_stick = x[0]
    d = 0.0
    stuck = True
    for k in range(n):
        xk = x[k]
        if not stuck:
            v = np.sign(xk - x_prev)
            if v == d and v != 0.0:
                pos = xk - d * half                  # keeps moving, lagging by (S - J) / 2
            else:
                stuck = True                         # stops
                x_stick = x_prev
        if stuck:
            dx = xk - x_stick
            reversal = d == 0.0 or np.sign(dx) != d
            threshold = s if reversal else j
            if abs(dx) > threshold:                  # slip
                d = np.sign(dx)
                pos = xk - d * half
                stuck = False
        out[k] = pos
        x_prev = xk
    return out


try:                                                 # the loop is ~60x faster compiled
    from numba import njit
    _valve_path = njit(cache=True)(_valve_path_py)
except ImportError:                                  # pragma: no cover - optional speed-up
    _valve_path = _valve_path_py


@dataclass(frozen=True)
class ValveFit:
    s: float           # S, in CO standard deviations (nan when the record cannot be fitted)
    j_ratio: float     # J / S at the best candidate
    gain: float        # 1 - RSS(best) / RSS(perfect valve)
    co_scale: float    # standard deviation of CO in its own units
    n: int             # samples used

    @property
    def s_co(self) -> float:
        """S in the units of CO - % of travel when CO is stored in %."""
        return self.s * self.co_scale

    def as_row(self) -> dict:
        return {"s_hat": self.s, "j_ratio_hat": self.j_ratio, "fit_gain": self.gain}


def fit_valve(co, cv, s_grid: tuple[float, ...] = S_GRID, j_ratios: tuple[float, ...] = J_RATIOS,
              output_lags: int = ARX_OUTPUT_LAGS, input_lags: int = ARX_INPUT_LAGS,
              max_samples: int = MAX_SAMPLES) -> ValveFit:
    """Best (S, J) for one loop from its CO and CV records."""
    co = np.asarray(co, dtype=float)[:max_samples]
    cv = np.asarray(cv, dtype=float)[:max_samples]
    n = min(len(co), len(cv))
    co, cv = co[:n], cv[:n]
    sc, sy = (float(np.std(co)), float(np.std(cv))) if n else (0.0, 0.0)
    if n < 200 or not (np.isfinite(co).all() and np.isfinite(cv).all()) or sc < 1e-9 or sy < 1e-9:
        return ValveFit(float("nan"), float("nan"), float("nan"), sc, n)

    x = (co - np.median(co)) / sc
    y = (cv - cv.mean()) / sy
    m = max(output_lags, input_lags)
    target = y[m:]
    y_lags = np.column_stack([y[m - i:n - i] for i in range(1, output_lags + 1)])
    ones = np.ones(n - m)

    def rss(u: np.ndarray) -> float:
        u_lags = np.column_stack([u[m - i:n - i] for i in range(1, input_lags + 1)])
        X = np.column_stack([y_lags, u_lags, ones])
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        r = target - X @ beta
        return float(r @ r)

    r_perfect = rss(x)
    best = (r_perfect, 0.0, 0.0)
    for s in s_grid:
        for jr in j_ratios:
            r = rss(_valve_path(x, float(s), float(s * jr)))
            if r < best[0]:
                best = (r, float(s), float(jr))
    gain = 1.0 - best[0] / r_perfect if r_perfect > 0 else 0.0
    return ValveFit(best[1], best[2], float(gain), sc, n)


# ================================================================ adopted: S from the reversals

REVERSAL_SMOOTHING = 3        # samples, moving average before looking for reversals
REVERSAL_HYSTERESIS = 0.5     # a reversal must come back by this many standard deviations
MIN_REVERSALS = 4


def _moving_average(z: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return z
    return np.convolve(z, np.ones(w) / w, mode="same")


def turning_points(z: np.ndarray, h: float) -> list[int]:
    """Indices of the extrema where `z` reverses by at least `h` (zig-zag with hysteresis).
    Small wiggles inside `h` are not reversals - a sticky valve ignores them too."""
    out: list[int] = []
    i_ext, v_ext, d = 0, float(z[0]), 0
    for i in range(1, len(z)):
        v = float(z[i])
        if d == 0:
            if v - v_ext > h:
                d, i_ext, v_ext = 1, i, v
            elif v_ext - v > h:
                d, i_ext, v_ext = -1, i, v
        elif d == 1:
            if v > v_ext:
                i_ext, v_ext = i, v
            elif v_ext - v > h:
                out.append(i_ext)
                d, i_ext, v_ext = -1, i, v
        else:
            if v < v_ext:
                i_ext, v_ext = i, v
            elif v - v_ext > h:
                out.append(i_ext)
                d, i_ext, v_ext = 1, i, v
    return out


@dataclass(frozen=True)
class ReversalTravel:
    s: float           # median CO travel from a CO reversal to the next CV reversal, CO std units
    lag: float         # median time of that wait, as a fraction of CO's half-cycle
    reversals: int     # CO reversals found
    co_scale: float    # standard deviation of CO in its own units

    @property
    def s_co(self) -> float:
        """S in the units of CO - % of travel when CO is stored in %."""
        return self.s * self.co_scale

    def as_row(self) -> dict:
        return {"s_travel": self.s, "travel_lag": self.lag, "reversals": self.reversals}


def reversal_travel(co, cv, smoothing: int = REVERSAL_SMOOTHING,
                    hysteresis: float = REVERSAL_HYSTERESIS,
                    max_samples: int = MAX_SAMPLES) -> ReversalTravel:
    """How far CO travels after it reverses before CV answers - S, read from the trend."""
    co = np.asarray(co, dtype=float)[:max_samples]
    cv = np.asarray(cv, dtype=float)[:max_samples]
    n = min(len(co), len(cv))
    co, cv = co[:n], cv[:n]
    sc = float(np.std(co)) if n else 0.0
    nan = float("nan")
    if n < 200 or not (np.isfinite(co).all() and np.isfinite(cv).all()) or sc < 1e-9 or np.std(cv) < 1e-9:
        return ReversalTravel(nan, nan, 0, sc)
    x = _moving_average((co - co.mean()) / sc, smoothing)
    y = _moving_average((cv - cv.mean()) / np.std(cv), smoothing)
    tx = turning_points(x, hysteresis)
    ty = np.asarray(turning_points(y, hysteresis), dtype=int)
    if len(tx) < MIN_REVERSALS or len(ty) == 0:
        return ReversalTravel(nan, nan, len(tx), sc)
    travel, wait = [], []
    for i in tx:
        later = ty[ty > i]
        if len(later) == 0:
            break
        j = int(later[0])
        travel.append(abs(x[j] - x[i]))
        wait.append(j - i)
    half = float(np.median(np.diff(tx)))
    if not travel or half <= 0:
        return ReversalTravel(nan, nan, len(tx), sc)
    return ReversalTravel(float(np.median(travel)), float(np.median(wait) / half), len(tx), sc)
