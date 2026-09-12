"""Loop performance indices.

Two questions, deliberately kept apart:

    harris_index            HOW MUCH better could this loop be?   (a score)
    oscillation_regularity  IS it oscillating?                     (a detector)

Neither says WHY. A loop with stiction, one tuned too tight and one shaken
by an upstream oscillation can all score the same Harris index - the
diagnosis is a separate step (Week 2).
"""
from __future__ import annotations

import numpy as np


def cycles_in_record(n: int, period: float) -> float:
    """How many oscillation periods the record holds. Below about ten, the
    regularity index is unreliable however clean the signal is (VALIDATION.md
    15.1): the same sine in noise is detected in half the runs at 10 cycles
    and in all of them at 20."""
    return float(n / period) if period and np.isfinite(period) else 0.0


def delay_samples(theta: float, ts: float) -> int:
    """Discrete time delay d, in samples, for dead time `theta` at sampling `ts`.

    d = floor(theta / ts) + 1. The +1 is the zero-order hold: even with no dead
    time, a controller move cannot affect the measurement at the same sample.
    """
    if theta < 0 or ts <= 0:
        raise ValueError("theta must be >= 0 and ts > 0")
    return int(np.floor(theta / ts + 1e-9)) + 1


def _ar_fit(y: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Least-squares AR(order) fit. Returns (phi, innovation variance)."""
    n = len(y)
    if n < 5 * order:
        raise ValueError(f"need at least {5 * order} samples for AR({order}), got {n}")
    X = np.column_stack([y[order - i - 1:n - i - 1] for i in range(order)])
    target = y[order:]
    phi, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ phi
    return phi, float(resid.var())


def harris_index(y: np.ndarray, d: int, ar_order: int = 30) -> float:
    """Harris index: minimum-variance benchmark / actual variance, in (0, 1].

        eta = sigma2_mv / sigma2_y,   sigma2_mv = sigma2_e * sum_{j<d} psi_j^2

    where psi are the impulse-response weights of the fitted AR model and
    sigma2_e its innovation variance (Harris, 1989; Desborough & Harris, 1992).

    eta = 1  the loop is at the minimum-variance limit set by its dead time.
    eta -> 0 far from it.

    `y` should be the control error (SP - PV) of REGULATORY data: the
    benchmark assumes stochastic disturbances, not setpoint changes.
    The result depends on `d` - an overestimated delay flatters the loop.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    y = y - y.mean()
    if d < 1:
        raise ValueError("d must be >= 1")
    var_y = float(y.var())
    if var_y == 0:
        raise ValueError("signal has zero variance")
    phi, var_e = _ar_fit(y, ar_order)
    psi = np.zeros(d)
    psi[0] = 1.0
    for j in range(1, d):
        psi[j] = sum(phi[i] * psi[j - i - 1] for i in range(min(j, ar_order)))
    return float(var_e * np.sum(psi ** 2) / var_y)


def harris_sensitivity(y: np.ndarray, d_values, ar_order: int = 30) -> dict[int, float]:
    """Harris index across a range of assumed delays - report this whenever
    the dead time is not known exactly."""
    return {int(d): harris_index(y, int(d), ar_order) for d in d_values}


def oscillation_regularity(x: np.ndarray) -> tuple[float, float]:
    """Regularity of oscillation from autocorrelation zero crossings
    (after Thornhill, Huang & Zhang, 2003).

    Returns (period_in_samples, r) with r = mean(T) / (3 std(T)), T being the
    intervals between successive ACF zero crossings, doubled. Fewer than four
    zero crossings gives (nan, 0.0): nothing periodic to measure - usually a
    record too short for its own period, not a quiet loop.

    r is a CONFIDENCE SCORE, not a gate. The published r > 1 does not separate
    the real data: seven runs of one sticky valve score 0.98 to 3.15 with no
    gap (VALIDATION.md 15.2). Carry r into the diagnosis; never use it to
    decide which loops get diagnosed.

    The ACF is taken to half the record length. Detrend before calling.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = float(x @ x)
    if denom == 0:
        return float("nan"), 0.0
    r = np.correlate(x, x, "full")[n - 1:n - 1 + n // 2] / denom
    zc = np.where(np.diff(np.sign(r)) != 0)[0]
    if len(zc) < 4:
        return float("nan"), 0.0
    iv = np.diff(zc) * 2.0
    return float(iv.mean()), float(iv.mean() / (3.0 * iv.std() + 1e-9))
