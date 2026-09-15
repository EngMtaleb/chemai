"""Oscillation shape - triangular (a sticky valve) or sinusoidal (tuning)?

Curve-fitting stiction index of He, Wang, Pottmann & Qin (2007): each half
cycle between zero crossings is fitted with a half sine and with a
free-apex triangle, and the index is

    SI = MSE_sine / (MSE_sine + MSE_triangle)

near 1 = triangular, near 0 = sinusoidal.

Three limits, measured, not assumed (VALIDATION.md section 16):

  * a SQUARE wave reads as a sine (0.44) - so fast loops are measured on OP,
    where the sticky fingerprint is triangular, and level loops on PV, where
    the vessel integrates the square into a triangle;
  * noise collapses every shape onto 0.5 - `prepare` low-pass filters first,
    and the filter settings are ours, not part of the published method;
  * a triangular fingerprint TRAVELS: a loop shaken by a sticky valve
    elsewhere inherits the triangle.

The index is evidence, never a verdict: on 31 real regularly-oscillating
loops it catches 15 of 20 stiction cases at a threshold of 0.6.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import butter, detrend, filtfilt

__all__ = ["prepare_for_shape", "half_cycle_fits", "shape_index", "shape_signal"]


def prepare_for_shape(x: np.ndarray, period: float, harmonics: int = 7) -> np.ndarray:
    """Detrend, remove slow drift, and low-pass at `harmonics` x the oscillation.

    Without this, noise of 0.3 x amplitude pulls every shape to about 0.5, because
    noise adds spurious zero crossings that chop the half cycles. `harmonics`
    must stay high enough to keep a triangle triangular - a triangle IS its
    harmonics, and filtering too hard turns it into a sine by hand.
    """
    x = detrend(np.asarray(x, dtype=float))
    if period <= 0 or not np.isfinite(period):
        raise ValueError("period must be a positive number of samples")
    w = int(max(5, 4 * period))
    if len(x) > 6 * period:
        kernel = np.ones(w) / w
        padded = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
        x = x - np.convolve(padded, kernel, "valid")
    b, a = butter(2, min(0.99, 2 * harmonics / period))
    return filtfilt(b, a, x)


def half_cycle_fits(seg: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit one half cycle with a half sine and a free-apex triangle.
    Returns (sine MSE, triangle MSE, fitted sine, fitted triangle)."""
    seg = np.asarray(seg, dtype=float)
    n = len(seg)
    if n < 3:
        raise ValueError("half cycle too short to fit")
    t = np.arange(n) / (n - 1)
    s = np.sin(np.pi * t)
    amp = (s @ seg) / (s @ s)
    e_sine = float(np.mean((seg - amp * s) ** 2))

    def triangle(p):
        tr = np.where(t <= p, t / p, (1 - t) / (1 - p))
        return tr * ((tr @ seg) / (tr @ tr))

    best = minimize_scalar(lambda p: np.mean((seg - triangle(p)) ** 2),
                           bounds=(0.05, 0.95), method="bounded")
    return e_sine, float(best.fun), amp * s, triangle(best.x)


def shape_index(x: np.ndarray, min_half_cycle: int = 6, min_half_cycles: int = 4) -> float:
    """Curve-fitting index over all usable half cycles of `x`.
    NaN when fewer than `min_half_cycles` are usable - not enough evidence."""
    x = np.asarray(x, dtype=float)
    zc = np.flatnonzero(np.diff(np.sign(x)) != 0)
    sine_errors, triangle_errors = [], []
    for a, b in zip(zc[:-1], zc[1:]):
        seg = x[a + 1:b + 1]
        if len(seg) < min_half_cycle:
            continue
        e_s, e_t, *_ = half_cycle_fits(seg)
        sine_errors.append(e_s)
        triangle_errors.append(e_t)
    if len(sine_errors) < min_half_cycles:
        return float("nan")
    total = np.sum(sine_errors) + np.sum(triangle_errors)
    return float(np.sum(sine_errors) / total) if total else float("nan")


def shape_signal(loop_type: str, pv: np.ndarray, sp: np.ndarray, op: np.ndarray
                 ) -> tuple[np.ndarray, str]:
    """Which signal carries the triangular fingerprint for this loop type.

    Self-regulating (F, P, Q, T): OP - the integral ramps it while the valve
    sticks, and PV tends to square, which the index misreads (0.44).
    Integrating (L): PV - the vessel integrates the square flow into a triangle.
    Falls back to the control error when OP is missing or never moves.
    """
    pv, sp, op = (np.asarray(v, dtype=float) for v in (pv, sp, op))
    usable_op = np.isfinite(op).all() and len(op) == len(pv) and np.ptp(op) > 0
    if loop_type.upper() in ("F", "P", "Q", "T") and usable_op:
        return op, "OP"
    return pv - sp, "PV"
