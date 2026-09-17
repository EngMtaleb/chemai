"""Archive compression - what a historian throws away, and what that costs.

A plant historian does not store every sample. It reduces the rate, applies a
deadband, or fits straight lines between turning points, because keeping
hundreds of thousands of tags at one second for years is expensive.

Every index in this package reads the SHAPE of a signal, and compression
changes the shape. These functions reproduce the two common methods so any
diagnosis can be tested under plant conditions before it is trusted
(VALIDATION.md 21).

The finding that matters: a deadband does not make a diagnosis uncertain, it
makes it WRONG. Sticky valves come back as 'tuning', which sends the control
engineer instead of maintenance.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def downsample(data, factor: int):
    """Keep every `factor`-th sample: a historian logging at a slower rate.

    Accepts an array or a DataFrame. The first sample is always kept, so the
    result starts at the same instant.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    if isinstance(data, pd.DataFrame):
        return data.iloc[::factor].reset_index(drop=True)
    return np.asarray(data)[::factor]


def apply_deadband(x: np.ndarray, band: float) -> np.ndarray:
    """Store a new value only when the signal has moved more than `band`;
    otherwise repeat the last stored value.

    `band` is in the signal's own units - scale it by the signal's standard
    deviation to compare loops. Corners are the first thing a deadband destroys,
    and corners are exactly what the shape index measures (§16).
    """
    x = np.asarray(x, dtype=float)
    if band < 0:
        raise ValueError("band must not be negative")
    if band == 0 or len(x) == 0:
        return x.copy()
    out = np.empty_like(x)
    stored = x[0]
    for i, value in enumerate(x):
        if abs(value - stored) > band:
            stored = value
        out[i] = stored
    return out


def compress(data: pd.DataFrame, factor: int = 1, deadband_std: float = 0.0,
             columns: tuple[str, ...] = ("pv", "op")) -> pd.DataFrame:
    """Both effects together, as a historian applies them: a deadband on each
    stored tag, then a slower logging rate. `deadband_std` is in units of each
    column's own standard deviation."""
    out = data.copy()
    if deadband_std:
        for column in columns:
            if column in out:
                out[column] = apply_deadband(out[column].to_numpy(),
                                             deadband_std * out[column].std())
    return downsample(out, factor)


def samples_per_cycle(period: float, factor: int = 1) -> float:
    """How many samples fall inside one oscillation period at this logging rate.

    The measured limit (VALIDATION.md 21.3): the diagnosis holds above about ten
    samples per cycle and collapses below five. This is a property of the
    ARCHIVE, not of the algorithm - no method recovers a shape that was never
    stored.
    """
    if not np.isfinite(period) or period <= 0 or factor < 1:
        return 0.0
    return float(period / factor)


def resolution_is_sufficient(period: float, factor: int = 1, minimum: float = 10.0) -> bool:
    """Is the record fine enough to read the waveform shape at all?"""
    return samples_per_cycle(period, factor) >= minimum
