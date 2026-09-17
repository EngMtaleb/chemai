"""Plant-wide propagation - one sticking valve, ten loops that look faulty.

A cascade has a known connection (section 19). Here it is unknown: thirty loops
recorded together, many oscillating, and nothing saying which drives which.
The fingerprint is shared frequency - victims shake with a motion that is not
theirs, at the source's period.

    find_oscillation_cluster   which loops carry the same oscillation
    rank_source_candidates     which of them is nearest the source

**What this module does NOT do: identify the source.** Ranking by band share
orders loops by how close they are to the source, which is not the same thing
(VALIDATION.md 20.3). Two biases are measured and must be read with the list:

  * a source sitting in a NOISY loop is outranked by a quiet victim;
  * a NON-LINEAR source is penalised by its own fingerprint - a sticking valve
    makes a square-ish limit cycle whose power spreads into harmonics, so less
    of it lands in the fundamental band than in a victim that received a
    filtered, purer version of the same oscillation.

A simple non-linearity measure was tried and dropped - it measured the noise
floor, ranking the documented root cause LAST. Doing it properly needs
surrogate-data testing (Thornhill et al., 2005), out of scope.

So the output is a CANDIDATE list for an engineer to confirm in the field,
and - more valuable - the knowledge that ten of the eleven "faults" are
probably victims of one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

log = logging.getLogger(__name__)


def loop_spectra(signals: np.ndarray, fs: float, nperseg: int = 2048):
    """(frequencies, power) for every loop. Columns are loops, rows are samples."""
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2:
        raise ValueError("signals must be a 2-D array: samples x loops")
    x = np.column_stack([detrend(x[:, j]) for j in range(x.shape[1])])
    return welch(x, fs=fs, nperseg=min(nperseg, len(x)), axis=0)


def dominant_periods(signals: np.ndarray, fs: float, nperseg: int = 2048) -> np.ndarray:
    """Period of each loop's strongest oscillation, in seconds."""
    f, p = loop_spectra(signals, fs, nperseg)
    return np.array([1 / f[1:][p[1:, j].argmax()] for j in range(p.shape[1])])


def band_share(signals: np.ndarray, period: float, fs: float,
               nperseg: int = 2048, width: int = 1) -> np.ndarray:
    """Fraction of each loop's oscillation power sitting in the band at `period`.

    Near the source the oscillation is almost everything the loop does, so the
    share is high; further away it mixes with local noise and upsets.
    """
    f, p = loop_spectra(signals, fs, nperseg)
    i = int(np.argmin(abs(f - 1 / period)))
    return p[max(i - width, 1):i + width + 1].sum(0) / p[1:].sum(0)


@dataclass
class PropagationEvent:
    period: float                  # seconds
    members: list[str]
    shares: dict[str, float]
    note: str = field(default=(
        "Candidates ordered by band share - nearness to the source, not proof of it. "
        "Confirm the first candidate in the field before working on the others."))

    @property
    def candidates(self) -> pd.DataFrame:
        return (pd.DataFrame({"loop": self.members,
                              "band_share": [self.shares[m] for m in self.members]})
                .sort_values("band_share", ascending=False).reset_index(drop=True))

    def victims(self) -> list[str]:
        """Everything but the leading candidate: do not raise work orders for these
        until the source is repaired and the plant is measured again."""
        return self.candidates.loop.tolist()[1:]


def find_oscillation_cluster(signals: np.ndarray, names: list[str], fs: float,
                             period: float | None = None, min_share: float = 0.3,
                             tolerance: float = 0.1, nperseg: int = 2048
                             ) -> PropagationEvent | None:
    """Loops sharing one oscillation, and their band shares.

    With no `period`, the most frequently occurring dominant period is used -
    the oscillation seen in the largest number of loops.
    """
    signals = np.asarray(signals, dtype=float)
    if len(names) != signals.shape[1]:
        raise ValueError("one name per loop is required")

    if period is None:
        periods = dominant_periods(signals, fs, nperseg)
        counts = [(p, int(np.sum(abs(periods - p) <= tolerance * p))) for p in periods]
        period, n_at = max(counts, key=lambda c: c[1])
        if n_at < 2:
            log.info("no period is shared by two or more loops")
            return None

    shares = band_share(signals, period, fs, nperseg)
    members = [n for n, s in zip(names, shares) if s > min_share]
    if len(members) < 2:
        log.info("only %d loop above the share threshold - no propagation event", len(members))
        return None
    log.info("%d loops share a %.1f min oscillation", len(members), period / 60)
    return PropagationEvent(period=float(period), members=members,
                            shares={n: round(float(s), 3) for n, s in zip(names, shares)})


def rank_source_candidates(event: PropagationEvent) -> pd.DataFrame:
    """The candidate list, most likely source first. See the module docstring for
    why this is a filter and not a verdict."""
    table = event.candidates.copy()
    table["rank"] = np.arange(1, len(table) + 1)
    table["role"] = np.where(table["rank"] == 1, "source candidate", "likely victim")
    table["action"] = np.where(table["rank"] == 1,
                               "Field-check this loop first",
                               "No work order until the source is repaired")
    return table
