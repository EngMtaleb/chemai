"""Data-quality checks - run on every loop BEFORE any diagnosis.

A loop's shape can lie for reasons that have nothing to do with the valve or
the tuning. Five traps, each with one rule (projects/p02_control_loops/
VALIDATION.md, section 14; the teaching notebook p02_week2_data_quality):

    frozen sensor     PV flat for a long stretch WHILE OP moves
    manual / inactive OP flat for a long stretch, or never moving
    saturation        OP staying at its recorded limit
    quantisation      PV taking only a handful of distinct values
    moving setpoint   the indices assume a constant SP
    coarse archive    fewer than ten samples per oscillation cycle (Week 6)

Checks FLAG; they do not repair. Downstream code decides what a flag means
for its own purpose (a quantised PV spoils PV-shape analysis but not OP-shape
analysis).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chemai.config import DataQualityConfig
from chemai.data.compression import samples_per_cycle


# ------------------------------------------------------------- primitives

def longest_flat_run(x: np.ndarray) -> tuple[int, int]:
    """(length, start) of the longest stretch where x does not change at all.
    Length counts the repeated samples: [5, 5, 5] has a run of 2."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0, 0
    same = np.r_[False, np.diff(x) == 0]
    edges = np.diff(np.r_[0, same.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return 0, 0
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return int(lengths[i]), int(starts[i]) - 1        # run begins one sample earlier


def n_levels(x: np.ndarray) -> int:
    """Number of distinct values, ignoring floating-point dust."""
    x = np.asarray(x, dtype=float)
    return int(len(np.unique(np.round(x[np.isfinite(x)], 10))))


def saturation_fraction(op: np.ndarray, min_run: int = 3) -> float:
    """Fraction of samples inside stays of at least `min_run` samples at the
    recorded minimum or maximum of OP. A single touch of the limit is not counted."""
    op = np.asarray(op, dtype=float)
    if len(op) == 0 or np.ptp(op) == 0:
        return 0.0
    at = (op <= op.min()) | (op >= op.max())
    edges = np.diff(np.r_[0, at.astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths[lengths >= min_run].sum() / len(op))


def sp_segments(sp: np.ndarray) -> tuple[int, float, slice]:
    """(number of SP changes, fraction of samples where SP changes,
    slice of the longest constant-SP segment)."""
    sp = np.asarray(sp, dtype=float)
    changed = np.diff(sp) != 0
    seg = np.r_[0, np.cumsum(changed)]
    longest = int(np.bincount(seg).argmax())
    idx = np.flatnonzero(seg == longest)
    return int(changed.sum()), float(changed.mean()) if len(changed) else 0.0, \
        slice(int(idx[0]), int(idx[-1]) + 1)


def pi_fit(sp: np.ndarray, pv: np.ndarray, op: np.ndarray) -> tuple[float, float]:
    """Does OP follow a PI law on the error?  dOP = Kc (de + dt/Ti e).
    Returns (R2, Kc estimate). R2 = 0 when OP never moves."""
    e = np.asarray(sp, float) - np.asarray(pv, float)
    dop = np.diff(np.asarray(op, float))
    if len(dop) < 3 or np.var(dop) == 0:
        return 0.0, 0.0
    X = np.column_stack([np.diff(e), e[1:]])
    coef, *_ = np.linalg.lstsq(X, dop, rcond=None)
    r2 = 1.0 - np.var(dop - X @ coef) / np.var(dop)
    return float(r2), float(coef[0])


# ----------------------------------------------------------------- report

@dataclass
class QualityReport:
    n: int
    pv_flat_run: int
    op_flat_run: int | None
    pv_levels: int
    op_levels: int | None
    saturation: float | None
    sp_changes: int
    sp_change_fraction: float
    regulatory_segment: slice
    pi_r2: float | None
    kc_estimate: float | None
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok_for_harris(self) -> bool:
        """Regulatory data with a working controller and an honest sensor."""
        blocking = {"frozen_sensor", "op_constant", "op_inactive", "saturated",
                    "moving_setpoint", "short_regulatory_segment", "no_op"}
        return not blocking & set(self.flags)

    @property
    def ok_for_pv_shape(self) -> bool:
        return not {"frozen_sensor", "quantised_pv", "coarse_archive"} & set(self.flags)

    @property
    def ok_for_op_shape(self) -> bool:
        return not {"frozen_sensor", "op_constant", "op_coarse", "no_op",
                    "coarse_archive"} & set(self.flags)


def assess(sp, pv, op, cfg: DataQualityConfig | None = None,
           period: float | None = None) -> QualityReport:
    """Run the checks on one loop. OP may be all-NaN (not recorded).

    Pass `period` (in samples, from `oscillation_regularity`) to also check that
    the archive is fine enough to read a waveform shape - Week 6."""
    cfg = cfg or DataQualityConfig()
    sp, pv, op = (np.asarray(v, dtype=float) for v in (sp, pv, op))
    flags, notes = [], []
    has_op = np.isfinite(op).all() and len(op) == len(pv)

    pv_run, pv_start = longest_flat_run(pv)
    lv_pv = n_levels(pv)
    if lv_pv < cfg.pv_levels_min:
        flags.append("quantised_pv")
        notes.append(f"PV takes only {lv_pv} distinct values - PV shape fingerprints not trusted")

    op_run = lv_op = sat = r2 = kc = None
    if not has_op:
        flags.append("no_op")
        notes.append("OP not recorded - controller cannot be assessed")
    else:
        lv_op = n_levels(op)
        op_run, _ = longest_flat_run(op)
        r2, kc = pi_fit(sp, pv, op)
        if lv_op == 1:
            flags.append("op_constant")
            notes.append("OP never moves - loop in manual, or OP not truly recorded")
        elif lv_op < cfg.op_levels_min:
            flags.append("op_coarse")
            notes.append(f"OP recorded with only {lv_op} distinct values - "
                         "OP-based checks (manual, saturation) skipped")
        else:
            if op_run > cfg.op_flat_run:
                flags.append("op_inactive")
                notes.append(f"OP flat for {op_run} samples - manual or controller inactive")
            sat = saturation_fraction(op, cfg.saturation_min_run)
            if sat > cfg.saturation_fraction:
                flags.append("saturated")
                notes.append(f"OP at its limit {100 * sat:.0f}% of the time - valve or load problem")
        if pv_run > cfg.pv_flat_run and lv_op > 1:
            seg = op[pv_start:pv_start + pv_run + 1]
            if np.ptp(seg) > 0:
                flags.append("frozen_sensor")
                notes.append(f"PV frozen for {pv_run} samples while OP moved - URGENT: check the transmitter")
        if lv_op > 1 and r2 < cfg.pi_r2_min:
            notes.append(f"PI law not confirmed (R2 = {r2:.2f}) - note only")

    if period is not None and np.isfinite(period):
        per_cycle = samples_per_cycle(period)
        if per_cycle and per_cycle < cfg.min_samples_per_cycle:
            flags.append("coarse_archive")
            notes.append(f"only {per_cycle:.1f} samples per oscillation cycle - the archive is "
                         "too coarse to read the waveform shape (Week 6)")

    n_ch, frac_ch, seg = sp_segments(sp)
    if frac_ch > cfg.moving_sp_fraction:
        flags.append("moving_setpoint")
        notes.append("SP changes in most samples - likely a cascade slave")
    elif n_ch and seg.stop - seg.start < cfg.min_segment:
        flags.append("short_regulatory_segment")
    elif n_ch:
        notes.append(f"SP changes {n_ch} times - analyse samples {seg.start}-{seg.stop - 1}")

    return QualityReport(n=len(pv), pv_flat_run=pv_run, op_flat_run=op_run, pv_levels=lv_pv,
                         op_levels=lv_op, saturation=sat, sp_changes=n_ch,
                         sp_change_fraction=frac_ch, regulatory_segment=seg,
                         pi_r2=r2, kc_estimate=kc, flags=flags, notes=notes)
