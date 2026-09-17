"""Cascade loops - who is the source when master and slave both oscillate?

A cascade is standard in refineries: a slow master (level, temperature) does not
move a valve. Its output becomes the SETPOINT of a fast slave (usually flow),
and the slave moves the valve. The slave absorbs fast upsets before they reach
the master.

Two questions, two answers:

    find_cascades      which loops are connected - from the data alone
    attribute_source   which of the two is faulty - master or slave

Nothing here scores a cascade's performance. A cascade-specific Harris index is
advanced research and is out of scope, stated in the README (VALIDATION.md 13.3).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy.signal import detrend

from chemai.config import ControlLoopConfig
from chemai.features import oscillation_regularity

log = logging.getLogger(__name__)


def find_cascades(setpoints: np.ndarray, outputs: np.ndarray, names: list[str],
                  r_min: float = 0.95, sp_moves_min: float = 0.05) -> pd.DataFrame:
    """Find master/slave pairs among loops recorded at the same time.

    The structure leaves one fingerprint: **the slave's setpoint IS the master's
    output**. Columns of `setpoints` and `outputs` are loops, rows are samples.

    `scale` is the ratio of the two signals' ranges. A true cascade gives about
    1.0 because the signals are the same signal. RATIO CONTROL leaves the same
    correlation but scales the setpoint by a factor, so the pairs are
    CANDIDATES until a plant engineer confirms them (VALIDATION.md 19.1).
    """
    setpoints, outputs = np.asarray(setpoints, float), np.asarray(outputs, float)
    if setpoints.shape != outputs.shape:
        raise ValueError("setpoints and outputs must have the same shape")
    if len(names) != setpoints.shape[1]:
        raise ValueError("one name per loop is required")

    pairs = []
    for slave in range(setpoints.shape[1]):
        sp = setpoints[:, slave]
        if not np.isfinite(sp).all() or np.std(sp) == 0:
            continue
        if np.mean(np.diff(sp) != 0) < sp_moves_min:      # a fixed setpoint has no master
            continue
        for master in range(outputs.shape[1]):
            if master == slave:
                continue
            op = outputs[:, master]
            if not np.isfinite(op).all() or np.std(op) == 0:
                continue
            r = float(np.corrcoef(sp, op)[0, 1])
            if abs(r) > r_min:
                pairs.append({"master": names[master], "slave": names[slave],
                              "correlation": round(r, 4),
                              "scale": round(float(np.ptp(sp) / np.ptp(op)), 3)})
    log.info("checked %d loops, found %d cascade candidates", setpoints.shape[1], len(pairs))
    return pd.DataFrame(pairs, columns=["master", "slave", "correlation", "scale"])


@dataclass(frozen=True)
class SourceVerdict:
    source: str            # "slave", "master" or "none"
    slave_error: float     # regularity of the slave's control error
    slave_setpoint: float  # regularity of the slave's setpoint
    reason: str

    @property
    def owner(self) -> str:
        return {"slave": "maintenance", "master": "control engineering"}.get(self.source, "")


def attribute_source(slave_sp, slave_pv, cfg: ControlLoopConfig | None = None,
                     sp_share: float = 0.5) -> SourceVerdict:
    """Which loop is the source of the oscillation?

    Both loops of a cascade oscillate together - they are closed on each other -
    so the test is not whether the setpoint moves but whether it moves
    REGULARLY:

      * slave error oscillates, setpoint does not  -> the slave cannot follow a
        setpoint it is given: the fault is in the slave, usually its valve;
      * both oscillate with the same regularity    -> the slave is executing a bad
        order faithfully: the fault is in the master.

    Limits (VALIDATION.md 19.2): two levels only; tested against ONE labelled
    real pair; it names the LOOP, not the fault; and a fault in BOTH loops reads
    as 'master', leaving the slave's valve unrepaired.
    """
    cfg = cfg or ControlLoopConfig()
    sp = np.asarray(slave_sp, dtype=float)
    pv = np.asarray(slave_pv, dtype=float)
    if len(sp) != len(pv):
        raise ValueError("setpoint and measurement must be the same length")
    error_reg = oscillation_regularity(detrend(sp - pv))[1]
    sp_reg = oscillation_regularity(detrend(sp))[1]

    if error_reg <= cfg.regularity_threshold:
        return SourceVerdict("none", round(error_reg, 2), round(sp_reg, 2),
                             "no regular oscillation in the slave")
    if sp_reg > cfg.regularity_threshold and sp_reg > sp_share * error_reg:
        return SourceVerdict("master", round(error_reg, 2), round(sp_reg, 2),
                             "the slave is following an oscillating setpoint")
    return SourceVerdict("slave", round(error_reg, 2), round(sp_reg, 2),
                         "the slave cannot follow a setpoint that is not oscillating regularly")


def simulate_cascade(T: float = 4000, dt: float = 0.2, ts: float = 1.0,
                     gain_s: float = 1.0, tau_s: float = 2.0, theta_s: float = 0.5,
                     kc_s: float = 1.2, ti_s: float = 2.0,
                     stiction_s: float = 0.0, slip_s: float = 0.0,
                     gain_m: float = 0.8, tau_m: float = 25.0, theta_m: float = 4.0,
                     kc_m: float = 0.6, ti_m: float = 30.0,
                     load_std: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """A temperature master over a flow slave, with the fault injectable in either.

    `stiction_s`/`slip_s` stick the slave's valve; `kc_m`/`ti_m` detune the master.
    The load disturbance hits the flow, as it does in a plant - which is why the
    slave exists. Returns t, sp_s, pv_s, op_s, sp_m, pv_m, op_m at `ts`.
    """
    from chemai.data.loop_sim import StictionValve

    rng = np.random.default_rng(seed)
    n, every = int(T / dt), max(1, int(round(ts / dt)))
    nd_s, nd_m = int(round(theta_s / dt)), int(round(theta_m / dt))
    valve = StictionValve(stiction_s, slip_s)
    pipe_s, pipe_m = [50.0] * nd_s, [50.0] * nd_m
    a_s, a_m, a_load = np.exp(-dt / tau_s), np.exp(-dt / tau_m), np.exp(-dt / 10.0)
    x_s = x_m = load = 0.0
    sp_m, op_s, op_m, e_s_prev, e_m_prev = 50.0, 50.0, 50.0, 0.0, 0.0
    rows = []
    for k in range(n):
        pv_s = 50 + x_s + 0.05 * rng.normal()
        pv_m = 50 + x_m + 0.05 * rng.normal()
        e_m = sp_m - pv_m                                  # master: temperature
        op_m = min(max(op_m + kc_m * ((e_m - e_m_prev) + dt / ti_m * e_m), 0), 100)
        e_m_prev = e_m
        sp_s = op_m                                        # the master's output IS the slave's SP
        e_s = sp_s - pv_s                                  # slave: flow
        op_s = min(max(op_s + kc_s * ((e_s - e_s_prev) + dt / ti_s * e_s), 0), 100)
        e_s_prev = e_s
        mv = valve.step(op_s)
        pipe_s.append(mv)
        u = pipe_s.pop(0) if nd_s else mv
        load = a_load * load + load_std * np.sqrt(1 - a_load ** 2) * rng.normal()
        x_s = a_s * x_s + (1 - a_s) * gain_s * ((u - 50.0) + load)
        pipe_m.append(50 + x_s)
        u_m = pipe_m.pop(0) if nd_m else 50 + x_s          # the flow feeds the temperature
        x_m = a_m * x_m + (1 - a_m) * gain_m * (u_m - 50.0)
        if k % every == 0:
            rows.append((k * dt, sp_s, pv_s, op_s, sp_m, pv_m, op_m))
    return pd.DataFrame(rows, columns=["t", "sp_s", "pv_s", "op_s", "sp_m", "pv_m", "op_m"])
