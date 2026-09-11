"""Control loop simulator - PI controller, sticky valve, three process types.

Why simulate when SACAC exists: SACAC has 13 labelled loops, confounded by
source and loop type, and no valve position. Here every quantity is known,
including the one the plant data lacks:

    SP -> PI -> OP -> valve -> MV -> process -> PV
                      ^^^^^^^^^^^
                      stiction lives here; MV is recorded

Every simulated loop is a different plant (parameters drawn once per loop),
so the loop is the natural group for any train/test split.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import logging

import numpy as np
import pandas as pd

from chemai.config import ControlLoopConfig, ProcessRange

log = logging.getLogger(__name__)


# ================================================================ valve

class StictionValve:
    """Two-parameter data-driven stiction model (Choudhury, Thornhill & Shah,
    2005, Control Eng. Practice 13, 641-658).

    S  deadband + stickband, % of travel. After a REVERSAL the command must
       move S before the valve moves.
    J  slip jump, % of travel. After the valve STOPS and resumes in the same
       direction, the command must move J; each slip jumps the stem by J.

    While moving, the stem lags the command by (S - J) / 2.
    S = J = 0 is a perfect valve. J = 0 is pure deadband (backlash).
    """

    def __init__(self, s: float = 0.0, j: float = 0.0, initial: float = 50.0):
        if s < 0 or j < 0 or j > s + 1e-12:
            raise ValueError("require 0 <= J <= S")
        self.s, self.j = float(s), float(j)
        self.pos = float(initial)          # MV
        self._x_prev = float(initial)
        self._x_stick = float(initial)     # command at which the valve stuck
        self._dir = 0                      # last direction of MOVEMENT: +1, -1, 0
        self._stuck = True

    def step(self, x: float) -> float:
        x = float(x)
        if self.s == 0.0:                  # ideal valve
            self.pos, self._x_prev = x, x
            return self.pos
        half = (self.s - self.j) / 2.0

        if not self._stuck:
            v = np.sign(x - self._x_prev)
            if v == self._dir and v != 0:
                self.pos = x - self._dir * half          # keeps moving
            else:
                self._stuck, self._x_stick = True, self._x_prev   # stops
        if self._stuck:
            dx = x - self._x_stick
            reversal = self._dir == 0 or np.sign(dx) != self._dir
            threshold = self.s if reversal else self.j
            if abs(dx) > threshold:                      # slip
                self._dir = int(np.sign(dx))
                self.pos = x - self._dir * half
                self._stuck = False

        self.pos = min(max(self.pos, 0.0), 100.0)
        self._x_prev = x
        return self.pos


# ============================================================ loop spec

@dataclass(frozen=True)
class LoopSpec:
    """One simulated plant + controller + fault. Everything needed to
    regenerate the run exactly."""
    loop_id: str
    loop_type: str            # F, P, L
    condition: str
    integrating: bool
    gain: float
    tau: float | None
    theta: float
    ts: float
    substeps: int
    kc: float
    ti: float
    gain_margin: float
    stiction_s: float = 0.0
    stiction_j: float = 0.0
    ext_period: float = 0.0
    ext_amplitude: float = 0.0
    load_std: float = 0.5
    load_offset: float = 0.0
    load_corr_time: float = 10.0
    meas_std: float = 0.05

    @property
    def slip_to_load(self) -> float:
        """Slip jump over load-noise std. Below ~1 a sticky valve is usually
        buried in the disturbance and produces no regular limit cycle
        (Week 1 finding) - the fault exists but is not visible in OP/PV."""
        return self.stiction_j / self.load_std if self.load_std > 0 else float("inf")

    def to_dict(self) -> dict:
        return {**asdict(self), "slip_to_load": self.slip_to_load}


# ================================================================ tuning

def simc_pi(gain: float, tau: float | None, theta: float, tau_c: float,
            integrating: bool) -> tuple[float, float]:
    """SIMC PI settings (Skogestad, 2003, J. Process Control 13, 291-309).

    Self-regulating:  Kc = tau / (K (tau_c + theta)),  Ti = min(tau, 4 (tau_c + theta))
    Integrating:      Kc = 1 / (K' (tau_c + theta)),   Ti = 4 (tau_c + theta)
    """
    if integrating:
        return 1.0 / (gain * (tau_c + theta)), 4.0 * (tau_c + theta)
    return tau / (gain * (tau_c + theta)), min(tau, 4.0 * (tau_c + theta))


def gain_margin(kc: float, ti: float, gain: float, tau: float | None,
                theta: float, integrating: bool, dt: float) -> float:
    """Gain margin of the loop AS SIMULATED - discrete PI (velocity form),
    zero-order-hold process, delay of round(theta/dt) steps plus one.

    Computed on the discrete loop rather than the continuous one because the
    extra phase lag of sampling is not negligible near instability: a
    continuous gain margin of 1.15 can be below 1 once discretised.
    Phase crossover does not depend on Kc, so GM scales exactly as 1/Kc -
    which is what lets `tuning_tight` be set to a target gain margin.
    """
    nd = int(round(theta / dt))
    w = np.linspace(1e-6, np.pi, 200_000)          # rad per step, up to Nyquist
    zi = np.exp(-1j * w)                           # z^-1
    c = kc * ((1 + dt / ti) - zi) / (1 - zi)
    if integrating:
        g = gain * dt * zi ** (nd + 1) / (1 - zi)
    else:
        a = np.exp(-dt / tau)
        g = gain * (1 - a) * zi ** (nd + 1) / (1 - a * zi)
    loop = c * g
    ph = np.unwrap(np.angle(loop)) + np.pi
    down = np.where((ph[:-1] > 0) & (ph[1:] <= 0))[0]
    if len(down) == 0:
        return float("inf")
    i = down[0]
    f = ph[i] / (ph[i] - ph[i + 1])                # linear interpolation
    return float(1.0 / ((1 - f) * abs(loop[i]) + f * abs(loop[i + 1])))


# ============================================================ sampling

def sample_loop_spec(loop_type: str, condition: str, rng: np.random.Generator,
                     cfg: ControlLoopConfig | None = None, loop_id: str = "") -> LoopSpec:
    """Draw one plant and set the controller and fault for `condition`."""
    cfg = cfg or ControlLoopConfig()
    if condition not in cfg.conditions:
        raise ValueError(f"unknown condition '{condition}'")
    pr: ProcessRange = cfg.process(loop_type)
    u = lambda r: float(rng.uniform(*r))                                      # noqa: E731

    dt = pr.ts / pr.substeps
    gain = u(pr.gain)
    theta = max(1, round(u(pr.theta) / dt)) * dt      # on the simulation grid
    tau = None if pr.integrating else u(pr.tau)
    meas_std = u(cfg.meas_noise_std)

    # healthy speed, limited by noise amplification to the valve
    kc_max = cfg.op_noise_limit / meas_std
    tau_c_noise = (1.0 / (gain * kc_max) if pr.integrating
                   else tau / (gain * kc_max)) - theta
    tau_c = max(theta * u(cfg.healthy_tauc_over_theta), tau_c_noise)
    if condition == "tuning_sluggish":
        tau_c = (theta * u(cfg.sluggish_tauc_over_theta_integrating) if pr.integrating
                 else (tau + theta) * u(cfg.sluggish_tauc_over_open_loop))
    kc, ti = simc_pi(gain, tau, theta, tau_c, pr.integrating)
    gm = gain_margin(kc, ti, gain, tau, theta, pr.integrating, dt)
    if condition == "tuning_tight":
        target = u(cfg.tight_gain_margin)
        kc, gm = kc * gm / target, target

    extra = {}
    if condition == "stiction":
        s = u(cfg.stiction_s)
        extra.update(stiction_s=s, stiction_j=s * u(cfg.stiction_j_over_s))
    if condition == "external_oscillation":
        extra.update(ext_period=(tau_c + theta) * u(cfg.ext_period_over_loop),
                     ext_amplitude=u(cfg.ext_amplitude))

    return LoopSpec(
        loop_id=loop_id or f"{loop_type}-{condition}", loop_type=loop_type,
        condition=condition, integrating=pr.integrating, gain=gain, tau=tau,
        theta=theta, ts=pr.ts, substeps=pr.substeps, kc=float(kc), ti=float(ti),
        gain_margin=float(gm), load_std=u(cfg.load_noise_std),
        load_corr_time=u(cfg.load_noise_corr_time), meas_std=meas_std,
        load_offset=float(rng.choice([-1.0, 1.0])) * u(cfg.load_offset),
        **extra,
    )


# ============================================================ simulation

def simulate_loop(spec: LoopSpec, n_samples: int = 3000, seed: int = 0,
                  warmup_samples: int = 300, cfg: ControlLoopConfig | None = None
                  ) -> pd.DataFrame:
    """Run one loop. Returns columns t, sp, pv, op, mv sampled every `ts`.

    The controller runs `substeps` times per historian sample; the historian
    keeps an instantaneous snapshot (no averaging, no compression - both are
    Week 5 questions). Disturbances enter at the process input, as an
    upstream flow or pressure upset would.
    """
    cfg = cfg or ControlLoopConfig()
    rng = np.random.default_rng(seed)
    dt = spec.ts / spec.substeps
    n_total = (n_samples + warmup_samples) * spec.substeps
    nd = max(0, int(round(spec.theta / dt)))

    u0, sp = cfg.op_nominal, cfg.sp_nominal
    valve = StictionValve(spec.stiction_s, spec.stiction_j, initial=u0)
    delay = deque([u0] * nd, maxlen=nd) if nd else None

    a_load = np.exp(-dt / spec.load_corr_time)
    load_w = rng.normal(0.0, spec.load_std * np.sqrt(1 - a_load ** 2), n_total)
    meas = rng.normal(0.0, spec.meas_std, n_total)
    a_proc = 0.0 if spec.integrating else np.exp(-dt / spec.tau)
    phase0 = rng.uniform(0, 2 * np.pi)

    x = 0.0                                  # process state, deviation
    load = 0.0
    op, e_prev = u0, 0.0
    out = np.empty((n_samples + warmup_samples, 4))

    for k in range(n_total):
        pv = sp + x + meas[k]
        e = sp - pv
        # PI, velocity form: clipping the stored output is the anti-windup
        op = op + spec.kc * ((e - e_prev) + dt / spec.ti * e)
        op = min(max(op, 0.0), 100.0)
        e_prev = e

        mv = valve.step(op)
        if delay is not None:
            u_eff = delay[0]
            delay.append(mv)
        else:
            u_eff = mv

        load = a_load * load + load_w[k]
        d_in = load + spec.load_offset
        if spec.ext_amplitude:
            d_in += spec.ext_amplitude * np.sin(2 * np.pi * k * dt / spec.ext_period + phase0)
        du = (u_eff - u0) + d_in

        if spec.integrating:
            x = x + spec.gain * dt * du
        else:
            x = a_proc * x + (1 - a_proc) * spec.gain * du

        if (k + 1) % spec.substeps == 0:
            out[(k + 1) // spec.substeps - 1] = (sp, pv, op, mv)

    out = out[warmup_samples:]
    df = pd.DataFrame(out, columns=["sp", "pv", "op", "mv"])
    df.insert(0, "t", np.arange(n_samples) * spec.ts)
    return df


def generate_dataset(n_loops_per_cell: int = 6, seeds_per_loop: int = 2,
                     cfg: ControlLoopConfig | None = None,
                     loop_types: tuple[str, ...] = ("F", "P", "L")
                     ) -> tuple[list[LoopSpec], dict[tuple[str, int], pd.DataFrame]]:
    """Full factorial: loop type x condition x plant, with repeated seeds.

    Loop type and condition are crossed by design - the confound found in
    SACAC (all pressure loops are stiction) cannot exist here. Seeds of one
    plant share a loop_id and must stay on the same side of any split.
    """
    cfg = cfg or ControlLoopConfig()
    rng = np.random.default_rng(cfg.random_state)
    specs, runs = [], {}
    for lt in loop_types:
        for cond in cfg.conditions:
            for i in range(n_loops_per_cell):
                spec = sample_loop_spec(lt, cond, rng, cfg, loop_id=f"{lt}-{cond}-{i:02d}")
                specs.append(spec)
                for s in range(seeds_per_loop):
                    seed = int(rng.integers(0, 2**31))
                    runs[(spec.loop_id, s)] = simulate_loop(
                        spec, cfg.n_samples, seed, cfg.warmup_samples, cfg)
    log.info("simulated %d loops, %d runs", len(specs), len(runs))
    return specs, runs
