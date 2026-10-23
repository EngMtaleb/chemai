"""A seawater RO plant whose fouling we know - including what it would have done
had nobody cleaned it.

Why simulate when a real deposit exists: the deposit holds three trains and
eighteen cleanings, and for every cycle the operator's cleaning cut the
trajectory short. A forecast of "when will it reach the review level" can only
be scored against the path that was never allowed to happen. The simulator
keeps the true fouling state, so that path is known.

Model, per train and per day (train-level, lumped solution-diffusion):

    permeate held at its set point; feed pressure is whatever that takes
    NDP = J / (A * TCF(T)),  A falling slowly with fouling and compaction
    P_feed = NDP + dP/2 + P_perm + (pi_feed-conc - pi_perm)
    dP = dP_clean * (Q_avg/Q_design)^(2-m) * (mu(T)/mu_25)^m * G(phi)
    G(phi) = f (1-phi)^-n + (1-f)(1-k phi)^-n      lead element blocks first

phi is the blocked gap fraction of the lead element. It grows by one of three
laws - steady deposition (linear), biofilm (exponential, then capped), or
deposition balanced by removal (Kern-Seaton, asymptotic) - faster in warm
water, with upsets. A cleaning removes part of the removable layer, leaves a
little for good, and loosens some that flushes out over the next days.

An operator watches the normalised dP on the screen and cleans when it looks
high, after a delay, or when too long has passed. Their decisions are
therefore endogenous to the very signal a forecast reads - as in a real plant.

What it cannot represent: element-by-element flow redistribution, scaling of
the tail elements, membrane damage, and any fouling that is not hydraulic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from chemai.config import ROConfig, ROSimConfig
from chemai.features.ro_physics import (log_mean_concentration_factor, osmotic_pressure, tcf,
                                        water_viscosity)

log = logging.getLogger(__name__)

HORIZON_PAD = 400        # days simulated past the record for counterfactual paths
PHI_MAX = 0.85


@dataclass
class TrainParams:
    name: str
    area_m2: float
    q_perm_design: float          # m3/h
    recovery: float
    a_w: float                    # LMH/bar at 25 C
    b_s: float                    # LMH at 25 C
    polarisation: float
    dp_clean: float               # bar, design flow, 25 C
    friction_m: float
    gap_exponent: float
    lead_share: float
    rest_ratio: float
    permeability_loss: float
    compaction: float
    # fouling
    law: str
    linear_rate: float
    biofilm_k: float
    biofilm_seed: float
    biofilm_cap: float
    asym_level: float
    asym_tau: float
    q10: float
    # cleaning
    cip_efficiency: float
    irreversible_share: float
    relax_share: float
    relax_days: float
    # measurement
    dp_noise_rel: float


@dataclass
class PlantParams:
    seed: int
    law: str
    temp_mean: float
    temp_amp: float
    temp_phase: float
    feed_tds: float
    operator_level: float
    operator_delay: tuple[int, int]
    max_days_without_cip: int
    trains: list[TrainParams]


@dataclass
class PlantRun:
    """One simulated plant: daily rows for every train, cleaning events, and
    the state needed to replay any day forward without cleaning."""
    params: PlantParams
    daily: pd.DataFrame = field(repr=False)
    cleanings: pd.DataFrame = field(repr=False)
    # per train: arrays over days + HORIZON_PAD
    rate_mult: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    upset: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    state: dict[str, np.ndarray] = field(repr=False, default_factory=dict)   # (days, 3): rev, irr, relax
    baseline_g: dict[str, float] = field(default_factory=dict)              # G(phi) over the baseline window

    def train(self, name: str) -> pd.DataFrame:
        return self.daily[self.daily["train"] == name]


# ------------------------------------------------------------------ sampling

def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def sample_plant(rng: np.random.Generator, cfg: ROSimConfig | None = None,
                 law: str | None = None) -> PlantParams:
    """Draw one plant. Trains share the intake water and the fouling law; their
    rates and cleaning responses differ by ~20 %, as trains on one intake do."""
    cfg = cfg or ROSimConfig()
    law = law or str(rng.choice(cfg.laws))
    if law not in cfg.laws:
        raise ValueError(f"law must be one of {cfg.laws}")
    area = cfg.vessels * cfg.elements * cfg.element_area_m2
    flux = _u(rng, cfg.flux_lmh)
    base = dict(
        linear_rate=_u(rng, cfg.linear_rate), biofilm_k=_u(rng, cfg.biofilm_k),
        biofilm_seed=_u(rng, cfg.biofilm_seed), biofilm_cap=_u(rng, cfg.biofilm_cap),
        asym_level=_u(rng, cfg.asym_level), asym_tau=_u(rng, cfg.asym_tau_days),
        q10=_u(rng, cfg.q10), cip_efficiency=_u(rng, cfg.cip_efficiency),
        gap_exponent=_u(rng, cfg.gap_exponent), friction_m=_u(rng, cfg.friction_m))
    trains = []
    for k in range(cfg.trains):
        jit = lambda v, s=0.2: float(v * rng.uniform(1 - s, 1 + s))          # noqa: E731
        trains.append(TrainParams(
            name="ABCDEFGH"[k], area_m2=area, q_perm_design=flux * area / 1000.0,
            recovery=_u(rng, cfg.recovery), a_w=_u(rng, cfg.a_w_lmh_bar), b_s=_u(rng, cfg.b_s_lmh),
            polarisation=_u(rng, cfg.conc_polarisation), dp_clean=_u(rng, cfg.dp_clean_bar),
            friction_m=base["friction_m"], gap_exponent=base["gap_exponent"],
            lead_share=_u(rng, cfg.lead_share), rest_ratio=_u(rng, cfg.rest_fouling_ratio),
            permeability_loss=_u(rng, cfg.permeability_loss), compaction=_u(rng, cfg.compaction),
            law=law, linear_rate=jit(base["linear_rate"]), biofilm_k=jit(base["biofilm_k"]),
            biofilm_seed=jit(base["biofilm_seed"]), biofilm_cap=jit(base["biofilm_cap"], 0.1),
            asym_level=jit(base["asym_level"]), asym_tau=jit(base["asym_tau"]),
            q10=base["q10"], cip_efficiency=min(0.97, jit(base["cip_efficiency"], 0.1)),
            irreversible_share=_u(rng, cfg.irreversible_share),
            relax_share=_u(rng, cfg.relax_share), relax_days=_u(rng, cfg.relax_days),
            dp_noise_rel=_u(rng, cfg.dp_noise_rel)))
    d0, d1 = cfg.operator_delay_days
    return PlantParams(
        seed=int(rng.integers(0, 2**31 - 1)), law=law,
        temp_mean=_u(rng, cfg.temp_mean_c), temp_amp=_u(rng, cfg.temp_amp_c),
        temp_phase=float(rng.uniform(0, 365)), feed_tds=_u(rng, cfg.feed_tds_g_l),
        operator_level=_u(rng, cfg.operator_level), operator_delay=(int(d0), int(d1)),
        max_days_without_cip=int(rng.integers(*cfg.max_days_without_cip)), trains=trains)


# ------------------------------------------------------------------ fouling

def gap_law(phi, p: TrainParams):
    """G(phi): pressure drop over its clean value at reference flow and temperature."""
    phi = np.minimum(np.asarray(phi, float), PHI_MAX)
    n, f = p.gap_exponent, p.lead_share
    return f * (1 - phi) ** -n + (1 - f) * (1 - p.rest_ratio * phi) ** -n


def _grow(rev: float, total: float, p: TrainParams, mult: float) -> float:
    """One day of growth of the removable layer."""
    if p.law == "linear":
        d = p.linear_rate * mult
    elif p.law == "biofilm":
        d = p.biofilm_k * mult * (rev + 0.25 * p.biofilm_seed) * max(0.0, 1.0 - total / p.biofilm_cap)
    elif p.law == "asymptotic":
        d = mult * (p.asym_level - rev) / p.asym_tau
    else:
        raise ValueError(f"unknown fouling law '{p.law}'")
    return d


def step_state(s: np.ndarray, p: TrainParams, mult: float, upset: float) -> np.ndarray:
    """Advance (rev, irr, relax) by one day without cleaning."""
    rev, irr, relax = s
    total = rev + irr + relax
    rev = max(0.0, rev + _grow(rev, total, p, mult) + upset)
    relax = relax * np.exp(-1.0 / p.relax_days)
    out = np.array([rev, irr, relax])
    t = out.sum()
    if t > PHI_MAX:
        out[0] -= t - PHI_MAX
    return out


def clean(s: np.ndarray, p: TrainParams, efficiency: float) -> np.ndarray:
    """One cleaning leg: remove part of the removable layer, fix a little for
    good, loosen some that will flush out over the next days."""
    rev, irr, relax = s
    remaining = (1.0 - efficiency) * rev
    irr = irr + p.irreversible_share * remaining
    remaining *= (1.0 - p.irreversible_share)
    relax = relax + p.relax_share * remaining
    rev = remaining * (1.0 - p.relax_share)
    return np.array([rev, irr, relax])


def _temperature(days: np.ndarray, pp: PlantParams, rng) -> np.ndarray:
    return pp.temp_mean + pp.temp_amp * np.sin(2 * np.pi * (days - pp.temp_phase) / 365.0) \
        + rng.normal(0, 0.5, len(days))


def _availability(n_days: int, n_trains: int, cfg: ROSimConfig, rng) -> np.ndarray:
    on = np.zeros((n_trains, n_days), bool)
    for k in range(n_trains):
        d, state = 0, rng.random() < 0.7
        while d < n_days:
            mean = cfg.on_block_days if state else cfg.off_block_days
            L = max(1, int(rng.exponential(mean)))
            on[k, d:d + L] = state
            d += L
            state = not state
    return on


# ------------------------------------------------------------------ the plant

def simulate_plant(pp: PlantParams, cfg: ROSimConfig | None = None,
                   ro: ROConfig | None = None, operator: bool = True) -> PlantRun:
    """Simulate one plant for cfg.days days. Deterministic for a given pp.seed."""
    cfg, ro = cfg or ROSimConfig(), ro or ROConfig()
    rng = np.random.default_rng(pp.seed)
    n_days, n_tr = cfg.days, len(pp.trains)
    ext = n_days + HORIZON_PAD
    days_ext = np.arange(ext)
    temp_ext = _temperature(days_ext, pp, rng)
    temp = temp_ext[:n_days]
    tds = pp.feed_tds + 0.4 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0) \
        + rng.normal(0, 0.15, n_days)
    demand = 2.0 * (1 + 0.06 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0)
                    + rng.normal(0, 0.03, n_days))
    on = _availability(n_days, n_tr, cfg, rng)
    mu25 = water_viscosity(25.0)

    run = PlantRun(params=pp, daily=pd.DataFrame(), cleanings=pd.DataFrame())
    for p in pp.trains:
        noise = np.exp(rng.normal(0, cfg.rate_noise, ext) - cfg.rate_noise ** 2 / 2)
        run.rate_mult[p.name] = p.q10 ** ((temp_ext - 25.0) / 10.0) * noise
        up = np.zeros(ext)
        n_up = rng.poisson(cfg.upsets_per_year * ext / 365.0)
        for d in rng.integers(0, ext, n_up):
            L = int(rng.integers(3, 11))
            up[d:d + L] += _u(rng, cfg.upset_size) / L
        run.upset[p.name] = up
        run.state[p.name] = np.zeros((n_days, 3))

    rows, events = [], []
    drift = {p.name: np.cumsum(rng.normal(0, cfg.pt_drift_bar_per_day, (2, n_days)), axis=1)
             for p in pp.trains}
    st = {p.name: np.array([0.0, 0.0, 0.0]) for p in pp.trains}
    if pp.law == "biofilm":                   # a young biofilm is present from day one
        for p in pp.trains:
            st[p.name][0] = p.biofilm_seed * 0.3
    pending = {p.name: {} for p in pp.trains}     # scheduled cleaning day -> reason
    last_cip = {p.name: -10_000 for p in pp.trains}
    seen = {p.name: [] for p in pp.trains}        # operator's screen: recent DP_norm values
    base_ref = {p.name: [] for p in pp.trains}
    q_ref = {p.name: None for p in pp.trains}

    for d in range(n_days):
        n_on = int(on[:, d].sum())
        for k, p in enumerate(pp.trains):
            s = step_state(st[p.name], p, float(run.rate_mult[p.name][d]), float(run.upset[p.name][d]))
            cip_today = d in pending[p.name]
            if cip_today:
                eff = p.cip_efficiency * (rng.uniform(0.85, 1.0))
                s = clean(s, p, eff)
                legs = 2 if rng.random() < cfg.two_leg_probability else 1
                if legs == 2:
                    s = clean(s, p, eff * 0.5)
                events.append(dict(train=p.name, day=d, legs=legs, reason=pending[p.name].pop(d)))
                last_cip[p.name] = d
                seen[p.name] = []
            st[p.name] = s
            run.state[p.name][d] = s
            phi = float(s.sum())
            g = float(gap_law(phi, p))

            running = bool(on[k, d]) and not cip_today
            if not running:
                rows.append(dict(train=p.name, day=d, running=False, recorded=False, phi=phi, g=g,
                                 cip=cip_today))
                continue
            # operating point
            share = np.clip(demand[d] / max(n_on, 1), 0.75, 1.10)
            q_p = p.q_perm_design * share
            rec = float(np.clip(p.recovery + rng.normal(0, 0.005), 0.3, 0.6))
            q_f = q_p / rec
            q_c = q_f - q_p
            q_avg = (q_f + q_c) / 2.0
            q_avg_design = (p.q_perm_design / p.recovery + p.q_perm_design / p.recovery - p.q_perm_design) / 2.0
            T = float(temp[d])
            mu = float(water_viscosity(T))
            dp = p.dp_clean * (q_avg / q_avg_design) ** (2 - p.friction_m) * (mu / mu25) ** p.friction_m * g
            flux = q_p * 1000.0 / p.area_m2
            a_eff = p.a_w * (1 - p.compaction * (1 - np.exp(-d / 90.0))) / (1 + p.permeability_loss * phi)
            ndp = flux / (a_eff * float(tcf(T, ro.tcf_e_water)))
            cf = float(log_mean_concentration_factor(rec))
            pol = p.polarisation * (1 + 0.3 * phi)
            c_fc = tds[d] * cf * pol
            c_p = p.b_s * float(tcf(T, ro.tcf_e_salt)) * (1 + 0.1 * d / 365.0) * c_fc / flux
            pi_fc = float(osmotic_pressure(c_fc, T))
            pi_p = float(osmotic_pressure(c_p, T))
            p_perm = cfg.perm_pressure_bar
            p_feed = ndp + dp / 2.0 + p_perm + (pi_fc - pi_p)
            p_conc = p_feed - dp
            recorded = n_on >= 2
            # measurement: daily medians
            out = rng.random() < cfg.outlier_probability
            sc = cfg.outlier_scale if out else 1.0
            dp_dpt = dp * (1 + rng.normal(0, p.dp_noise_rel * sc))
            pf_m = p_feed + rng.normal(0, cfg.pt_noise_bar * sc) + drift[p.name][0, d]
            pc_m = p_conc + rng.normal(0, cfg.pt_noise_bar * sc) + drift[p.name][1, d]
            row = dict(train=p.name, day=d, running=True, recorded=recorded, n_on=n_on, phi=phi, g=g,
                       cip=False, temperature_c=T + rng.normal(0, 0.2), true_temperature=T,
                       feed_flow=q_f * (1 + rng.normal(0, 0.005)), conc_flow=q_c * (1 + rng.normal(0, 0.005)),
                       perm_flow=q_p * (1 + rng.normal(0, 0.005)), feed_pressure=pf_m, conc_pressure=pc_m,
                       perm_pressure=p_perm + rng.normal(0, 0.02), feed_tds=tds[d] * (1 + rng.normal(0, 0.01)),
                       perm_tds=c_p * (1 + rng.normal(0, 0.02)), dp=dp_dpt, true_dp=dp,
                       true_feed_pressure=p_feed, true_q_avg=q_avg)
            rows.append(row)
            # the operator's screen: vendor-style normalisation against start-up
            if recorded:
                if q_ref[p.name] is None:
                    q_ref[p.name] = q_avg
                dp_ref = dp_dpt * (q_ref[p.name] / q_avg) ** ro.vendor_flow_exponent \
                    * (mu25 / mu) ** ro.dp_visc_exponent
                if len(base_ref[p.name]) < cfg.baseline_days:
                    base_ref[p.name].append(dp_ref)
                    run.baseline_g.setdefault(p.name + "_acc", []).append(g)
                else:
                    seen[p.name] = (seen[p.name] + [dp_ref / np.median(base_ref[p.name])])[-7:]
            if not operator or len(base_ref[p.name]) < cfg.baseline_days:
                continue
            since = d - last_cip[p.name]
            if pending[p.name] or since < cfg.min_days_between_cip:
                continue
            look = np.median(seen[p.name]) if len(seen[p.name]) >= 3 else 0.0
            if look >= pp.operator_level:
                delay = int(rng.integers(pp.operator_delay[0], pp.operator_delay[1] + 1))
                pending[p.name] = {d + delay + 1: "dP on screen"}
            elif since >= pp.max_days_without_cip:
                pending[p.name] = {d + 1: "time since last cleaning"}
            elif rng.random() < cfg.other_reason_cip_per_year / 365.0:
                pending[p.name] = {d + 1: "other reason"}

    daily = pd.DataFrame(rows)
    for p in pp.trains:
        acc = run.baseline_g.pop(p.name + "_acc", [1.0])
        run.baseline_g[p.name] = float(np.median(acc))
    daily["dp_true"] = daily["g"] / daily["train"].map(run.baseline_g)
    run.daily = daily
    run.cleanings = pd.DataFrame(events, columns=["train", "day", "legs", "reason"])
    return run


def counterfactual(run: PlantRun, train: str, day: int, horizon: int) -> np.ndarray:
    """True DP_norm (plant scale) for days day+1 .. day+horizon had nobody cleaned.

    Uses the same temperatures, rate noise and upsets the plant would have seen,
    so the only thing removed is the operator's decision.
    """
    p = next(t for t in run.params.trains if t.name == train)
    mult, up = run.rate_mult[train], run.upset[train]
    s = run.state[train][day].copy()
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        i = min(day + h, len(mult) - 1)
        s = step_state(s, p, float(mult[i]), float(up[i]))
        out[h - 1] = float(gap_law(s.sum(), p)) / run.baseline_g[train]
    return out


def generate_plants(n: int, seed: int = 42, cfg: ROSimConfig | None = None,
                    laws: tuple[str, ...] | None = None) -> list[PlantRun]:
    """n plants with the fouling laws cycled evenly, so no law is under-represented."""
    cfg = cfg or ROSimConfig()
    laws = laws or cfg.laws
    rng = np.random.default_rng(seed)
    return [simulate_plant(sample_plant(rng, cfg, laws[i % len(laws)]), cfg) for i in range(n)]
