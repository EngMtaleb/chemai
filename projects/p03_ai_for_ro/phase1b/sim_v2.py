"""Phase 3b - the simulator, version 2, as declared in declaration.md.

Project 3's simulator (chemai/data/ro_sim.py) is imported and never changed. Version 2 copies its daily loop and adds,
behind switches (and a fourth, below):
  1. real_temperature  feed temperature = a window of the real daily sea-temperature series (sensor noise as in the base)
  2. driver            a plant-wide fouling driver Z(t) multiplies the fouling rate by exp(sigma_z*Z - sigma_z^2/2)
  3. flux_factor       the fouling rate scales with J/J_design, and is zero on a day the train does not run
  +  clock_from_start  the clock for the longest time without a cleaning starts at day 0; the base starts it at minus
                       10,000 days, so it cleaned every train as soon as its 14-day baseline was in (a change after the
                       declaration, recorded there)
and, after the run (so they can never change the plant):
  4. logged()          operating data (feed temperature, flux, recovery) and two feed-quality instruments (SDI, turbidity)
                       reading L + r*e, with L = sqrt(s)*Z + sqrt(1-s)*V and V an independent twin of Z
  5. deposit_view()    dp_pct built the deposit's way from measured values: normalised dP over its first-14-days baseline,
                       minus 1, clipped 0-1, never reset at a cleaning

The new random draws come from a second stream, so with switches 1-3 off a run is the base simulator's run, draw for draw
(check_sim_v2.py checks it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import ROConfig, ROSimConfig
from chemai.data import ro_sim as B
from chemai.features.ro_physics import log_mean_concentration_factor, osmotic_pressure, tcf, water_viscosity

N_DAYS = 772                       # the deposit's day 0-771


@dataclass(frozen=True)
class V2:
    real_temperature: bool = True
    driver: bool = True
    flux_factor: bool = True
    clock_from_start: bool = True  # the longest-time-without-a-cleaning clock starts at day 0 (the base: at minus 10,000)
    sigma_z: float = 0.5           # set in calibration (step 3); 0.5 is only a placeholder before it
    seasonal_share: float = 0.3    # declared: 30 % yearly cycle, 70 % slow random variation
    tau_days: float = 7.0          # declared: correlation time of the slow variation
    wander_sd: float = 0.0         # attempt 2: slow movement of the ΔP reading that is not fouling (relative); 0 = off
    wander_tau: float = 7.0        # attempt 2: its correlation time, days


@dataclass
class RunV2:
    run: B.PlantRun
    z: np.ndarray                  # the driver, over the record plus the base's pad
    twin: np.ndarray               # V: the same kind of signal, independent of Z (the part instruments see that is not Z)
    e1: np.ndarray                 # daily instrument noise, SDI
    e2: np.ndarray                 # daily instrument noise, turbidity
    temp_start: int | None         # where the temperature window starts in the series
    flux_fac: dict = field(default_factory=dict)      # per train: the flux factor applied each day
    wander: dict = field(default_factory=dict)        # per train: u(t), the reading's slow movement (attempt 2)
    base_mult: dict = field(default_factory=dict)     # per train: q10 term x the base's daily noise


def load_sea_temperature(path) -> np.ndarray:
    """NASA POWER daily file: TS (MERRA-2 earth skin temperature), C, in date order."""
    path = Path(path)
    lines = path.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("-END HEADER-"))
    s = pd.read_csv(path, skiprows=i + 1)
    if (s["TS"] <= -900).any():
        raise ValueError("missing values in the temperature series")
    return s["TS"].to_numpy(float)


def driver_signal(rng: np.random.Generator, n: int, seasonal_share: float, tau_days: float, phase: float) -> np.ndarray:
    """Unit variance: sqrt(w) x a yearly sine (scaled to unit variance) + sqrt(1-w) x first-order noise."""
    t = np.arange(n)
    seasonal = np.sqrt(2.0) * np.sin(2.0 * np.pi * (t - phase) / 365.0)
    a = np.exp(-1.0 / tau_days)
    e = rng.normal(0.0, np.sqrt(1.0 - a * a), n)
    ar = np.empty(n)
    ar[0] = rng.normal()
    for i in range(1, n):
        ar[i] = a * ar[i - 1] + e[i]
    return np.sqrt(seasonal_share) * seasonal + np.sqrt(1.0 - seasonal_share) * ar


def ar1(rng: np.random.Generator, n: int, tau_days: float) -> np.ndarray:
    """A first-order random process with unit variance and correlation time tau_days."""
    a = np.exp(-1.0 / tau_days)
    e = rng.normal(0.0, np.sqrt(1.0 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def simulate(pp: B.PlantParams, cfg: ROSimConfig | None = None, v2: V2 = V2(), sea_t: np.ndarray | None = None,
             ro: ROConfig | None = None, operator: bool = True) -> RunV2:
    """One plant. The base's simulate_plant, line for line, with the three switches."""
    cfg, ro = cfg or ROSimConfig(days=N_DAYS), ro or ROConfig()
    rng = np.random.default_rng(pp.seed)                 # the base stream
    rng2 = np.random.default_rng([pp.seed, 2])           # version 2's own draws, always made in this order
    rng3 = np.random.default_rng([pp.seed, 3])           # attempt 2's draws: a third stream, so attempt 1 is unchanged
    n_days, n_tr = cfg.days, len(pp.trains)
    ext = n_days + B.HORIZON_PAD
    days_ext = np.arange(ext)

    temp_start = int(rng2.integers(0, len(sea_t) - ext + 1)) if sea_t is not None else None
    phase = float(rng2.uniform(0.0, 365.0))
    z = driver_signal(rng2, ext, v2.seasonal_share, v2.tau_days, phase)
    # the twin's yearly cycle runs a quarter-year from Z's: two yearly sines with unrelated phases would be correlated
    # within a plant (by the cosine of their phase gap), and the instruments would then see more or less than sqrt(s) of Z
    twin = driver_signal(rng2, n_days, v2.seasonal_share, v2.tau_days, phase + 365.0 / 4.0)
    e1, e2 = rng2.normal(size=n_days), rng2.normal(size=n_days)
    if v2.real_temperature and sea_t is None:
        raise ValueError("real_temperature needs the sea-temperature series")

    temp_ext = B._temperature(days_ext, pp, rng)         # drawn even when replaced, to keep the base stream in step
    if v2.real_temperature:
        temp_ext = np.asarray(sea_t[temp_start:temp_start + ext], float).copy()
    temp = temp_ext[:n_days]
    tds = pp.feed_tds + 0.4 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0) \
        + rng.normal(0, 0.15, n_days)
    demand = 2.0 * (1 + 0.06 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0)
                    + rng.normal(0, 0.03, n_days))
    on = B._availability(n_days, n_tr, cfg, rng)
    mu25 = water_viscosity(25.0)

    run = B.PlantRun(params=pp, daily=pd.DataFrame(), cleanings=pd.DataFrame())
    out = RunV2(run=run, z=z, twin=twin, e1=e1, e2=e2, temp_start=temp_start)
    for p in pp.trains:
        out.wander[p.name] = v2.wander_sd * ar1(rng3, n_days, v2.wander_tau)
    drv = np.exp(v2.sigma_z * z - v2.sigma_z ** 2 / 2.0) if v2.driver else np.ones(ext)
    for p in pp.trains:
        noise = np.exp(rng.normal(0, cfg.rate_noise, ext) - cfg.rate_noise ** 2 / 2)
        out.base_mult[p.name] = p.q10 ** ((temp_ext - 25.0) / 10.0) * noise
        run.rate_mult[p.name] = out.base_mult[p.name] * drv
        up = np.zeros(ext)
        n_up = rng.poisson(cfg.upsets_per_year * ext / 365.0)
        for d in rng.integers(0, ext, n_up):
            L = int(rng.integers(3, 11))
            up[d:d + L] += B._u(rng, cfg.upset_size) / L
        run.upset[p.name] = up
        run.state[p.name] = np.zeros((n_days, 3))
        out.flux_fac[p.name] = np.ones(n_days)

    rows, events = [], []
    drift = {p.name: np.cumsum(rng.normal(0, cfg.pt_drift_bar_per_day, (2, n_days)), axis=1)
             for p in pp.trains}
    st = {p.name: np.array([0.0, 0.0, 0.0]) for p in pp.trains}
    if pp.law == "biofilm":
        for p in pp.trains:
            st[p.name][0] = p.biofilm_seed * 0.3
    pending = {p.name: {} for p in pp.trains}
    last_cip = {p.name: (0 if v2.clock_from_start else -10_000) for p in pp.trains}
    seen = {p.name: [] for p in pp.trains}
    base_ref = {p.name: [] for p in pp.trains}
    q_ref = {p.name: None for p in pp.trains}

    for d in range(n_days):
        n_on = int(on[:, d].sum())
        share = np.clip(demand[d] / max(n_on, 1), 0.75, 1.10)        # no draw: safe to compute before the state step
        for k, p in enumerate(pp.trains):
            cip_today = d in pending[p.name]
            fac = (float(share) if (on[k, d] and not cip_today) else 0.0) if v2.flux_factor else 1.0
            out.flux_fac[p.name][d] = fac
            s = B.step_state(st[p.name], p, float(run.rate_mult[p.name][d]) * fac, float(run.upset[p.name][d]))
            if cip_today:
                eff = p.cip_efficiency * (rng.uniform(0.85, 1.0))
                s = B.clean(s, p, eff)
                legs = 2 if rng.random() < cfg.two_leg_probability else 1
                if legs == 2:
                    s = B.clean(s, p, eff * 0.5)
                events.append(dict(train=p.name, day=d, legs=legs, reason=pending[p.name].pop(d)))
                last_cip[p.name] = d
                seen[p.name] = []
            st[p.name] = s
            run.state[p.name][d] = s
            phi = float(s.sum())
            g = float(B.gap_law(phi, p))

            running = bool(on[k, d]) and not cip_today
            if not running:
                rows.append(dict(train=p.name, day=d, running=False, recorded=False, phi=phi, g=g,
                                 cip=cip_today))
                continue
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
            outl = rng.random() < cfg.outlier_probability
            sc = cfg.outlier_scale if outl else 1.0
            dp_dpt = dp * (1 + rng.normal(0, p.dp_noise_rel * sc))
            if v2.wander_sd > 0:
                dp_dpt *= 1.0 + float(out.wander[p.name][d])
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
    return out


def deposit_view(rv: RunV2, ro: ROConfig | None = None, baseline_days: int = 14) -> tuple[pd.DataFrame, dict]:
    """What the plant would deposit: one row per recorded train-day, and the cleaning legs per train.

    dp_pct from measured values only: the dP transmitter, the measured flows and the measured temperature, normalised as on
    the operator's screen (flow exponent, viscosity exponent), over the median of the first `baseline_days` recorded days.
    Also the level-1 operating data: feed temperature, permeate flux (LMH) and recovery, as measured."""
    ro = ro or ROConfig()
    d = rv.run.daily
    rec = d[d["recorded"]].copy()
    area = {p.name: p.area_m2 for p in rv.run.params.trains}
    mu25 = water_viscosity(25.0)
    parts = []
    for tr, g in rec.groupby("train", sort=True):
        g = g.sort_values("day")
        q_avg = (g["feed_flow"] + g["conc_flow"]) / 2.0
        dp_ref = g["dp"] * (q_avg.iloc[0] / q_avg) ** ro.vendor_flow_exponent \
            * (mu25 / water_viscosity(g["temperature_c"].to_numpy())) ** ro.dp_visc_exponent
        base = float(np.median(dp_ref.iloc[:baseline_days]))
        parts.append(pd.DataFrame({
            "train": tr, "day_index": g["day"].astype(int).to_numpy(),
            "dp_pct": np.clip(dp_ref.to_numpy() / base - 1.0, 0.0, 1.0),
            "temperature_c": g["temperature_c"].to_numpy(), "flux_lmh": (g["perm_flow"] * 1000.0 / area[tr]).to_numpy(),
            "recovery": (g["perm_flow"] / g["feed_flow"]).to_numpy(), "dp_true_pct": (g["dp_true"] - 1.0).to_numpy()}))
    view = pd.concat(parts, ignore_index=True)
    cl = rv.run.cleanings
    legs = {p.name: sorted(int(x) for x in cl.loc[cl["train"] == p.name, "day"]) for p in rv.run.params.trains}
    return view, legs


def logged(rv: RunV2, s: float, r: float) -> pd.DataFrame:
    """The two feed-quality instruments for coverage s and noise r: one row per day, the same for every train (one intake).
    SDI-like and NTU-like scales are monotone and for readability only."""
    if not (0.0 <= s <= 1.0) or r < 0:
        raise ValueError("s must be in [0, 1] and r >= 0")
    n = len(rv.twin)
    seen = np.sqrt(s) * rv.z[:n] + np.sqrt(1.0 - s) * rv.twin
    return pd.DataFrame({"day_index": np.arange(n),
                         "sdi": 3.0 * np.exp(0.25 * (seen + r * rv.e1)),
                         "turbidity_ntu": 0.5 * np.exp(0.4 * (seen + r * rv.e2))})
