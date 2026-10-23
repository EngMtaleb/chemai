"""Running a seawater UF -> RO plant on the sun: an hour-by-hour plan for a year.

Part 2 of Project 4 found that the pre-treatment membrane fouls at the same
rate in daylight as at night and does not foul faster after a restart. So a
plant built from modules - each module on or off, always at its design flux -
can follow the sun with no fouling price per cubic metre. What remains is
hardware: how much production capacity, product-water storage and PV a given
solar share needs. That is a linear programme.

    each hour t:   E_t = SEC * p_t = pv_used_t + grid_t,   pv_used_t <= pv_t
                   s_{t+1} = s_t + p_t - demand,           0 <= s_t <= storage
                   0 <= p_t <= capacity                    (modules on at design flux)
    minimise       sum grid_t         (perfect foresight: the best any controller can do)

A simple rule is replayed next to it - what a plant without a forecast would
do - so the gain of planning is measured, not assumed.

Units: demand is 1 m3/day (results scale linearly); PV size in kWp per m3/day.
The weather is synthetic and generic (no site): day length and clear-day
yield vary through the year, and each day draws a cloudiness factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class SolarPlanConfig:
    sec_ro_kwh_m3: float = 3.0           # seawater RO, literature range 2.5-3.5
    sec_uf_kwh_m3: float = 0.12          # pilot: 0.083 kWh/m3 hydraulic (median of 105 runs) / pump eff. 0.7
    clear_day_yield: float = 6.2         # kWh per kWp on a clear day, yearly mean
    seasonal_amp: float = 0.16           # summer/winter swing of the clear-day yield
    day_length_mean: float = 12.0        # h
    day_length_amp: float = 1.6          # h
    # (share of days, cloudiness factor low, high)
    cloud: tuple = ((0.80, 0.85, 1.00), (0.15, 0.50, 0.85), (0.05, 0.15, 0.50))
    seed: int = 2026

    @property
    def sec(self) -> float:
        return self.sec_ro_kwh_m3 + self.sec_uf_kwh_m3


def pv_year(cfg: SolarPlanConfig | None = None, days: int = 365) -> np.ndarray:
    """Hourly PV yield, kWh per kWp, for one synthetic year."""
    cfg = cfg or SolarPlanConfig()
    rng = np.random.default_rng(cfg.seed)
    shares = np.array([c[0] for c in cfg.cloud])
    out = np.zeros(days * 24)
    hours = np.arange(24) + 0.5
    for d in range(days):
        season = np.sin(2 * np.pi * (d - 80) / 365)
        length = cfg.day_length_mean + cfg.day_length_amp * season
        rise = 12 - length / 2
        shape = np.clip(np.sin(np.pi * (hours - rise) / length), 0, None) ** 1.3
        shape /= shape.sum()
        kind = rng.choice(len(shares), p=shares / shares.sum())
        _, lo, hi = cfg.cloud[kind]
        out[d * 24:(d + 1) * 24] = cfg.clear_day_yield * (1 + cfg.seasonal_amp * season) * rng.uniform(lo, hi) * shape
    return out


def tidal_year(days: int = 365, rated_speed: float = 1.1, seed: int = 2026) -> np.ndarray:
    """Hourly output of a tidal-stream turbine, kWh per kW rated, for one year.

    The current follows M2 and S2 (spring and neap from their beat); power goes
    with speed cubed up to the rated speed, then holds. The default rating gives
    a capacity factor of 0.35, typical of tidal-stream turbines. Generic - no site. The
    point it makes does not depend on the site: the moon delivers at night and
    on cloudy days, on a timetable known years ahead.
    """
    t = np.arange(days * 24) + 0.5
    m2, s2 = 360.0 / 28.9841042, 12.0
    v = np.abs(np.cos(2 * np.pi * t / m2) + 0.35 * np.cos(2 * np.pi * t / s2 + 0.7))
    return np.clip(v / rated_speed, 0, 1) ** 3


@dataclass
class PlanResult:
    capacity: float          # production capacity / average demand rate
    storage_h: float         # product-water storage, hours of demand
    pv_kwp: float            # kWp per m3/day of demand
    solar_fraction: float
    grid_kwh_m3: float
    curtailed_share: float   # PV energy not used
    run_hours_per_day: float # module-hours / modules (equivalent full-capacity hours)
    starts_per_day: float    # times production goes from zero to positive
    method: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _summary(p, g, pv, cfg, capacity, storage_h, pv_kwp, method) -> PlanResult:
    """Shares are taken against the energy the DEMAND needs (1 m3/day), so a
    run that ends with a fuller or emptier tank is not flattered."""
    days = len(p) / 24
    demand_kwh = cfg.sec * days
    on = p > 1e-6 * capacity / 24
    starts = int(np.sum(on[1:] & ~on[:-1]) + on[0])
    used = demand_kwh - g.sum()
    return PlanResult(capacity=capacity, storage_h=storage_h, pv_kwp=pv_kwp,
                      solar_fraction=float(1 - g.sum() / demand_kwh),
                      grid_kwh_m3=float(g.sum() / days),
                      curtailed_share=float(1 - used / (pv.sum() * pv_kwp)) if pv_kwp > 0 else 0.0,
                      run_hours_per_day=float(p.sum() / (capacity / 24) / days),
                      starts_per_day=starts / days, method=method)


def plan_lp(pv: np.ndarray, capacity: float, storage_h: float, pv_kwp: float,
            cfg: SolarPlanConfig | None = None, series: bool = False,
            other: np.ndarray | None = None):
    """The least grid energy any controller could achieve (perfect foresight).

    With `series=True` also returns the hourly production, grid energy and
    tank level (for plotting a week). `other` is a second renewable supply in
    kWh per hour (a tidal turbine, say); the solar fraction then reads as the
    renewable fraction."""
    cfg = cfg or SolarPlanConfig()
    n = len(pv)
    d = 1 / 24                          # demand per hour, m3 (1 m3/day)
    pmax, smax = capacity * d, storage_h * d
    # variables: p (n), u = pv used (n), g = grid (n), s = storage at start of hour (n)
    ip, iu, ig, i_s = 0, n, 2 * n, 3 * n
    c = np.zeros(4 * n)
    c[ig:ig + n] = 1.0
    rows, cols, vals, b = [], [], [], []
    r = 0
    for t in range(n):                  # energy: sec * p - u - g = 0
        rows += [r, r, r]; cols += [ip + t, iu + t, ig + t]; vals += [cfg.sec, -1.0, -1.0]; b.append(0.0); r += 1
    for t in range(n):                  # storage: s_{t+1} - s_t - p_t = -d   (cyclic)
        t1 = (t + 1) % n
        rows += [r, r, r]; cols += [i_s + t1, i_s + t, ip + t]; vals += [1.0, -1.0, -1.0]; b.append(-d); r += 1
    a_eq = coo_matrix((vals, (rows, cols)), shape=(r, 4 * n)).tocsr()
    supply = pv * pv_kwp + (other if other is not None else 0.0)
    bounds = ([(0, pmax)] * n + [(0, float(supply[t])) for t in range(n)]
              + [(0, None)] * n + [(0, smax)] * n)
    res = linprog(c, A_eq=a_eq, b_eq=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    out = _summary(x[ip:ip + n], x[ig:ig + n], pv, cfg, capacity, storage_h, pv_kwp, "plan (LP)")
    if series:
        return out, {"production_m3": x[ip:ip + n], "grid_kwh": x[ig:ig + n],
                     "pv_kwh": pv * pv_kwp, "tank_m3": x[i_s:i_s + n]}
    return out


def plan_rule(pv: np.ndarray, capacity: float, storage_h: float, pv_kwp: float,
              cfg: SolarPlanConfig | None = None, reserve_h: float = 2.0) -> PlanResult:
    """A plant without a forecast: produce with whatever sun there is, up to
    capacity and while the tank has room; take grid power only to keep the tank
    above a small reserve."""
    cfg = cfg or SolarPlanConfig()
    n, d = len(pv), 1 / 24
    pmax, smax, reserve = capacity * d, storage_h * d, min(reserve_h * d, storage_h * d)
    s = smax / 2
    p, g = np.zeros(n), np.zeros(n)
    for t in list(range(n)) + list(range(n)):       # first pass warms the tank up; the second counts
        room = smax - s + d
        p_sun = min(pmax, pv[t] * pv_kwp / cfg.sec, room)
        need = max(0.0, reserve + d - (s + p_sun))
        p_grid = min(pmax - p_sun, need, room - p_sun)
        if s + p_sun + p_grid < d:                        # never miss demand
            p_grid = min(pmax - p_sun, d - s - p_sun)
        p[t] = p_sun + p_grid
        g[t] = p_grid * cfg.sec
        s = s + p[t] - d
    return _summary(p, g, pv, cfg, capacity, storage_h, pv_kwp, "rule")


def flux_penalty(capacity_by_flux: float, n_exponent: float) -> float:
    """Fouling per m3, relative to design flux, if the extra capacity came from
    running the same membrane faster instead of adding modules (fouling rate
    ~ flux^n, so per m3 ~ flux^(n-1)). Daylight production runs at `capacity`
    times the design flux."""
    return capacity_by_flux ** (n_exponent - 1)
