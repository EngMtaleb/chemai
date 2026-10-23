"""Is a plant signal locked to the astronomical tide? A test that survives the day.

A process can oscillate every ~12 hours for many reasons - shifts, sunlight,
temperature, a controller. Only the tide keeps the SAME phase against the
astronomical clock across runs months and years apart. The test uses exactly
that, and nothing about the site: no tide gauge, no coordinates, only the
constituent periods (astronomy), with amplitude and phase fitted.

    base model  y ~ run levels + slow trend per episode + time of day (24/12/8/6 h)
                    + temperature + flow + segment length + element pair
    tide block  cos/sin at M2, N2, O1, M4  (S2, K1, P1 sit on 12.00 / 23.93 / 24.07 h
                and cannot be told apart from the day, so they are left out)
    statistic   variance the tide block explains beyond the base model
    null        recompute the tide block with each shift block's clock moved by a whole
                number of weeks (7 to 364 days). Whole weeks keep time of day and day of
                week; they only scramble the moon. The data are never touched.

Frisch-Waugh-Lovell: residualise y and the tide block on the base design once
(QR), then every surrogate costs one projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Constituent periods, hours (speeds in deg/h: M2 28.9841042, N2 28.4397295,
# O1 13.9430356, M4 57.9682084).
CONSTITUENTS: dict[str, float] = {
    "M2": 360.0 / 28.9841042,
    "N2": 360.0 / 28.4397295,
    "O1": 360.0 / 13.9430356,
    "M4": 360.0 / 57.9682084,
}
# Periods no known forcing sits on; a calibrated test must not find them.
PLACEBO: dict[str, float] = {"P11.0": 11.0, "P13.5": 13.5, "P27.0": 27.0, "P5.3": 5.3}
# The sun's clock: the first four harmonics of the day (hours since a local midnight).
DAY_HARMONICS: dict[str, float] = {"D24": 24.0, "D12": 12.0, "D8": 8.0, "D6": 6.0}

EPOCH = pd.Timestamp("2000-01-01")


@dataclass(frozen=True)
class TideTestConfig:
    tod_harmonics: int = 4                  # 24, 12, 8, 6 h
    knot_hours: float = 48.0                # slow trend: cubic spline, a knot every 48 h
    block_gap_hours: float = 72.0           # episodes closer than this shift together
    shift_weeks: int = 52                   # surrogate shifts: +-1..52 whole weeks
    n_surrogates: int = 999
    alpha: float = 0.01                     # primary test
    alpha_confirm: float = 0.05             # out-of-time confirmation
    winsor: tuple[float, float] = (0.01, 0.99)   # within-run winsorising of the response
    seed: int = 20261023
    relevance: float = 0.20                 # 5-95 % range of the tide component / median y


# ------------------------------------------------------------- regressors

def hours_since_epoch(t) -> np.ndarray:
    return np.asarray((pd.DatetimeIndex(t) - EPOCH) / pd.Timedelta(hours=1), float)


def harmonic_block(t_hours: np.ndarray, periods: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for name, p in periods.items():
        w = 2 * np.pi * np.asarray(t_hours, float) / p
        cols += [np.cos(w), np.sin(w)]
        names += [f"{name}_cos", f"{name}_sin"]
    return np.column_stack(cols), names


def time_of_day_block(t, k: int) -> np.ndarray:
    t = pd.DatetimeIndex(t)
    h = t.hour + t.minute / 60 + t.second / 3600
    cols = []
    for j in range(1, k + 1):
        w = 2 * np.pi * j * h / 24
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def episode_spline(t_hours: np.ndarray, episode: np.ndarray, knot_hours: float) -> np.ndarray:
    """Per-episode slow trend without intercept (run levels carry the intercepts).

    Linear under two knot spacings; cubic with truncated-power knots every
    `knot_hours` otherwise. Too stiff to follow a 12-hour wave.
    """
    cols = []
    for e in np.unique(episode):
        sel = episode == e
        x = t_hours[sel] - t_hours[sel].min()
        span = x.max()
        x_d = x / 24.0
        basis = [x_d]
        if span >= 2 * knot_hours:
            basis += [x_d ** 2, x_d ** 3]
            for k in np.arange(knot_hours, span - knot_hours / 2, knot_hours):
                basis.append(np.clip(x_d - k / 24.0, 0, None) ** 3)
        for b in basis:
            c = np.zeros(len(t_hours))
            c[sel] = b - b.mean()
            cols.append(c)
    return np.column_stack(cols) if cols else np.zeros((len(t_hours), 0))


def dummies(labels: np.ndarray, drop_first: bool = False) -> np.ndarray:
    u = np.unique(labels)
    if drop_first:
        u = u[1:]
    return (labels[:, None] == u[None, :]).astype(float)


def shift_blocks(t_hours: np.ndarray, episode: np.ndarray, gap_hours: float) -> np.ndarray:
    """Episodes whose gaps are shorter than `gap_hours` share one surrogate shift."""
    order = pd.DataFrame({"t": t_hours, "e": episode}).groupby("e").t.agg(["min", "max"]).sort_values("min")
    block, last_end, ids = 0, None, {}
    for e, r in order.iterrows():
        if last_end is None or r["min"] - last_end > gap_hours:
            block += 1
        ids[e] = block
        last_end = r["max"] if last_end is None else max(last_end, r["max"])
    return np.array([ids[e] for e in episode])


# ------------------------------------------------------------- the design

@dataclass
class Prepared:
    """Everything fixed for one test: the residualised response and base projection."""
    y: np.ndarray
    t_hours: np.ndarray
    episode: np.ndarray
    block: np.ndarray
    q: np.ndarray                   # orthonormal basis of the base design
    y_res: np.ndarray
    run: np.ndarray
    n_base: int
    extra: dict = field(default_factory=dict)

    def residualise(self, x: np.ndarray) -> np.ndarray:
        return x - self.q @ (self.q.T @ x)


def winsorise_within(y: np.ndarray, group: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = y.astype(float).copy()
    for g in np.unique(group):
        sel = group == g
        a, b = np.quantile(out[sel], [lo, hi])
        out[sel] = np.clip(out[sel], a, b)
    return out


def prepare(seg: pd.DataFrame, response: str, cfg: TideTestConfig | None = None,
            time_col: str = "t_std", covariates: bool = True, tod: bool = True,
            temperature: bool = True, trend: bool = True) -> Prepared:
    """Build the base design and residualise the response on it.

    Defaults are the tide test's declared base model. The sun test drops the
    time-of-day block (it is what is tested) and, in its primary form, the
    temperature term (DECLARED_BEFORE_RUN_2.md explains why).
    """
    cfg = cfg or TideTestConfig()
    seg = seg.dropna(subset=[response]).reset_index(drop=True)
    run = seg["run"].to_numpy()
    episode = seg["episode"].to_numpy()
    t_hours = hours_since_epoch(seg[time_col])
    y = winsorise_within(seg[response].to_numpy(float), run, *cfg.winsor)

    parts = [dummies(run)]
    if trend:
        parts.append(episode_spline(t_hours, episode, cfg.knot_hours))
    if tod:
        parts.append(time_of_day_block(seg[time_col], cfg.tod_harmonics))
    if covariates:
        dev = lambda v: v - pd.Series(v).groupby(run).transform("mean").to_numpy()
        if temperature:
            parts.append(dev(seg["temp_c"].to_numpy(float))[:, None])
        parts += [dev(np.log(seg["q_gpm"].to_numpy(float)))[:, None],
                  np.log(seg["minutes"].to_numpy(float))[:, None],
                  dummies(seg["config"].to_numpy(), drop_first=True)]
    x0 = np.column_stack(parts)
    x0 = x0[:, np.linalg.norm(x0, axis=0) > 0]
    q, r = np.linalg.qr(x0)
    keep = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
    q = q[:, keep]
    y_res = y - q @ (q.T @ y)
    block = shift_blocks(t_hours, episode, cfg.block_gap_hours)
    return Prepared(y=y, t_hours=t_hours, episode=episode, block=block, q=q, y_res=y_res,
                    run=run, n_base=int(keep.sum()))


def explained(prep: Prepared, xt: np.ndarray) -> tuple[float, np.ndarray]:
    """Sum of squares the (residualised) block explains, and its coefficients."""
    xr = prep.residualise(xt)
    beta, *_ = np.linalg.lstsq(xr, prep.y_res, rcond=None)
    fit = xr @ beta
    return float(fit @ fit), beta


# ------------------------------------------------------------- the test
#
# A whole-week shift of one block's clock rotates that block's cos/sin pair by
# a fixed angle per constituent. Residualising is linear, so with
# Z_k = resid(block k's harmonic columns) every surrogate is an exact
# combination  T~ = sum_k Z_k R_k, and only the Gram matrix of the Z_k is
# needed. The base projection is paid once, not once per surrogate.

@dataclass
class TideDesign:
    periods: dict
    blocks: np.ndarray            # block label per column group
    zcat: np.ndarray              # n x (K * 2C): residualised, block-masked harmonics
    gram: np.ndarray              # (K*2C) x (K*2C)

    @property
    def width(self) -> int:
        return 2 * len(self.periods)


def tide_design(prep: Prepared, periods: dict[str, float]) -> TideDesign:
    x, _ = harmonic_block(prep.t_hours, periods)
    blocks = np.unique(prep.block)
    cols = []
    for b in blocks:
        cols.append(prep.residualise(x * (prep.block == b)[:, None]))
    zcat = np.concatenate(cols, axis=1)
    return TideDesign(periods=periods, blocks=blocks, zcat=zcat, gram=zcat.T @ zcat)


def rotation_stack(design: TideDesign, shift_hours_per_block: np.ndarray) -> np.ndarray:
    """(K*2C) x 2C: the rotation of each block's pairs, stacked."""
    c = design.width
    out = np.zeros((len(design.blocks) * c, c))
    for k, dt in enumerate(shift_hours_per_block):
        for j, p in enumerate(design.periods.values()):
            th = 2 * np.pi * dt / p
            co, si = np.cos(th), np.sin(th)
            r = k * c + 2 * j
            out[r:r + 2, 2 * j:2 * j + 2] = [[co, si], [-si, co]]
    return out


def _fit(design: TideDesign, zy: np.ndarray, m: np.ndarray) -> tuple[float, np.ndarray]:
    g = m.T @ design.gram @ m
    cr = m.T @ zy
    beta = np.linalg.lstsq(g, cr, rcond=None)[0]
    return float(cr @ beta), beta


def block_shifts(n_blocks: int, rng: np.random.Generator, weeks: int) -> np.ndarray:
    """Whole weeks: keep the day and the week, scramble the moon (the tide test)."""
    k = rng.integers(1, weeks + 1, n_blocks) * rng.choice([-1, 1], n_blocks)
    return k * 7 * 24.0


def hour_shifts(n_blocks: int, rng: np.random.Generator, weeks: int = 0) -> np.ndarray:
    """1 to 23 hours: scramble the time of day (the sun test). `weeks` is ignored."""
    return rng.uniform(1.0, 23.0, n_blocks)


SHIFTS = {"weeks": block_shifts, "hours": hour_shifts}


@dataclass
class TideResult:
    response: str
    n: int
    n_runs: int
    n_episodes: int
    n_blocks: int
    partial_r2: float
    p_value: float
    null_partial_r2_q95: float
    amplitudes: dict
    phases_deg: dict
    range_5_95: float
    range_null_q95: float
    median_y: float
    relative_range: float
    beta: list

    def as_dict(self) -> dict:
        return dict(self.__dict__)


_SYNTH_T = np.arange(0, 4 * 8766, 1.0) + 12 * 8766          # 2012-2015, hours since 2000


def component_range(beta: np.ndarray, periods: dict[str, float]) -> float:
    """5-95 % range of the fitted tide component over four years, hourly."""
    x, _ = harmonic_block(_SYNTH_T, periods)
    a, b = np.quantile(x @ beta, [0.05, 0.95])
    return float(b - a)


def tide_test(prep: Prepared, periods: dict[str, float] | None = None,
              cfg: TideTestConfig | None = None, response: str = "y",
              design: TideDesign | None = None, with_ranges: bool = True,
              shift: str = "weeks") -> TideResult:
    cfg = cfg or TideTestConfig()
    shifter = SHIFTS[shift]
    periods = periods or CONSTITUENTS
    design = design or tide_design(prep, periods)
    zy = design.zcat.T @ prep.y_res
    k = len(design.blocks)
    ss_obs, beta = _fit(design, zy, rotation_stack(design, np.zeros(k)))
    ss_tot = float(prep.y_res @ prep.y_res)
    rng = np.random.default_rng(cfg.seed)
    null_ss, null_rng = np.empty(cfg.n_surrogates), np.full(cfg.n_surrogates, np.nan)
    synth = harmonic_block(_SYNTH_T, periods)[0] if with_ranges else None
    for b in range(cfg.n_surrogates):
        m = rotation_stack(design, shifter(k, rng, cfg.shift_weeks))
        null_ss[b], beta_b = _fit(design, zy, m)
        if with_ranges:
            lo, hi = np.quantile(synth @ beta_b, [0.05, 0.95])
            null_rng[b] = hi - lo
    p = (1 + np.sum(null_ss >= ss_obs)) / (1 + cfg.n_surrogates)
    amps, phases = {}, {}
    for i, name in enumerate(periods):
        a, s = beta[2 * i], beta[2 * i + 1]
        amps[name] = float(np.hypot(a, s))
        phases[name] = float(np.degrees(np.arctan2(s, a)) % 360)     # y = A cos(wt - phase)
    rng_obs = component_range(beta, periods)
    med = float(np.median(prep.y))
    return TideResult(response=response, n=len(prep.y), n_runs=len(np.unique(prep.run)),
                      n_episodes=len(np.unique(prep.episode)), n_blocks=k,
                      partial_r2=ss_obs / ss_tot, p_value=float(p),
                      null_partial_r2_q95=float(np.quantile(null_ss, 0.95) / ss_tot),
                      amplitudes=amps, phases_deg=phases, range_5_95=rng_obs,
                      range_null_q95=float(np.nanquantile(null_rng, 0.95)) if with_ranges else float("nan"),
                      median_y=med, relative_range=rng_obs / abs(med) if med else float("nan"),
                      beta=[float(v) for v in beta])


def confirm_out_of_sample(fit_prep: Prepared, test_prep: Prepared, periods: dict[str, float] | None = None,
                          cfg: TideTestConfig | None = None, shift: str = "weeks") -> dict:
    """Freeze the tide component fitted on one set of runs; does it explain another?

    Statistic: the share of the test runs' residual variance removed by the
    frozen component (negative if it makes things worse). Null: the same
    frozen component, with the test runs' clocks shifted by whole weeks.
    """
    cfg = cfg or TideTestConfig()
    periods = periods or CONSTITUENTS
    fit_design = tide_design(fit_prep, periods)
    _, beta = _fit(fit_design, fit_design.zcat.T @ fit_prep.y_res,
                   rotation_stack(fit_design, np.zeros(len(fit_design.blocks))))
    d = tide_design(test_prep, periods)
    zy = d.zcat.T @ test_prep.y_res
    rr = float(test_prep.y_res @ test_prep.y_res)

    def gain(m):
        return float(2 * beta @ (m.T @ zy) - beta @ (m.T @ d.gram @ m) @ beta) / rr

    k = len(d.blocks)
    obs = gain(rotation_stack(d, np.zeros(k)))
    rng = np.random.default_rng(cfg.seed + 1)
    null = np.array([gain(rotation_stack(d, SHIFTS[shift](k, rng, cfg.shift_weeks)))
                     for _ in range(cfg.n_surrogates)])
    p = (1 + np.sum(null >= obs)) / (1 + cfg.n_surrogates)
    return {"oos_variance_share": obs, "p_value": float(p),
            "null_q95": float(np.quantile(null, 0.95)), "n_fit": int(len(fit_prep.y)),
            "n_test": int(len(test_prep.y)), "beta_frozen": [float(v) for v in beta]}


# ------------------------------------------------------------- power

def ar1_noise(resid: np.ndarray, group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise with each run's own spread and lag-1 correlation."""
    out = np.empty_like(resid)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        r = resid[idx]
        sd = r.std() if len(r) > 1 else 1.0
        phi = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 3 else 0.0
        phi = float(np.clip(np.nan_to_num(phi), -0.9, 0.9))
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), len(r))
        x = np.empty(len(r))
        x[0] = rng.normal(0, sd)
        for i in range(1, len(r)):
            x[i] = phi * x[i - 1] + e[i]
        out[idx] = x
    return out


def power_curve(prep: Prepared, relative_ranges=(0.05, 0.10, 0.20, 0.40), n_sim: int = 200,
                cfg: TideTestConfig | None = None, periods: dict[str, float] | None = None,
                wave_period: float | None = None, shift: str = "weeks") -> dict:
    """How often the test finds an M2 wave of a given size in noise shaped like the data.

    The wave's 5-95 % range is a fraction of the median response; its phase is
    random but the same in every run (that is what a tide is). Noise: AR(1) per
    run with the residuals' own spread and persistence. The data's real
    alignment with the tide is never used.
    """
    cfg = cfg or TideTestConfig()
    periods = periods or CONSTITUENTS
    wave_period = wave_period or CONSTITUENTS["M2"]
    rng = np.random.default_rng(cfg.seed + 2)
    design = tide_design(prep, periods)
    k = len(design.blocks)
    m0 = rotation_stack(design, np.zeros(k))
    fitted = prep.y - prep.y_res
    med = abs(float(np.median(prep.y)))
    w = 2 * np.pi * prep.t_hours / wave_period
    out = {}
    for rr in relative_ranges:
        amp = rr * med / (2 * np.cos(0.05 * np.pi))       # 5-95 % range of A cos = 1.975 A
        hits = 0
        for _ in range(n_sim):
            ysim = fitted + ar1_noise(prep.y_res, prep.run, rng) + amp * np.cos(w - rng.uniform(0, 2 * np.pi))
            zy = design.zcat.T @ prep.residualise(ysim)
            ss_obs, _ = _fit(design, zy, m0)
            null = np.array([_fit(design, zy, rotation_stack(design, SHIFTS[shift](k, rng, cfg.shift_weeks)))[0]
                             for _ in range(cfg.n_surrogates)])
            p = (1 + np.sum(null >= ss_obs)) / (1 + cfg.n_surrogates)
            hits += p <= cfg.alpha
        out[str(rr)] = hits / n_sim
    return out


# ------------------------------------------------------------- any plant log

def prepare_series(df: pd.DataFrame, cfg: TideTestConfig | None = None, gap_hours: float = 6.0,
                   block_days: float = 14.0, tod: bool = True) -> Prepared:
    """A plant's own log -> the same base model, ready for the tide or the clock test.

    `df` needs columns `t` (local standard time) and `value`; `run` is optional.
    Without it, a gap longer than `gap_hours` starts a new run. A long
    continuous record is cut into pieces of `block_days`, each shifted on its
    own in the null - one unbroken year would otherwise be a single block, and
    rotating a single block's phase changes nothing.
    """
    cfg = cfg or TideTestConfig()
    d = df.dropna(subset=["t", "value"]).sort_values("t").reset_index(drop=True)
    t = pd.DatetimeIndex(pd.to_datetime(d["t"]))
    th = hours_since_epoch(t)
    if "run" in d and d["run"].notna().all():
        run = d["run"].astype(str).to_numpy()
    else:
        run = np.cumsum(np.r_[True, np.diff(th) > gap_hours]).astype(str)
    run_start = pd.Series(th).groupby(run).transform("min").to_numpy()
    piece = np.floor((th - run_start) / (24 * block_days)).astype(int)     # counted from each run's start
    episode = np.array([f"{r}|{p}" for r, p in zip(run, piece)])
    seg = pd.DataFrame({"t_std": t, "y": d["value"].to_numpy(float), "run": episode, "episode": episode})
    prep = prepare(seg, "y", replace(cfg, block_gap_hours=0.0), covariates=False, tod=tod)
    return prep


def series_test(df: pd.DataFrame, kind: str = "tide", n_surrogates: int = 499,
                block_days: float = 14.0) -> dict:
    """The declared test, pointed at any timestamped plant signal.

    kind='tide': M2, N2, O1, M4 against whole-week shifts, time of day in the base.
    kind='clock': harmonics of the day against 1-23 h shifts, no time of day in the base.
    The placebo periods run beside it on the same data as a calibration check.
    """
    cfg = TideTestConfig(n_surrogates=n_surrogates)
    tide = kind == "tide"
    prep = prepare_series(df, cfg, block_days=block_days, tod=tide)
    n_blocks = len(np.unique(prep.block))
    span_h = float(prep.t_hours.max() - prep.t_hours.min())
    notes = []
    if n_blocks < 4 or span_h < 72 or len(prep.y) < 100:
        return {"status": "insufficient", "n": int(len(prep.y)), "n_blocks": int(n_blocks),
                "span_hours": round(span_h, 1),
                "notes": ["needs at least 100 points over 72 hours, in at least 4 blocks of up to "
                          f"{block_days:g} days - fewer cannot tell a tide from chance"]}
    if n_blocks < 8:
        notes.append(f"only {n_blocks} blocks: the null has little room; treat p as rough")
    periods, shift = (CONSTITUENTS, "weeks") if tide else (DAY_HARMONICS, "hours")
    res = tide_test(prep, periods, cfg, "value", shift=shift)
    # The placebos always run with the time of day in the base. For the clock test this
    # differs from the declared pilot analysis (which had no daily wave to leak): a real,
    # strong daily wave leaks into the 27-h placebo whenever runs start near multiples of
    # nine days (the 24/27-h beat) and would otherwise flag a true answer as untrustworthy.
    prep_placebo = prep if tide else prepare_series(df, cfg, block_days=block_days, tod=True)
    plac = tide_test(prep_placebo, PLACEBO, cfg, "value", with_ranges=False, shift=shift)
    if plac.p_value <= cfg.alpha:
        notes.append("the placebo periods also came out significant: something in this record "
                     "fools the null; do not trust the main p")
    verdict = ("locked" if res.p_value <= cfg.alpha and plac.p_value > cfg.alpha else "not locked")
    out = res.as_dict()
    out.update(status="ok", kind=kind, verdict=verdict, placebo_p_value=plac.p_value,
               span_hours=round(span_h, 1), notes=notes)
    out.pop("beta", None)
    return out
