"""When will this train reach its review level? A forecast, its interval, and
whether it is an extrapolation too far to trust.

The forecast is deliberately simple: a robust straight line through the recent
days of the current cleaning cycle - in DP_norm itself (`law='linear'`), or in
DP_norm^(-1/n), which the gap law says is straight when the foulant layer grows
steadily (`law='blockage'`). Which one ships was decided on simulated plants,
where the answer is known, before the real deposit was touched
(projects/p03_ro_fouling/VALIDATION.md, sections 7-10).

Three things travel with every number:

  * a P10-P90 interval. The line's uncertainty is widened for day-to-day
    correlation (an effective sample size from the lag-1 autocorrelation of the
    residuals) and by one factor calibrated on simulated development plants.
    A residual bootstrap was tried first and rejected: it recovered only
    41-84 % of the slope's true spread, depending on the autocorrelation
    (VALIDATION.md section 9; projects/p03_ro_fouling/step4b_interval_check.py);
  * the extrapolation ratio, days ahead over days observed. A forecast that
    looks further ahead than twice what it has seen is flagged, and the flag
    was tested: flagged forecasts are the ones that miss;
  * plain warnings when the data cannot carry a forecast at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import logging

import numpy as np

from chemai.config import ROConfig

log = logging.getLogger(__name__)

LAWS = ("linear", "blockage")


@dataclass
class Forecast:
    """What the forecast returns. Never a date without its interval and its flag."""
    status: str                     # 'at_or_above' | 'crossing' | 'not_in_sight' | 'falling' | 'insufficient'
    law: str
    review_level: float
    level_now: float | None         # DP_norm at the last observed day, from the fit
    rate_per_week: float | None     # DP_norm units per week, now
    rate_lo: float | None
    rate_hi: float | None
    days_to_level: float | None     # P50; None when beyond the horizon
    days_lo: float | None           # P10
    days_hi: float | None           # P90; None when beyond the horizon
    extrapolation_ratio: float | None
    in_envelope: bool
    n_points: int
    span_days: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ the line

def _transform(x, law: str, n: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if law == "linear":
        return x
    if law == "blockage":
        return np.power(x, -1.0 / n)
    raise ValueError(f"law must be one of {LAWS}")


def _level_from(z, law: str, n: float):
    z = np.asarray(z, dtype=float)
    if law == "linear":
        return z
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(z > 0, np.power(np.maximum(z, 1e-12), -n), np.inf)


def _theil_sen(t: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Theil-Sen for one series or many at once (rows of Z): the median of all
    pairwise slopes, and the median intercept. One bad day cannot tilt it."""
    Z = np.atleast_2d(Z)
    i, j = np.triu_indices(len(t), 1)
    slopes = (Z[:, j] - Z[:, i]) / (t[j] - t[i])
    b = np.median(slopes, axis=1)
    a = np.median(Z - b[:, None] * t[None, :], axis=1)
    return a, b


def _fit_line(t: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    a, b = _theil_sen(t, z)
    return float(a[0]), float(b[0])


def residual_structure(t: np.ndarray, z: np.ndarray, a: float, b: float) -> tuple[float, float]:
    """Robust residual scatter (MAD) and lag-1 autocorrelation of the residuals."""
    r = z - (a + b * t)
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    rho = 0.0
    if len(r) > 4 and np.std(r[:-1]) > 0 and np.std(r[1:]) > 0:
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return max(sigma, 1e-9), float(np.clip(rho, 0.0, 0.9))


def line_draws(t: np.ndarray, z: np.ndarray, a: float, b: float, n_draws: int, inflation: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Plausible lines around the fit: intercept at the window's centre and slope
    drawn independently (they are uncorrelated there), each widened by the
    effective-sample-size factor and the calibration factor."""
    sigma, rho = residual_structure(t, z, a, b)
    n = len(t)
    n_eff = max(3.0, n * (1 - rho) / (1 + rho))
    t_bar = float(t.mean())
    sxx = float(((t - t_bar) ** 2).sum())
    infl = np.sqrt(n / n_eff) * inflation
    c = a + b * t_bar
    c_k = c + rng.normal(0, sigma * np.sqrt(1.0 / n) * infl, n_draws)
    b_k = b + rng.normal(0, sigma / np.sqrt(sxx) * infl, n_draws)
    return c_k - b_k * t_bar, b_k


def _crossing(a, b, t_now: float, target_z: float, law: str) -> np.ndarray:
    """Days from t_now until each line reaches the review level (inf if never)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rising = b > 0 if law == "linear" else b < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = (target_z - a) / b
    out = np.where(rising, np.maximum(0.0, t_star - t_now), np.inf)
    return out


def _rate(a, b, t: float, law: str, n: float) -> np.ndarray:
    """dDP_norm/dt per week at day t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if law == "linear":
        return 7.0 * b
    z = a + b * t
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(z > 0, 7.0 * (-n) * np.maximum(z, 1e-12) ** (-n - 1.0) * b, np.inf)


# ------------------------------------------------------------------ the window

def prepare_window(day, dp_norm, cfg: ROConfig | None = None, censored=None):
    """The points a forecast is fitted on: the trailing window of the current
    cycle, without the flushing transient right after the cleaning."""
    cfg = cfg or ROConfig()
    t = np.asarray(day, dtype=float)
    x = np.asarray(dp_norm, dtype=float)
    c = np.zeros(len(t), bool) if censored is None else np.asarray(censored, bool)
    if not (len(t) == len(x) == len(c)):
        raise ValueError("day, dp_norm and censored must have the same length")
    ok = np.isfinite(t) & np.isfinite(x) & (x > 0)
    t, x, c = t[ok], x[ok], c[ok]
    order = np.argsort(t)
    t, x, c = t[order], x[order], c[order]
    if len(t) and np.any(np.diff(t) == 0):
        raise ValueError("one value per day: repeated day found")
    if len(t) and t[0] < 0:
        raise ValueError("days are counted from the end of the last cleaning: no negative days")
    keep = t >= cfg.skip_after_cip_days
    if len(t):
        keep &= t >= t[-1] - cfg.fit_window_days
    return t[keep], x[keep], c[keep]


# ------------------------------------------------------------------ forecast

def forecast_cycle(day, dp_norm, review_level: float | None = None, law: str | None = None,
                   censored=None, cfg: ROConfig | None = None, seed: int | None = None) -> Forecast:
    """Forecast when DP_norm reaches `review_level` in the current cycle.

    `day`: days since the last cleaning ended. `dp_norm`: the normalised pressure
    drop over its start-up value (1.15 = +15 %). `censored`: days whose value is
    only known to be at or below the baseline (the deposit's zeros).
    """
    cfg = cfg or ROConfig()
    law = law or cfg.law
    level = cfg.review_level if review_level is None else float(review_level)
    n_exp = cfg.gap_exponent
    if law not in LAWS:
        raise ValueError(f"law must be one of {LAWS}")
    if not 1.0 < level < 3.0:
        raise ValueError("review_level is DP_norm over start-up: between 1 and 3 (1.15 = +15 %)")
    t, x, c = prepare_window(day, dp_norm, cfg, censored)
    warnings: list[str] = []
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    base = dict(law=law, review_level=level, n_points=int(len(t)), span_days=span)
    if len(t) < cfg.min_points or span < cfg.min_span_days:
        warnings.append(f"insufficient data: {len(t)} points over {span:.0f} days since the "
                        f"cleaning (need {cfg.min_points} over {cfg.min_span_days})")
        return Forecast(status="insufficient", level_now=None, rate_per_week=None, rate_lo=None,
                        rate_hi=None, days_to_level=None, days_lo=None, days_hi=None,
                        extrapolation_ratio=None, in_envelope=False, warnings=warnings, **base)
    censored_share = float(c.mean())
    if censored_share > 0.2:
        warnings.append(f"{censored_share:.0%} of the window is at or below the start-up baseline "
                        "(censored): the level is an upper bound")

    z = _transform(x, law, n_exp)
    a, b = _fit_line(t, z)
    t_now = float(t[-1])
    target = float(_transform([level], law, n_exp)[0])
    level_now = float(_level_from(a + b * t_now, law, n_exp))
    rate = float(_rate(a, b, t_now, law, n_exp))

    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    ak, bk = line_draws(t, z, a, b, cfg.n_draws, cfg.interval_inflation, rng)
    days_k = np.minimum(_crossing(ak, bk, t_now, target, law), 1e9)     # never -> very far
    rate_k = _rate(ak, bk, t_now, law, n_exp)
    rate_k = rate_k[np.isfinite(rate_k)] if np.isfinite(rate_k).any() else np.array([rate])
    lo_q, hi_q = (1 - cfg.interval) / 2, 1 - (1 - cfg.interval) / 2
    rate_lo, rate_hi = (float(np.quantile(rate_k, q)) for q in (lo_q, hi_q))
    d_lo, d_mid, d_hi = (float(np.quantile(days_k, q)) for q in (lo_q, 0.5, hi_q))
    d_point = float(_crossing(a, b, t_now, target, law))

    horizon = cfg.horizon_max_days
    fin = lambda v: None if not np.isfinite(v) or v > horizon else round(float(v), 1)   # noqa: E731
    d_lo, d_mid, d_hi = (np.inf if v >= 1e9 else v for v in (d_lo, d_mid, d_hi))
    if level_now >= level:
        status = "at_or_above"
    elif rate_hi < 0:
        status = "falling"
    elif not np.isfinite(d_point) or d_point > horizon:
        status = "not_in_sight"
    else:
        status = "crossing"
    ratio = None
    if status == "crossing":
        ratio = round(d_point / max(span, 1.0), 2)
    elif status == "not_in_sight" and np.isfinite(d_lo) and d_lo <= horizon:
        ratio = round(d_lo / max(span, 1.0), 2)
    in_env = bool(censored_share <= 0.2 and (ratio is None or ratio <= cfg.extrapolation_ratio_max))
    if ratio is not None and ratio > cfg.extrapolation_ratio_max:
        warnings.append(f"looking {ratio:.1f}x further ahead than the window it has seen: "
                        "treat the date as weak")
    at = status == "at_or_above"
    return Forecast(
        status=status, level_now=round(level_now, 4), rate_per_week=round(rate, 5),
        rate_lo=round(rate_lo, 5), rate_hi=round(rate_hi, 5),
        days_to_level=0.0 if at else fin(d_point),
        days_lo=0.0 if at else fin(d_lo),
        days_hi=0.0 if at else fin(d_hi),
        extrapolation_ratio=ratio, in_envelope=in_env, warnings=warnings, **base)


def level_at(day, dp_norm, horizon_days: float, law: str | None = None, censored=None,
             cfg: ROConfig | None = None) -> float | None:
    """The fitted DP_norm `horizon_days` after the last observation - used to score
    the forecast against what was later measured, where crossings are too rare."""
    cfg = cfg or ROConfig()
    law = law or cfg.law
    t, x, _ = prepare_window(day, dp_norm, cfg, censored)
    if len(t) < cfg.min_points or (t[-1] - t[0]) < cfg.min_span_days:
        return None
    z = _transform(x, law, cfg.gap_exponent)
    a, b = _fit_line(t, z)
    return float(_level_from(a + b * (t[-1] + horizon_days), law, cfg.gap_exponent))
