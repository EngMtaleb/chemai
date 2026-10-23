"""Does the membrane foul differently in the first hours after a restart?

A plant run on the sun stops every evening and starts every morning, so the
first hours after a restart happen once a day instead of once a month. If they
foul faster, daily cycling has a price; if slower, the night's rest is a free
partial cleaning. The pilot record has 82 restarts after the unit stood idle for
at least 2 hours, at many different times of day.

    window   segments in the first 12 h after a restart
    model    fouling rate ~ episode level + early (first 3 h) + time of day
                            + flow + segment length + element pair
    test     the `early` coefficient, cluster-robust by episode (CR1), t with G-1 df
    placebo  the same model at a pretend restart 24 h into long episodes
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from chemai.evaluation.tide_test import dummies, time_of_day_block, winsorise_within


@dataclass(frozen=True)
class RestartConfig:
    min_gap_hours: float = 2.0          # idle at least this long before the episode
    early_hours: float = 3.0            # the window right after the restart
    window_hours: float = 12.0          # compared with the rest of the first 12 h
    placebo_start_hours: float = 24.0   # pretend restart this far into episodes of 36 h or more
    tod_harmonics: int = 4
    winsor: tuple[float, float] = (0.01, 0.99)
    alpha: float = 0.01
    alpha_confirm: float = 0.05
    relevance: float = 0.10             # |early effect| / median fouling rate
    seed: int = 20261023


@dataclass
class RestartResult:
    n_segments: int
    n_episodes: int
    effect: float                       # fouling-rate units (%/h)
    se: float
    t: float
    p_value: float
    ci95: tuple
    median_y: float
    relative_effect: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def window(seg: pd.DataFrame, start_col: str, cfg: RestartConfig, offset_h: float = 0.0) -> pd.DataFrame:
    """Segments within [offset, offset + window) hours of the episode start."""
    age = (seg["t_start"] - seg[start_col]).dt.total_seconds() / 3600 - offset_h
    out = seg[(age >= 0) & (age < cfg.window_hours)].copy()
    out["age_h"] = age[out.index]
    out["early"] = (out["age_h"] < cfg.early_hours).astype(float)
    return out


def design(w: pd.DataFrame, response: str, cfg: RestartConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run = w["run"].to_numpy()
    y = winsorise_within(w[response].to_numpy(float), w["episode"].to_numpy(), *cfg.winsor)
    dev = lambda v: v - pd.Series(v).groupby(w["episode"].to_numpy()).transform("mean").to_numpy()
    x = np.column_stack([
        w["early"].to_numpy(float),
        dummies(w["episode"].to_numpy()),
        time_of_day_block(w["t_std"], cfg.tod_harmonics),
        dev(np.log(w["q_gpm"].to_numpy(float)))[:, None],
        np.log(w["minutes"].to_numpy(float))[:, None],
        dummies(w["config"].to_numpy(), drop_first=True),
    ])
    return y, x, w["episode"].to_numpy()


def cluster_ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray, coef: int = 0) -> tuple[float, float, int]:
    """OLS coefficient and CR1 cluster-robust standard error."""
    keep = np.linalg.norm(x, axis=0) > 0
    x = x[:, keep]
    q, r = np.linalg.qr(x)
    ok = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
    if not ok[coef]:
        raise ValueError("the tested column is collinear with the rest of the design")
    x = x[:, ok]
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    e = y - x @ beta
    groups = np.unique(cluster)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for g in groups:
        sel = cluster == g
        s = x[sel].T @ e[sel]
        meat += np.outer(s, s)
    n, k, g_n = len(y), x.shape[1], len(groups)
    scale = g_n / (g_n - 1) * (n - 1) / (n - k)
    v = scale * xtx_inv @ meat @ xtx_inv
    return float(beta[coef]), float(np.sqrt(v[coef, coef])), g_n


def restart_test(w: pd.DataFrame, response: str = "fouling_pct_h", cfg: RestartConfig | None = None) -> RestartResult:
    cfg = cfg or RestartConfig()
    y, x, cl = design(w, response, cfg)
    b, se, g = cluster_ols(y, x, cl)
    t = b / se
    p = 2 * stats.t.sf(abs(t), df=g - 1)
    half = stats.t.ppf(0.975, df=g - 1) * se
    med = float(np.median(y))
    return RestartResult(n_segments=len(y), n_episodes=g, effect=b, se=se, t=t, p_value=float(p),
                         ci95=(b - half, b + half), median_y=med, relative_effect=b / med)


def power(w: pd.DataFrame, response: str = "fouling_pct_h", cfg: RestartConfig | None = None,
          sizes=(0.05, 0.10, 0.20, 0.40), n_sim: int = 300) -> dict:
    """Detection rate for an early-window step of a given size (share of the median).

    Noise: the base model's residuals reshuffled as AR(1) per episode with
    their own spread and lag-1 correlation. The real early coefficient is
    never estimated here.
    """
    from chemai.evaluation.tide_test import ar1_noise
    cfg = cfg or RestartConfig()
    rng = np.random.default_rng(cfg.seed + 3)
    y, x, cl = design(w, response, cfg)
    base = x[:, 1:]
    base = base[:, np.linalg.norm(base, axis=0) > 0]
    beta = np.linalg.lstsq(base, y, rcond=None)[0]
    fitted = base @ beta
    resid = y - fitted
    med = abs(float(np.median(y)))
    out = {}
    for s in sizes:
        hits = 0
        for _ in range(n_sim):
            ysim = fitted + ar1_noise(resid, cl, rng) + s * med * x[:, 0]
            b, se, g = cluster_ols(ysim, x, cl)
            hits += 2 * stats.t.sf(abs(b / se), df=g - 1) <= cfg.alpha
        out[str(s)] = hits / n_sim
    return out
