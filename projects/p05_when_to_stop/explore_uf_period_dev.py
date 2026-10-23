"""Exploration, development years only (2012-2013): does a longer filtration period leave more
fouling behind? Kept to show why Project 5 makes no claim about it (VALIDATION.md 7).

For each segment of an element pair, the next segment of the same pair (one rotation later):
the change in ln R at the segment start is what the rotation left behind. Two reasonable
regressions, cluster-robust by day, give opposite signs for the period's effect.

    python projects/p05_when_to_stop/explore_uf_period_dev.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
P4 = HERE.parent / "p04_tide_fouling" / "results"
GAL_M3 = 3.785411784e-3


def rotations(seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in seg.groupby("run"):
        g = g.sort_values("t_start").reset_index(drop=True)
        for i in range(len(g) - 1):
            c = g.config[i]
            if c not in (3, 5, 6):
                continue
            j = next((k for k in range(i + 1, min(i + 5, len(g))) if g.config[k] == c), None)
            if j is None:
                continue
            rot = (g.t_start[j] - g.t_start[i]).total_seconds() / 60
            if not 0 < rot <= 150:
                continue
            rows.append({"day": g.t_start[i].normalize(), "config": c, "period_min": rot / 3,
                         "vol_m3": float(np.mean(g.q_gpm[i:j]) * rot * GAL_M3),
                         "y": g.ln_r_start[j] - g.ln_r_start[i], "rate": g.fouling_pct_h[i],
                         "temp": g.temp_c[i], "lnr": g.ln_r_start[i]})
    return pd.DataFrame(rows)


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    ids = np.unique(groups)
    for g in ids:
        s = X[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    n, k, m = len(y), X.shape[1], len(ids)
    v = bread @ meat @ bread * (m / (m - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(v))


def main():
    from scipy.stats import norm
    seg = pd.read_parquet(P4 / "segments.parquet")
    seg = seg[pd.DatetimeIndex(seg.t_start).year <= 2013]            # development years only
    r = rotations(seg)
    groups = pd.factorize(r.day)[0]
    cfg = pd.get_dummies(r.config, prefix="c", drop_first=True).to_numpy(float)
    base = np.column_stack([np.ones(len(r)), r.rate, r.temp, r.lnr, cfg])
    out = {"rotations": int(len(r)), "days": int(len(np.unique(groups)))}
    # 1: what a rotation leaves behind per m3, against the period
    X1 = np.column_stack([base, r.period_min])
    b1, s1 = ols_cluster(X1, (r.y / r.vol_m3).to_numpy(), groups)
    # 2: what a rotation leaves behind, against volume and volume x period
    X2 = np.column_stack([base, r.vol_m3, r.vol_m3 * r.period_min])
    b2, s2 = ols_cluster(X2, r.y.to_numpy(), groups)
    for name, b, s in (("per_m3_on_period", b1[-1], s1[-1]), ("volume_x_period", b2[-1], s2[-1])):
        z = b / s
        out[name] = {"coef": float(b), "se": float(s), "p": float(2 * (1 - norm.cdf(abs(z))))}
    (HERE / "results" / "explore_uf_period_dev.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
