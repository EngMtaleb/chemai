"""Step 4b - why the forecast's interval is drawn, not bootstrapped (recorded re-run).

The check that first rejected the residual bootstrap was an exploratory run that
was not saved. This script repeats it so the numbers are on record. It was
written after the external test; it uses no plant data and changes no setting.

A known straight line plus AR(1) noise, fitted the way the forecast fits it
(Theil-Sen through days 3-42). Monte Carlo gives the true spread of the fitted
slope; each method's estimate of that spread is compared with it:

  * residual bootstrap, independent draws;
  * residual bootstrap in blocks of seven days;
  * the forecast's own drawn lines (AR(1) effective sample size), without and
    with the calibrated inflation factor.

    python projects/p03_ro_fouling/step4b_interval_check.py
"""
import json
from pathlib import Path

import numpy as np

from chemai.config import ROConfig
from chemai.models.fouling_forecast import _fit_line, _theil_sen, line_draws

OUT = Path(__file__).parent / "results"


def ar1(n: int, rho: float, sigma: float, rng) -> np.ndarray:
    """Stationary AR(1) noise with standard deviation sigma."""
    e = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def bootstrap_sd(t, z, a, b, rng, n_boot=300, block=1) -> float:
    """Spread of the Theil-Sen slope over residual-bootstrap series."""
    r = z - (a + b * t)
    n = len(t)
    if block == 1:
        R = r[rng.integers(0, n, (n_boot, n))]
    else:
        k = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, (n_boot, k))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        R = r[idx]
    _, slopes = _theil_sen(t, a + b * t + R)
    return float(np.std(slopes))


def main():
    cfg = ROConfig()
    rng = np.random.default_rng(2026)
    t = np.arange(3, 43, dtype=float)              # the forecast's window: skip 3 days, 40 days
    slope, sigma, n_mc, n_est = 0.0016, 0.005, 2000, 300
    res = dict(window_days=[3, 42], slope_per_day=slope, noise_sd=sigma, monte_carlo=n_mc, estimated_on=n_est,
               note="ratio = median estimated slope spread / true (Monte Carlo) slope spread; 1.00 is right")
    for rho in (0.0, 0.3, 0.6):
        fits, est = [], {"bootstrap_iid": [], "bootstrap_block7": [], "drawn_no_inflation": [], "drawn_inflation_2": []}
        for k in range(n_mc):
            z = 1.05 + slope * t + ar1(len(t), rho, sigma, rng)
            a, b = _fit_line(t, z)
            fits.append(b)
            if k < n_est:
                est["bootstrap_iid"].append(bootstrap_sd(t, z, a, b, rng))
                est["bootstrap_block7"].append(bootstrap_sd(t, z, a, b, rng, block=7))
                for key, infl in (("drawn_no_inflation", 1.0), ("drawn_inflation_2", cfg.interval_inflation)):
                    _, bk = line_draws(t, z, a, b, 2000, infl, rng)
                    est[key].append(float(np.std(bk)))
        true_sd = float(np.std(fits))
        res[f"rho {rho}"] = dict(true_slope_sd=round(true_sd, 7),
                                 **{k: round(float(np.median(v)) / true_sd, 2) for k, v in est.items()})
        print(rho, json.dumps(res[f"rho {rho}"]), flush=True)
    (OUT / "step4b_interval_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
