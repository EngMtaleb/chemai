"""Restart test: finds a planted early-hours step, stays quiet without one."""
import numpy as np
import pandas as pd
import pytest

from chemai.evaluation.restart_test import RestartConfig, cluster_ols, restart_test, window


def synthetic(seed=0, early_step=0.0, n_episodes=60):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_episodes):
        start = pd.Timestamp("2013-03-01") + pd.Timedelta(days=int(rng.integers(0, 700))) \
            + pd.Timedelta(hours=float(rng.uniform(6, 20)))
        t = start + pd.to_timedelta(np.arange(0, 14, 0.25), "h")
        age = (t - start) / pd.Timedelta(hours=1)
        y = 13 + rng.normal(0, 4, len(t)) + early_step * (age < 3) + rng.normal(0, 2)
        rows.append(pd.DataFrame({"t_start": t, "t_std": t, "fouling_pct_h": y, "run": f"r{e}",
                                  "episode": e, "episode_start": start, "q_gpm": 20.0,
                                  "minutes": 13.0, "config": 3}))
    return pd.concat(rows, ignore_index=True)


def test_cluster_se_matches_plain_ols_when_clusters_are_single_rows():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=200), np.ones(200)])
    y = 2 * x[:, 0] + rng.normal(size=200)
    b, se, g = cluster_ols(y, x, np.arange(200))
    assert b == pytest.approx(2, abs=0.2)
    assert g == 200 and 0.03 < se < 0.12


def test_finds_a_planted_early_step():
    w = window(synthetic(early_step=3.0), "episode_start", RestartConfig())
    res = restart_test(w)
    assert res.p_value <= 0.01
    assert res.effect == pytest.approx(3.0, abs=0.8)


def test_quiet_without_a_step():
    ps = [restart_test(window(synthetic(seed=10 + i), "episode_start", RestartConfig())).p_value for i in range(30)]
    assert np.mean(np.array(ps) <= 0.05) <= 0.15
