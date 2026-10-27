"""Project 6: the ladder's helpers, the physics inside the networks, and the features."""
import numpy as np
import pandas as pd
import pytest

from chemai.features.ccro_forecast import GRID_R, best_stop
from chemai.features.uf_forecast import LAWS, fit_hermia, hermia_ln_ratio
from chemai.models.ladder import group_bootstrap, rp, threshold_at_precision, threshold_best_f1


def test_rp_counts_and_rates():
    y = np.array([1, 1, 0, 0, 1], bool)
    f = np.array([1, 0, 1, 0, 1], bool)
    d = rp(y, f)
    assert (d["tp"], d["fp"], d["fn"]) == (2, 1, 1)
    assert d["recall"] == pytest.approx(2 / 3, abs=1e-3) and d["precision"] == pytest.approx(2 / 3, abs=1e-3)


def test_threshold_at_precision_gives_the_most_recall_that_precision_allows():
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    thr = threshold_at_precision(y, p, 0.75)
    flag = p >= thr
    assert rp(y.astype(bool), flag)["precision"] >= 0.75
    # one step lower would break the precision target
    assert rp(y.astype(bool), p >= 0.5)["precision"] < 0.75
    assert threshold_best_f1(y, p) in p


def test_group_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(10), 5)
    x = np.repeat(np.arange(10, dtype=float), 5)            # one value per group
    lo, hi = group_bootstrap(groups, lambda i: float(x[i].mean()), n=300)
    assert lo < 4.5 < hi and lo > 1.5 and hi < 7.5


@pytest.mark.parametrize("law", LAWS)
def test_the_four_hermia_laws_agree_when_the_rise_is_small(law):
    k, t = 1e-4, 12.0                                         # per minute: a 0.12 % rise
    assert hermia_ln_ratio(law, k, t) == pytest.approx(k * t, rel=0.01)


def test_hermia_laws_have_their_constant_flux_forms():
    k, t = 0.02, 10.0
    assert hermia_ln_ratio("complete", k, t) == pytest.approx(-np.log(1 - k * t))
    assert hermia_ln_ratio("standard", k, t) == pytest.approx(-2 * np.log(1 - k * t / 2))
    assert hermia_ln_ratio("intermediate", k, t) == pytest.approx(k * t)
    assert hermia_ln_ratio("cake", k, t) == pytest.approx(np.log1p(k * t))


def test_fit_hermia_recovers_a_cake_curve_with_a_large_rise():
    t = np.linspace(0, 12, 13)
    z = hermia_ln_ratio("cake", 0.03, t)                      # resistance x1.36 by minute 12
    h = fit_hermia(t, z)
    assert min(LAWS, key=lambda lw: h[lw][1]) == "cake"
    assert h["cake"][0] == pytest.approx(0.03, rel=0.05)


def test_best_stop_on_a_straight_line_matches_the_square_root_law():
    # m = m0 + s V with V in m3: V* = sqrt(2 F_eff / s) when the cost is integrated from V = 0
    m0, s, V20 = 10.0, 100.0, 0.03                            # kWh/m3, kWh/m6, m3
    F = 0.05
    E20 = m0 * V20 + s * V20 ** 2 / 2                         # energy if the line held from V = 0
    feat = dict(V20_l=V20 * 1000, E20_kwh=E20, m20=m0 + s * V20, prior_drain_kwh=F, prior_brine_l=0.0)
    ratio = (m0 + s * GRID_R * V20) / feat["m20"]
    v = best_stop(ratio, feat, b=0.0) / 1000
    assert v == pytest.approx(np.sqrt(2 * F / s), rel=0.03)


def test_networks_respect_their_physics():
    torch = pytest.importorskip("torch")
    from chemai.models.deep import GapLawNet, HermiaNet, PowerLawNet
    torch.manual_seed(0)
    x = torch.randn(16, 5)
    p = GapLawNet(3, 5)(x, torch.randn(16, 3, 42), torch.tensor([[1.15, 1.05]] * 16))
    assert torch.all((p >= 0) & (p <= 1))
    rise = HermiaNet(5)(x)
    assert torch.all(rise >= 0)                              # a resistance that only climbs
    net = PowerLawNet(5)
    r = torch.linspace(1, 5, 16)
    out = net(x, r, torch.full((16,), 0.05))
    assert torch.all(out >= 1 - 1e-6)                        # cost never below its minute-20 value
    assert torch.all(torch.diff(net(x[:1].repeat(16, 1), r, torch.full((16,), 0.05))) > 0)


def test_ro_decision_features_read_only_the_past():
    from chemai.features.ro_decisions import decision_features
    day = np.arange(0, 60, dtype=float)
    dp = 1.0 + 0.002 * day
    dp[day > 40] = 1.5                                          # a jump after the decision day
    cens = dp <= 1.0
    f, seq = decision_features(day, dp, cens, d=35, level=1.15)
    assert f["now7"] == pytest.approx(1.0 + 0.002 * 32, abs=1e-6)  # the median of days 29-35
    assert f["max_cycle"] < 1.1                                 # nothing after day 35 leaks in
    assert f["margin_flag"] is False and seq.shape == (2, 42)
