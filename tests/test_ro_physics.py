"""RO physics. Each test is a rule the physics imposes, or an error that was made
once and must not come back."""
import numpy as np
import pandas as pd
import pytest

from chemai.config import ROConfig
from chemai.features.ro_physics import (blockage_fraction, blockage_transform, dp_from_blockage,
                                        fit_flow_exponent, log_mean_concentration_factor,
                                        net_driving_pressure, normalise, normalised_dp,
                                        osmotic_pressure, tcf, water_viscosity)

cfg = ROConfig()


def test_viscosity_of_water_at_25c_is_089_mpa_s():
    assert water_viscosity(25.0) == pytest.approx(0.89e-3, rel=0.02)
    assert water_viscosity(35.0) < water_viscosity(25.0) < water_viscosity(15.0)


def test_tcf_is_one_at_reference_and_rises_with_temperature():
    assert tcf(25.0) == pytest.approx(1.0)
    assert tcf(30.0) > 1.0 > tcf(20.0)
    # about 3 % per kelvin for water near 25 C
    assert tcf(26.0) == pytest.approx(1.03, abs=0.005)


def test_rejects_kelvin_given_as_celsius():
    with pytest.raises(ValueError, match="Kelvin"):
        water_viscosity(298.15)
    with pytest.raises(ValueError):
        tcf([25.0, 300.0])


def test_seawater_osmotic_pressure_is_about_27_bar():
    assert osmotic_pressure(35.0, 25.0) == pytest.approx(27.0, abs=1.5)


def test_concentration_factor_at_40_percent_recovery():
    assert log_mean_concentration_factor(0.40) == pytest.approx(1.277, abs=0.002)
    with pytest.raises(ValueError):
        log_mean_concentration_factor(1.2)            # recovery given in percent / 100 wrong


def test_net_driving_pressure_is_below_feed_pressure():
    ndp = net_driving_pressure(60.0, 58.6, 1.0, 35.0, 0.42, 25.0)
    assert 0 < ndp < 60.0


def test_normalised_dp_removes_flow_when_the_exponent_matches():
    """dP ~ Q^(2-m) mu^m: normalising with the same form returns the reference dP
    at any flow and temperature."""
    m = 0.25
    q = np.array([8.0, 10.0, 12.0, 14.0])
    t = np.array([18.0, 22.0, 28.0, 33.0])
    mu_ratio = water_viscosity(t) / water_viscosity(25.0)
    dp = 1.2 * (q / 10.0) ** (2 - m) * mu_ratio ** m
    out = normalised_dp(dp, q, t, 10.0, flow_exponent=2 - m, visc_exponent=m)
    assert np.allclose(out, 1.2)


def test_a_linear_flow_ratio_reads_a_production_increase_as_fouling():
    """The error that motivates the physics core: a 20 % production increase,
    normalised with exponent 1.0 when the channel behaves as 1.75, looks like
    +15 % of pressure drop that is not there."""
    dp = 1.2 * 1.2 ** 1.75
    wrong = normalised_dp(dp, 12.0, 25.0, 10.0, flow_exponent=1.0, visc_exponent=0.0)
    right = normalised_dp(dp, 12.0, 25.0, 10.0, flow_exponent=1.75, visc_exponent=0.0)
    assert right == pytest.approx(1.2)
    assert wrong / 1.2 - 1 == pytest.approx(0.146, abs=0.005)


def test_gap_law_is_one_when_clean_and_accelerates_as_the_gap_closes():
    phi = np.array([0.0, 0.1, 0.2, 0.3])
    x = dp_from_blockage(phi, n=2.2)
    assert x[0] == pytest.approx(1.0)
    step = np.diff(x)
    assert np.all(step[1:] > step[:-1])               # convex: each 10 % costs more
    assert x[1] == pytest.approx(0.9 ** -2.2)         # 10 % gap loss -> +26 %


def test_blockage_transform_inverts_the_gap_law():
    phi = np.linspace(0, 0.5, 11)
    assert np.allclose(blockage_fraction(dp_from_blockage(phi, n=2.2), n=2.2), phi)
    # steady growth of the layer is a straight line in DP_norm^(-1/n)
    z = blockage_transform(dp_from_blockage(0.01 * np.arange(20), n=2.2), n=2.2)
    assert np.allclose(np.diff(z), -0.01)


def test_lead_element_fouling_is_more_convex_than_even_fouling():
    """Biofouling collects in the lead element: for the same initial slope the
    train-level curve bends upward much faster."""
    even = dp_from_blockage(np.array([0.0, 0.01, 0.02]), n=2.2)
    lead = dp_from_blockage(np.array([0.0, 0.05, 0.10]), n=2.2, lead_share=0.2, rest_ratio=0.0)
    curv_even = (even[2] - 2 * even[1] + even[0]) / (even[1] - even[0]) ** 2
    curv_lead = (lead[2] - 2 * lead[1] + lead[0]) / (lead[1] - lead[0]) ** 2
    assert curv_lead > 3 * curv_even


def _plant_days(n=200, exponent=1.75, seed=0, steady=False):
    rng = np.random.default_rng(seed)
    q_p = np.full(n, 10.0) if steady else 10.0 * np.where(rng.random(n) < 0.5, 0.8, 1.05)
    rec = 0.42
    q_f, q_c = q_p / rec, q_p / rec - q_p
    t = 25 + 4 * np.sin(np.arange(n) / 58.0) + rng.normal(0, 0.3, n)
    mu = water_viscosity(t) / water_viscosity(25.0)
    q_avg = (q_f + q_c) / 2
    dp = 1.3 * (q_avg / q_avg.mean()) ** exponent * mu ** (2 - exponent) * (1 + 0.0005 * np.arange(n))
    dp = dp * (1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(dict(day=np.arange(n), feed_flow=q_f, conc_flow=q_c, perm_flow=q_p,
                             feed_pressure=60.0, conc_pressure=60.0 - dp, perm_pressure=1.0,
                             temperature_c=t, feed_tds=35.0, perm_tds=0.15, dp=dp))


@pytest.mark.parametrize("true_n", [1.5, 1.75, 1.85])
def test_the_plant_can_measure_its_own_flow_exponent(true_n):
    out = fit_flow_exponent(_plant_days(exponent=true_n))
    assert out["ok"] and out["n"] == pytest.approx(true_n, abs=0.05)


def test_steady_production_cannot_measure_the_exponent():
    out = fit_flow_exponent(_plant_days(steady=True))
    assert not out["ok"] and "steady" in out["note"]


def test_normalise_returns_one_at_start_up_and_rejects_unbalanced_flows():
    df = _plant_days()
    out = normalise(df, baseline_rows=14, flow_exponent=1.75, visc_exponent=0.25)
    assert out["dp_norm"].iloc[:14].median() == pytest.approx(1.0, abs=0.01)
    bad = df.copy()
    bad.loc[3, "feed_flow"] *= 1.2
    with pytest.raises(ValueError, match="flow meters"):
        normalise(bad)
