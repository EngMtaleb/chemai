"""Solar plan: the LP is a bound, the rule never beats it, demand is always met."""
import numpy as np
import pytest

from chemai.models.solar_plan import SolarPlanConfig, flux_penalty, plan_lp, plan_rule, pv_year


@pytest.fixture(scope="module")
def pv():
    return pv_year(SolarPlanConfig(), days=60)


def test_pv_year_has_nights_and_a_sensible_yield(pv):
    daily = pv.reshape(-1, 24).sum(axis=1)
    assert np.all(pv.reshape(-1, 24)[:, :4] == 0)                 # no sun before 04:00
    assert 4.0 < daily.mean() < 6.5


def test_no_pv_means_no_solar(pv):
    assert plan_lp(pv, 1.0, 4, 0.0).solar_fraction == pytest.approx(0.0, abs=1e-9)


def test_lots_of_everything_runs_on_the_sun(pv):
    assert plan_lp(pv, 3.0, 36, 3.0).solar_fraction > 0.98


def test_the_rule_never_beats_the_plan(pv):
    for c, s, p in [(1.5, 8, 0.6), (2.5, 16, 0.9)]:
        assert plan_rule(pv, c, s, p).solar_fraction <= plan_lp(pv, c, s, p).solar_fraction + 1e-6


def test_running_faster_costs_fouling_only_above_n_equal_1():
    assert flux_penalty(2.5, 1.0) == pytest.approx(1.0)
    assert flux_penalty(2.5, 1.5) == pytest.approx(2.5 ** 0.5)


def test_tidal_turbine_is_typical_and_runs_on_the_moon():
    from chemai.models.solar_plan import tidal_year
    td = tidal_year(days=60)
    assert 0.30 < td.mean() < 0.40                      # typical tidal-stream capacity factor
    assert td.reshape(-1, 24)[:, :6].mean() > 0.2       # it delivers before dawn


def test_a_second_supply_never_lowers_the_renewable_share(pv):
    from chemai.models.solar_plan import tidal_year
    td = tidal_year(days=60)
    a = plan_lp(pv, 2.0, 8, 0.5).solar_fraction
    b = plan_lp(pv, 2.0, 8, 0.5, other=td * 0.1).solar_fraction
    assert b >= a - 1e-9
