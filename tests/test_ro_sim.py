"""The RO plant simulator - the ground truth the forecast is chosen on."""
import numpy as np
import pytest

from chemai.config import ROSimConfig
from chemai.data.ro_sim import (clean, counterfactual, gap_law, generate_plants, sample_plant,
                                simulate_plant, step_state)
from chemai.evaluation.fouling_eval import plant_view, sim_cycles

cfg = ROSimConfig(days=300)


@pytest.fixture(scope="module")
def run():
    return simulate_plant(sample_plant(np.random.default_rng(3), cfg, "biofilm"), cfg)


def test_same_seed_same_plant():
    a = simulate_plant(sample_plant(np.random.default_rng(5), cfg, "linear"), cfg)
    b = simulate_plant(sample_plant(np.random.default_rng(5), cfg, "linear"), cfg)
    assert a.daily["dp"].dropna().equals(b.daily["dp"].dropna())
    assert a.cleanings.equals(b.cleanings)


def test_a_cleaning_lowers_the_true_pressure_drop(run):
    p = run.params.trains[0]
    s = np.array([0.20, 0.02, 0.0])
    after = clean(s, p, 0.8)
    assert gap_law(after.sum(), p) < gap_law(s.sum(), p)
    assert after[1] > s[1]                           # a little is left for good


def test_without_cleaning_the_path_stays_above_the_cleaned_one(run):
    tr = run.cleanings.iloc[0]
    d = int(tr["day"])
    path = counterfactual(run, tr["train"], d - 2, 10)
    real = run.daily[(run.daily["train"] == tr["train"]) & (run.daily["day"].between(d - 1, d + 8))]
    assert path[1:10].min() > real["dp_true"].min()


def test_warm_water_grows_biofilm_faster(run):
    p = run.params.trains[0]
    s = np.array([0.05, 0.0, 0.0])
    cold = step_state(s, p, p.q10 ** ((20 - 25) / 10), 0.0)
    warm = step_state(s, p, p.q10 ** ((30 - 25) / 10), 0.0)
    assert warm[0] - s[0] > cold[0] - s[0]


def test_a_day_is_recorded_only_when_two_trains_run(run):
    rec = run.daily[run.daily["recorded"]]
    assert (rec["n_on"] >= 2).all()


def test_production_per_train_moves_with_the_number_of_trains_running(run):
    rec = run.daily[run.daily["recorded"]]
    per = rec.groupby("n_on")["perm_flow"].median()
    assert per.loc[2] > per.loc[3]                   # two trains carry the demand of three


def test_the_operator_cleans_on_what_the_screen_shows(run):
    assert len(run.cleanings) > 0
    assert set(run.cleanings["reason"]) <= {"dP on screen", "time since last cleaning", "other reason"}


def test_cycles_never_span_a_cleaning(run):
    for c in sim_cycles(run, 0):
        assert np.all(c.day > 0) and np.all(c.day < c.end - c.start)


def test_the_view_censors_at_the_baseline_like_the_deposit(run):
    v = plant_view(run)
    assert (v["dp_norm_obs"] >= 1.0).all()
    assert (v.loc[v["censored"], "dp_norm_obs"] == 1.0).all()


def test_generate_plants_cycles_the_laws():
    runs = generate_plants(3, seed=9, cfg=ROSimConfig(days=60))
    assert [r.params.law for r in runs] == list(ROSimConfig().laws)
