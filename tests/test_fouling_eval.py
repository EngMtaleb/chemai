"""How forecasts are scored. Two of these were errors in the first version of the
evaluation; they are kept as tests so they cannot come back."""
import numpy as np
import pandas as pd

from chemai.evaluation.fouling_eval import real_decisions, summarise


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = dict(cycle=[f"c{i}" for i in range(n)], in_envelope=[True] * n,
                p10=[np.nan] * n, p90=[np.nan] * n)
    base.update(cols)
    return pd.DataFrame(base)


def test_decisions_already_above_the_level_are_not_scored():
    """A train already above the level needs no forecast. The first summary counted
    those zero-day truths and flattered every method."""
    sc = _rows(status=["crossing", "crossing", "at_or_above"], truth=[10.0, 0.0, 0.0],
               p50=[14.0, 0.0, None], p10=[8.0, 0.0, None], p90=[20.0, 1.0, None])
    s = summarise(sc)
    assert s["forecasts"] == 1 and s["dated"] == 1
    assert s["median_abs_err_days"] == 4.0


def test_an_open_upper_bound_covers_a_late_truth():
    sc = _rows(status=["crossing"], truth=[400.0], p50=[150.0], p10=[90.0], p90=[None])
    assert summarise(sc)["coverage"] == 1.0


def test_a_cleaning_before_the_horizon_leaves_the_answer_unknown():
    """On real data a cleaning can stop a cycle before it crosses. If that happens
    inside the 28-day horizon, nobody knows whether it would have crossed: the
    decision is dropped, not counted as a negative."""
    cr = pd.DataFrame(dict(
        status=["crossing", "not_in_sight", "not_in_sight", "insufficient"],
        truth=[12.0, 9.0, 40.0, 5.0],
        truth_censored=[False, True, True, False]))
    g = real_decisions(cr, horizon=28)
    assert list(g["y"]) == [1, 0]                       # crossed in 12 d; ran 40 d without
    assert 9.0 not in set(g["truth"])                   # cleaned on day 9: unknown
