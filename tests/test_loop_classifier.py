"""Classifier plumbing - the parts that must not silently break.
The measured result on real loops lives in week2_classifier.py, not here."""
import numpy as np
import pandas as pd

from chemai.config import ControlLoopConfig
from chemai.data import sample_loop_spec, simulate_loop
from chemai.features import delay_samples
from chemai.models import (FEATURES, baseline_predict, build_model, cross_validate,
                           design_matrix, loop_features, recall_by_visibility,
                           simulation_features, stiction_scores)

cfg = ControlLoopConfig()


def _spec(condition, loop_type="F", seed=0):
    return sample_loop_spec(loop_type, condition, np.random.default_rng(seed), cfg)


def test_loop_features_on_a_simulated_loop():
    spec = _spec("stiction")
    d = simulate_loop(spec, 3000, seed=1, cfg=cfg)
    f = loop_features(d.sp, d.pv, d.op, spec.loop_type,
                      delay_samples(spec.theta + spec.ts / spec.substeps, spec.ts), cfg)
    assert 0 <= f.harris <= 1 and f.regularity >= 0 and f.cycles > 0
    assert f.signal == "OP"                                   # flow loop
    assert np.isnan(f.shape) or 0 <= f.shape <= 1


def test_shape_is_missing_when_the_record_is_short():
    """Fewer than cfg.min_cycles periods: no shape is reported at all
    (section 15.1). 200 samples of this loop holds about 9 periods."""
    spec = _spec("healthy")
    short = loop_features(*[simulate_loop(spec, 200, seed=2, cfg=cfg)[c] for c in ("sp", "pv", "op")],
                          spec.loop_type, 2, cfg)
    long = loop_features(*[simulate_loop(spec, 600, seed=2, cfg=cfg)[c] for c in ("sp", "pv", "op")],
                         spec.loop_type, 2, cfg)
    assert short.cycles < cfg.min_cycles and np.isnan(short.shape)
    assert long.cycles > cfg.min_cycles and np.isfinite(long.shape)


def test_design_matrix_flags_missing_shape_instead_of_dropping_rows():
    df = pd.DataFrame({"harris": [0.5, 0.4], "regularity": [2.0, 0.3],
                       "cycles": [40.0, 5.0], "shape": [0.8, np.nan]})
    X = design_matrix(df)
    assert list(X.columns) == FEATURES and len(X) == 2
    assert X.shape_missing.tolist() == [0, 1]
    assert X.shape_value.tolist() == [0.8, 0.5]
    assert np.isfinite(X.to_numpy()).all()


def test_design_matrix_survives_infinities():
    df = pd.DataFrame({"harris": [np.inf], "regularity": [np.inf], "cycles": [np.inf],
                       "shape": [0.7]})
    assert np.isfinite(design_matrix(df).to_numpy()).all()


def test_loop_type_is_never_a_feature():
    """Section 6: all nine real pressure loops are stiction. Feeding the loop
    type would hand the model that shortcut."""
    assert not any("type" in f for f in FEATURES)


def test_baseline_needs_both_a_regular_oscillation_and_a_triangle():
    df = pd.DataFrame({"regularity": [3.0, 3.0, 0.5, 3.0], "shape": [0.8, 0.3, 0.9, np.nan]})
    assert baseline_predict(df, cfg).tolist() == [
        "stiction", "undetermined", "undetermined", "undetermined"]


def test_stiction_scores():
    truth = pd.Series(["stiction", "stiction", "healthy", "stiction"])
    pred = pd.Series(["stiction", "healthy", "stiction", "stiction"])
    assert stiction_scores(truth, pred) == {"caught": 2, "of": 3, "false_alarms": 1,
                                            "precision": 0.67, "recall": 0.67}


def test_recall_by_visibility_splits_buried_from_visible():
    df = pd.DataFrame({"condition": ["stiction"] * 4 + ["healthy"],
                       "slip_to_load": [0.5, 0.8, 3.0, 4.0, np.nan]})
    pred = pd.Series(["healthy", "healthy", "stiction", "stiction", "healthy"])
    out = recall_by_visibility(df, pred)
    assert out.loc["buried", "recall"] == 0.0 and out.loc["visible", "recall"] == 1.0


def test_end_to_end_on_a_small_simulated_set():
    """Runs of one loop must never straddle a split (section 5)."""
    df = simulation_features(cfg, n_loops_per_cell=2, seeds_per_loop=2)
    assert set(df.condition) == set(cfg.conditions)
    pred = cross_validate(df, build_model(), n_splits=2)
    assert len(pred) == len(df) and set(pred) <= set(cfg.conditions)
