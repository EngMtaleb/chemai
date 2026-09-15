"""Loop condition classifier - trained on simulation, tested once on plant data.

Five conditions: healthy, stiction, tuning_tight, tuning_sluggish,
external_oscillation. Three features per loop (Harris index, oscillation
regularity, shape index) plus how much evidence stands behind them.

Design rules, all fixed before any result was seen (VALIDATION.md 17):
  * trained ONLY on simulation, where the label is certain by construction;
  * GroupKFold by simulated loop - runs of one loop never straddle a split;
  * loop type selects which signal the shape is measured on; it is never a
    feature, or the model would learn "pressure means stiction" (section 6);
  * a missing shape (too few cycles) is a FLAG, not a dropped row.

Result, stated plainly: on real loops this model does NOT beat the
single-feature baseline (shape > 0.6). Harris is three times lower on plant
data than in simulation, because the dead time is unknown there - so the model
reads real stiction loops as sluggish tuning. Use `baseline_predict` until the
simulation is closer to plant reality (VALIDATION.md 17.3).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from chemai.config import ControlLoopConfig
from chemai.features import (cycles_in_record, delay_samples, harris_index,
                             oscillation_regularity, prepare_for_shape, shape_index,
                             shape_signal)

log = logging.getLogger(__name__)

FEATURES = ["harris", "regularity", "shape_value", "shape_missing", "cycles"]


@dataclass(frozen=True)
class LoopFeatures:
    harris: float
    regularity: float
    cycles: float
    shape: float          # nan when the record holds too few cycles
    signal: str           # which signal the shape was measured on

    def as_row(self) -> dict:
        return {"harris": self.harris, "regularity": self.regularity, "cycles": self.cycles,
                "shape": self.shape, "signal": self.signal}


def loop_features(sp, pv, op, loop_type: str, delay: int,
                  cfg: ControlLoopConfig | None = None) -> LoopFeatures:
    """The three indices for one loop, with the rules of sections 15-16 applied."""
    cfg = cfg or ControlLoopConfig()
    sp, pv, op = (np.asarray(v, dtype=float) for v in (sp, pv, op))
    period, regularity = oscillation_regularity(detrend(sp - pv))
    cycles = cycles_in_record(len(pv), period)
    signal, which = shape_signal(loop_type, pv, sp, op)
    shape = float("nan")
    if np.isfinite(period) and cycles >= cfg.min_cycles:
        shape = shape_index(prepare_for_shape(signal, period, cfg.shape_harmonics))
    return LoopFeatures(harris=harris_index(sp - pv, delay, cfg.harris_ar_order),
                        regularity=regularity, cycles=cycles, shape=shape, signal=which)


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature table -> model input. A missing shape becomes 0.5 ('neither shape')
    plus an explicit flag, so the model can use the absence as information."""
    out = df.copy()
    out["shape_missing"] = out["shape"].isna().astype(int)
    out["shape_value"] = out["shape"].fillna(0.5)
    return out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def simulation_features(cfg: ControlLoopConfig | None = None, n_loops_per_cell: int = 8,
                        seeds_per_loop: int = 2) -> pd.DataFrame:
    """Generate the simulated dataset and reduce every run to its features."""
    from chemai.data import generate_dataset

    cfg = cfg or ControlLoopConfig()
    specs, runs = generate_dataset(n_loops_per_cell, seeds_per_loop, cfg)
    by_id = {s.loop_id: s for s in specs}
    rows = []
    for (loop_id, seed), d in runs.items():
        s = by_id[loop_id]
        f = loop_features(d.sp, d.pv, d.op, s.loop_type,
                          delay_samples(s.theta + s.ts / s.substeps, s.ts), cfg)
        rows.append({"loop": loop_id, "seed": seed, "condition": s.condition,
                     "loop_type": s.loop_type,
                     "slip_to_load": s.slip_to_load if s.condition == "stiction" else np.nan,
                     **f.as_row()})
    return pd.DataFrame(rows)


def build_model(random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=random_state)


def cross_validate(df: pd.DataFrame, model=None, n_splits: int = 5) -> pd.Series:
    """Out-of-fold predictions, grouped by simulated loop."""
    model = build_model() if model is None else model   # an unfitted forest is falsy
    pred = cross_val_predict(model, design_matrix(df), df.condition,
                             groups=df.loop, cv=GroupKFold(n_splits))
    return pd.Series(pred, index=df.index, name="prediction")


def baseline_predict(df: pd.DataFrame, cfg: ControlLoopConfig | None = None) -> pd.Series:
    """The baseline the classifier has to beat: a regular oscillation with a
    triangular shape is stiction (VALIDATION.md 16.2). Everything else is
    'undetermined' - the baseline answers one question only."""
    cfg = cfg or ControlLoopConfig()
    stiction = (df.regularity > cfg.regularity_threshold) & (df["shape"] > cfg.shape_triangular)
    return pd.Series(np.where(stiction, "stiction", "undetermined"),
                     index=df.index, name="prediction")


def stiction_scores(truth: pd.Series, predicted: pd.Series) -> dict:
    """Caught / false alarms / precision / recall for the stiction class."""
    is_true = truth == "stiction"
    is_pred = predicted == "stiction"
    tp = int((is_true & is_pred).sum())
    fp = int((~is_true & is_pred).sum())
    fn = int((is_true & ~is_pred).sum())
    return {"caught": tp, "of": tp + fn, "false_alarms": fp,
            "precision": round(tp / max(tp + fp, 1), 2),
            "recall": round(tp / max(tp + fn, 1), 2)}


def recall_by_visibility(df: pd.DataFrame, predicted: pd.Series) -> pd.DataFrame:
    """Stiction recall split by how far the slip jump stands above the load noise.

    The answer to the open question of Week 1: buried stiction is not detectable
    from OP and PV, whatever the model. Logging the valve position would be.
    """
    st = df[df.condition == "stiction"].copy()
    st["found"] = predicted[st.index] == "stiction"
    st["visibility"] = pd.cut(st.slip_to_load, [0, 1, 2, np.inf],
                              labels=["buried", "marginal", "visible"])
    return st.groupby("visibility", observed=True).agg(
        loops=("found", "size"), recall=("found", "mean")).round(2)
