"""Project 2 - Week 2, part 4: the classifier.

Run:  python projects/p02_control_loops/week2_classifier.py

Trains on simulation, cross-validates by loop, then applies the model ONCE to
the labelled real loops and compares it with the shape-only baseline.
Teaching version: notebook p02_week2_classifier.

Result, recorded in VALIDATION.md 17: the classifier does not beat the
baseline on plant data. Nothing here is re-tuned after seeing that.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import load_isdb, load_sacac
from chemai.models import (baseline_predict, build_model, cross_validate, design_matrix,
                           loop_features, recall_by_visibility, simulation_features,
                           stiction_scores)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week2clf")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week2_classifier.json"

# In a plant the dead time is unknown. The smallest possible delay is used for
# every loop, which understates Harris - see VALIDATION.md 11.4 and 17.2.
PLANT_DELAY = 2
MIN_SAMPLES = 200


def real_features(cfg: ControlLoopConfig) -> pd.DataFrame:
    rows = []

    def add(name, label, tier, loop_type, sp, pv, op):
        if not np.isfinite(pv).all() or len(pv) < MIN_SAMPLES:
            return
        f = loop_features(sp, pv, op, loop_type, PLANT_DELAY, cfg)
        rows.append({"name": name, "label": label, "tier": tier,
                     "loop_type": loop_type, **f.as_row()})

    for r in load_sacac(DATA_DIR / "sacac"):
        if r.label is None:
            continue
        d = r.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        add(r.name, r.label, "stated", r.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.label is None or x.in_sacac:
            continue
        d = x.data
        sp = d.sp.to_numpy() if d.sp.notna().all() else np.zeros(len(d))
        add(x.key, x.label, x.tier, x.loop_type, sp, d.pv.to_numpy(), d.op.to_numpy())
    return pd.DataFrame(rows)


def main() -> dict:
    cfg = ControlLoopConfig()

    sim = simulation_features(cfg)
    log.info("simulated %d runs over %d loops", len(sim), sim.loop.nunique())
    log.info("feature medians by condition\n%s",
             sim.groupby("condition")[["harris", "regularity", "shape"]].median().round(2).to_string())

    oof = cross_validate(sim, build_model())
    log.info("cross-validated on simulation (GroupKFold by loop)\n%s",
             classification_report(sim.condition, oof, digits=2))
    visibility = recall_by_visibility(sim, oof)
    log.info("stiction recall by how far the slip jump stands above the noise\n%s",
             visibility.to_string())

    model = build_model().fit(design_matrix(sim), sim.condition)
    real = real_features(cfg)
    real["prediction"] = model.predict(design_matrix(real))
    stated = real[real.tier == "stated"]
    log.info("real loops, stated labels only\n%s",
             pd.crosstab(stated.label, stated.prediction).to_string())

    # the fair comparison: the loops the baseline was measured on (section 16.2)
    sub = stated[(stated.regularity > cfg.regularity_threshold) & (stated.cycles >= cfg.min_cycles)]
    truth = sub.label.where(sub.label == "stiction", "other")
    scores = {"baseline": stiction_scores(truth, baseline_predict(sub, cfg)),
              "classifier": stiction_scores(truth, sub.prediction)}
    log.info("on the same %d real loops: %s", len(sub), json.dumps(scores))
    verdict = ("classifier beats the baseline" if scores["classifier"]["caught"] >
               scores["baseline"]["caught"] else "classifier does NOT beat the baseline")
    log.info("VERDICT: %s", verdict)

    gap = pd.DataFrame({
        "simulation": sim[["harris", "regularity", "cycles", "shape"]].median(),
        "real": real[["harris", "regularity", "cycles", "shape"]].median()}).round(2)
    log.info("median features - the simulation-to-plant gap\n%s", gap.to_string())

    results = {
        "n_simulated_runs": int(len(sim)), "n_real_labelled": int(len(stated)),
        "n_compared": int(len(sub)),
        "simulation_report": classification_report(sim.condition, oof, output_dict=True,
                                                   zero_division=0),
        "stiction_recall_by_visibility": visibility.to_dict("index"),
        "scores": scores, "verdict": verdict,
        "feature_importances": dict(zip(design_matrix(sim).columns,
                                        model.feature_importances_.round(3))),
        "median_features": gap.to_dict(),
        "plant_delay_assumed": PLANT_DELAY,
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
