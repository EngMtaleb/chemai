"""Project 2 - Week 8, part 2: the plant check, run once (PROTOCOL_plant.md).

Run:  python projects/p02_control_loops/plant_check.py
      (after compare_models.py; needs data/sacac/ and data/isdb/isdb10.mat)

Every model is fitted on the simulation development set with the settings chosen
by inner cross-validation on simulation (results_compare_models.json ->
final_params), then applied once to the labelled plant loops. Nothing is
re-tuned after seeing these numbers. Result, recorded in VALIDATION.md 17.4:
no model beats the rule by the margin the protocol asked for.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from chemai.config import ControlLoopConfig
from chemai.models import baseline_predict, build_model, design_matrix

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
HERE = Path(__file__).parent

spec = importlib.util.spec_from_file_location("w2", HERE / "week2_classifier.py")
w2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w2)
spec2 = importlib.util.spec_from_file_location("cmp", HERE / "compare_models.py")
cmp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(cmp)

cfg = ControlLoopConfig()
res = json.load(open(HERE / "results_compare_models.json"))
dev = pd.read_csv(HERE / "results_compare_models_dev_features.csv")
Xd_all, yd = design_matrix(dev), dev.condition.values

real = w2.real_features(cfg)
stated = real[real.tier == "stated"].reset_index(drop=True)
sub = stated[(stated.regularity > cfg.regularity_threshold) & (stated.cycles >= cfg.min_cycles)].reset_index(drop=True)
print(f"real loops with features: {len(real)} · stated labels: {len(stated)} · the episode's comparison set: {len(sub)}")
print("labels in the comparison set:", sub.label.value_counts().to_dict())


def scores(df, pred):
    truth = np.where(df.label.values == "stiction", "stiction", "other")
    return cmp.stiction_binary(truth, pred)


out = {"n_real": int(len(real)), "n_stated": int(len(stated)), "n_compared": int(len(sub)),
       "labels_compared": sub.label.value_counts().to_dict(), "labels_stated": stated.label.value_counts().to_dict(), "models": {}}
out["models"]["rule"] = {"compared": scores(sub, baseline_predict(sub, cfg).values),
                         "all_stated": scores(stated, baseline_predict(stated, cfg).values)}
pub = clone(build_model()).fit(Xd_all, yd)
out["models"]["random_forest_published"] = {"compared": scores(sub, pub.predict(design_matrix(sub))),
                                            "all_stated": scores(stated, pub.predict(design_matrix(stated)))}
for name, (est, grid, feats) in cmp.MODELS.items():
    # restore the real types the grid used (int / float / None)
    fixed = {}
    for k, v in res["models"][name]["final_params"].items():
        cands = [g for g in grid[k] if str(g) == v]
        fixed[k] = cands[0]
    m = clone(est).set_params(**fixed).fit(Xd_all[feats], yd)
    out["models"][name] = {"params": {k: str(v) for k, v in fixed.items()},
                           "compared": scores(sub, m.predict(design_matrix(sub)[feats])),
                           "all_stated": scores(stated, m.predict(design_matrix(stated)[feats]))}
    if name == "random_forest":
        out["confusion_compared"] = pd.crosstab(sub.label, m.predict(design_matrix(sub)[feats])).to_dict()

for k, v in out["models"].items():
    c, a = v["compared"], v["all_stated"]
    print(f"{k:26s} 22 loops: caught {c['caught']}/{c['of']} FA {c['false_alarms']} P {c['precision']} R {c['recall']} F1 {c['f1']}"
          f"  | all stated ({len(stated)}): caught {a['caught']}/{a['of']} FA {a['false_alarms']} F1 {a['f1']}")
out["median_features_plant"] = real[["harris", "regularity", "cycles", "shape"]].median().round(3).to_dict()
out["median_features_sim"] = dev[["harris", "regularity", "cycles", "shape"]].median().round(3).to_dict()
json.dump(out, open(HERE / "results_plant_check.json", "w"), indent=2, default=str)
print("saved")
