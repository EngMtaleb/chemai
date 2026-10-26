"""Experiment 11 - the fold assignment, computed once (PROTOCOL_exp11.md). Produces no result."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
plant = pd.read_csv(HERE.parents[1] / "results_week8_plant_features.csv")   # written by week8_physics.py
p58 = plant[plant.tier == "stated"].reset_index(drop=True)


def group(name: str) -> str:
    if "stiction-P-oilgas" in name: return "oilgas-stiction"
    if "tuning-F-chemical-DB" in name: return "chemical-DB-tuning"
    if "horch-2003" in name: return "horch-2003"
    if "thornhill-2003" in name: return "thornhill-2003"
    if "stiction-P-chemical-baccidicapaci" in name: return "chemical-stiction-2018"
    if "stiction-L-power" in name: return "power-2018"
    m = re.match(r"(buildings|chemicals|pulpPapers|metals)\.(\d+)$", name)
    if m:
        if m.group(1) == "chemicals":
            return "isdb-chemicals-1-12" if int(m.group(2)) <= 12 else "isdb-chemicals-71-76"
        return "isdb-" + m.group(1)
    raise ValueError(name)


p58["group"] = p58.name.map(group)
p58["fold"] = -1
for k, (_, te) in enumerate(GroupKFold(5).split(p58, groups=p58.group)):
    p58.loc[te, "fold"] = k
assert (p58.fold >= 0).all() and p58.groupby("group").fold.nunique().max() == 1
out = p58[["name", "label", "group", "fold"]]
out.to_csv(HERE / "folds_exp11.csv", index=False)
print(out.groupby("fold").agg(loops=("name", "size"), sticky=("label", lambda s: int((s == "stiction").sum())),
                              groups=("group", lambda s: ", ".join(sorted(set(s))))).to_string())
