"""Phase 3 addition - descriptive diagnostics for the report (not part of the declared test; no verdict uses them).

    python3 -B diag.py           (writes diag.txt and seed_spread.csv next to this file)

- the networks' early stopping: E* (best epoch) and the stopping epoch, by network, test and indicator;
- run-to-run spread: the skill each single seed would have had (the declared forecast is the mean of the five seeds);
- Prophet: fits, failures, time.
"""
import json

import numpy as np
import pandas as pd

import p3add as A
L = A.L
lines = []
say = lambda s="": (print(s), lines.append(s))

log = pd.read_csv(A.HERE / "nets_log.csv", keep_default_na=False, na_values=[""])
say("== networks: early stopping (E* = epoch with the lowest validation MAE; 150 = cap)")
g = log.groupby(["method", "test"])
say(pd.DataFrame({"fits": g.size(), "E* median": g.best_epoch.median(), "E* min": g.best_epoch.min(),
                  "E* max": g.best_epoch.max(), "E* <= 2 (share %)": g.best_epoch.apply(lambda s: 100 * (s <= 2).mean()).round(1),
                  "stopped at 150": g.stopped_epoch.apply(lambda s: int((s >= A.MAX_EPOCHS).sum()))}).to_string())
say()
say("E* by indicator and horizon (median over folds and seeds):")
say(log.pivot_table(index=["method", "target"], columns="h", values="best_epoch", aggfunc="median").to_string())
say()

seeds = pd.read_csv(A.HERE / "pred_nets_seeds.csv", keep_default_na=False, na_values=[""])
rows = []
for test_name in ("rolling", "unseen_train"):
    p = pd.read_csv(A.HERE / f"pred_nets_{test_name}.csv", keep_default_na=False, na_values=[""])
    s = seeds[seeds.test == test_name].merge(p[["target", "h", "block", "train", "t", "now", "y"]],
                                             on=["target", "h", "block", "train", "t"], validate="m:1")
    for (c, h, meth, seed), gg in s.groupby(["target", "h", "method", "seed"]):
        f = np.clip(gg.now + gg.change, 0, 1)
        mae, base = np.mean(np.abs(gg.y - f)), np.mean(np.abs(gg.y - gg.now))
        rows.append({"test": test_name, "target": c, "h": h, "method": meth, "seed": seed, "n": len(gg),
                     "skill_pct": 100 * (1 - mae / base)})
sp = pd.DataFrame(rows)
sp.to_csv(A.HERE / "seed_spread.csv", index=False)
say("== run-to-run spread: pooled skill % of each single seed, min .. max over the five seeds (descriptive)")
t = sp.groupby(["test", "target", "method", "h"]).skill_pct.agg(["min", "max"]).round(1)
t["range"] = t["min"].map("{:+.1f}".format) + " .. " + t["max"].map("{:+.1f}".format)
say(t["range"].unstack("h").to_string())
say()

pl = json.loads((A.HERE / "prophet_log.json").read_text())
say(f"== Prophet: {pl['origins']} origins, {pl['fits']} fits, {pl['failed_fits']} failed (fell back to no change); "
    f"{pl['seconds_fitting']} s fitting, {pl['seconds_total']} s in all; prophet {pl['prophet_version']}, "
    f"pandas {pl['pandas']}, numpy {pl['numpy']}")
(A.HERE / "diag.txt").write_text("\n".join(lines) + "\n")
