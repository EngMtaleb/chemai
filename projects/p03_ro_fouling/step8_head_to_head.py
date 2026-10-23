"""Step 8 - the published threshold rule and this project's weekly plan, side by side.

The deposit's authors propose an objective trigger: clean when DP_norm reaches
1.20. It acts when the level is reached. This project's plan tries to say it
in advance (forecast P50 within 28 days, or the last 7 days within 0.03 of the
level). Same real deposit, same weekly decision points as step 6.

    python projects/p03_ro_fouling/step8_head_to_head.py

Descriptive, not a test: 7 crossings of 1.20 is too few to rank anything. The
plan's margin rule was added after step 6 had been run (README), so the
plan's numbers here are optimistic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
MARGIN = 0.03
HORIZON = 28


def head_to_head(cr: pd.DataFrame, level: float) -> dict:
    g = cr[(cr["level"] == level) & (cr["status"] != "insufficient")]
    leads, delays, eligible, cycles_crossing = [], [], 0, 0
    for _, h in g.groupby("cycle"):
        h = h.sort_values("abs_day")
        before = h[(h["truth"] > 0) & (~h["truth_censored"])]
        if (h["truth"] == 0).any():
            cycles_crossing += 1
        if before.empty:
            continue                                 # already above at the first decision, or never crossed
        eligible += 1
        crossing = int((before["abs_day"] + before["truth"]).iloc[0])
        flag = before[((before["status"] == "crossing") & (before["p50"] <= HORIZON))
                      | (before["obs_level_now"] >= level - MARGIN)]
        if len(flag):
            leads.append(crossing - int(flag["abs_day"].iloc[0]))
        fired = h[(h["truth"] == 0) & (h["abs_day"] >= crossing)]
        if len(fired):
            delays.append(int(fired["abs_day"].iloc[0]) - crossing)
    known = g[(g["truth"] > 0) & ((~g["truth_censored"] & (g["truth"] <= HORIZON)) | (g["truth"] > HORIZON))]
    flag = ((known["status"] == "crossing") & (known["p50"] <= HORIZON)) | (known["obs_level_now"] >= level - MARGIN)
    y = (~known["truth_censored"]) & (known["truth"] <= HORIZON)
    return {
        "level": level,
        "cycles_reaching_level": cycles_crossing,
        "crossings_with_a_decision_before": eligible,
        "plan_warned_before": len(leads),
        "plan_lead_days": sorted(leads),
        "plan_median_lead_days": float(np.median(leads)) if leads else None,
        "rule_acts_days_after_crossing": sorted(delays),
        "rule_median_days_after": float(np.median(delays)) if delays else None,
        "plan_flags": int(flag.sum()),
        "plan_false_flags": int((flag & ~y).sum()),
        "plan_precision": round(float((flag & y).sum() / max(flag.sum(), 1)), 3),
        "plan_recall": round(float((flag & y).sum() / max(y.sum(), 1)), 3),
    }


def main():
    cr = pd.read_csv(OUT / "step6_crossings.csv")
    out = {f"{lv:.2f}": head_to_head(cr, lv) for lv in (1.15, 1.20)}
    (OUT / "step8_head_to_head.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
