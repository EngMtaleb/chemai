"""Step 2 - the batch rule on the DEVELOPMENT batches only (ESC and open-loop experiments).

For every batch and every brine price b, the cost curve J(t) the batch would
have had if it had ended at t, the batch's own best end, the rule's end and
the controller's end. The rule's three settings (live-cost window, minimum
elapsed time, which drain sets the fixed cost) were chosen here and frozen in
DECLARED_BEFORE_RUN_A.md. The test batches (dataset 6) are not read.

Also here, because the ESC batches are development data: did the extremum-
seeking controller end its batches while the cost was still falling?

    python projects/p05_when_to_stop/step2_ccro_rule_dev.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import read_dataset
from chemai.features.ccro_batches import COLUMNS, batch_curve, score_batch

OUT = Path(__file__).parent / "results"


def score_datasets(names, cfg, batches):
    rows = []
    for name in names:
        df = read_dataset(name, columns=[c for c in COLUMNS if c != "conductivity_critical"]
                          + ["conductivity_critical"])
        b_ds = batches[batches.dataset == name].set_index(["exp", "file"])
        for (exp, file), g in df.groupby(["exp", "file"], observed=True, sort=False):
            if (exp, file) not in b_ds.index:
                continue
            row = b_ds.loc[(exp, file)]
            if not np.isfinite(row["brine_l"]):
                continue
            curve = batch_curve(g, cfg)
            for b in cfg.weights_b:
                s = score_batch(curve, row, b, cfg)
                s.update({"dataset": name, "exp": exp, "file": file, "seq": int(row["seq"]),
                          "constant_setpoints": bool(row["constant_setpoints"])})
                rows.append(s)
    return pd.DataFrame(rows)


def summarise(r: pd.DataFrame) -> dict:
    ev = r[r.evaluable & r.constant_setpoints]
    inside = ev[ev.inside]
    out = {}
    for b, g in inside.groupby("b"):
        out[f"b={b:g}"] = {
            "batches_inside": int(len(g)),
            "rule_regret_pct": {"median": round(float(g.regret_rule_pct.median()), 3),
                                "p90": round(float(g.regret_rule_pct.quantile(0.9)), 3),
                                "max": round(float(g.regret_rule_pct.max()), 3)},
            "controller_regret_pct": {"median": round(float(g.regret_end_pct.median()), 3),
                                      "p90": round(float(g.regret_end_pct.quantile(0.9)), 3)},
            "rule_stopped_share": round(float(g.rule_stopped.mean()), 3),
            "sqrt_law": {
                "batches": int(g.law_ratio.notna().sum()),
                "median_abs_error_pct": round(float((100 * (g.law_ratio - 1).abs()).median()), 2),
                "ratio_q10_q50_q90": [round(float(g.law_ratio.quantile(q)), 3) for q in (0.1, 0.5, 0.9)],
                "regret_median_pct": round(float(g.law_regret_pct.median()), 3),
            },
            "fixed_end_regret_median_pct": {
                f"{cf} mS/cm": round(float((100 * (g[f"J_c{cf}"] / g.J_best - 1)).median()), 3)
                for cf in (10, 15, 20, 25) if g[f"J_c{cf}"].notna().sum() >= 3
            },
        }
    notin = ev[~ev.inside]
    for b, g in notin.groupby("b"):
        early = 100 * (g.J_rule / g.J_end - 1) > 0.5
        out.setdefault(f"b={b:g}", {})["outside"] = {
            "batches": int(len(g)), "rule_stopped_share": round(float(g.rule_stopped.mean()), 3),
            "stopped_costlier_than_end_by_0.5pct": int(early.sum()),
        }
    return out


def main():
    cfg = CCROConfig()
    batches = pd.read_parquet(OUT / "ccro_batches.parquet")
    r = score_datasets(cfg.development, cfg, batches)
    r.to_parquet(OUT / "step2_dev_scores.parquet")

    # the controller audit on development data: ESC ran with b = 1 (Exp1) and b = 10 (Exp2, Exp3)
    esc_b = {"Exp1": 1.0, "Exp2": 10.0, "Exp2_continued/Data": 10.0, "Exp3": 10.0}
    esc = r[(r.dataset == "dataset3") & r.evaluable]
    esc = esc[esc.apply(lambda x: esc_b.get(x.exp) == x.b, axis=1)]
    esc_audit = esc.groupby("exp").agg(
        b=("b", "first"), batches=("seq", "size"), still_falling_at_end=("falling_at_end", "mean"),
        c_end_median=("c_end", "median"), inside=("inside", "mean")).round(3).reset_index()

    summary = {
        "development": list(cfg.development),
        "settings": {"live_window_s": cfg.live_window_s, "min_batch_min": cfg.min_batch_min,
                     "fixed_cost_from": "the drain that opens the batch's file (the previous batch's brine)",
                     "defaults_first_batch": {"drain_kwh": cfg.default_drain_kwh, "brine_l": cfg.default_brine_l}},
        "rule_on_development": summarise(r),
        "excluded_varying_setpoints": int((~r[r.b == 0].constant_setpoints).sum()),
        "esc_audit": esc_audit.to_dict("records"),
    }
    (OUT / "step2_ccro_rule_dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
