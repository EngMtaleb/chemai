"""Step 3 - the batch rule on the TEST batches (dataset 6, Bayesian optimisation). Run once.

Scores the frozen rule exactly as DECLARED_BEFORE_RUN_A.md says, and refuses
to run unless that file is committed and unchanged.

    python projects/p05_when_to_stop/step3_ccro_rule_test.py
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from step2_ccro_rule_dev import score_datasets

HERE = Path(__file__).parent
OUT = HERE / "results"
DECLARATION = HERE / "DECLARED_BEFORE_RUN_A.md"
BO_B = {"Exp1": 1.0, "Exp2": 10.0, "Exp3A": 10.0, "Exp3B": 10.0}   # the weight each experiment ran with


def declaration_commit() -> str:
    rel = str(DECLARATION.relative_to(HERE.parents[1]))
    root = HERE.parents[1]
    log = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode
    if not log or dirty:
        raise SystemExit(f"{rel} must be committed, unchanged, before the test is run")
    return log


def pct(x):
    return round(float(x), 3)


def t1_t2_t4(r: pd.DataFrame) -> dict:
    out = {}
    for b, g in r.groupby("b"):
        ins, outs = g[g.inside], g[~g.inside]
        d = {"batches": int(len(g)), "inside": int(len(ins))}
        if len(ins) >= 10:
            med, p90 = ins.regret_rule_pct.median(), ins.regret_rule_pct.quantile(0.9)
            d["T1"] = {"median_pct": pct(med), "p90_pct": pct(p90), "max_pct": pct(ins.regret_rule_pct.max()),
                       "pass": bool(med <= 0.5 and p90 <= 1.0)}
            law = ins[ins.law_ratio.notna()]
            mae = (100 * (law.law_ratio - 1).abs()).median()
            lreg = law.law_regret_pct.median()
            d["T4"] = {"batches": int(len(law)), "median_abs_error_pct": pct(mae),
                       "ratio_q10_q50_q90": [pct(law.law_ratio.quantile(q)) for q in (0.1, 0.5, 0.9)],
                       "stop_at_V_star_regret_median_pct": pct(lreg),
                       "pass": bool(mae <= 10.0 and lreg <= 0.5)}
        else:
            d["T1"] = d["T4"] = "too few inside batches"
        if len(outs):
            costlier = 100 * (outs.J_rule / outs.J_end - 1) > 0.5
            d["T2"] = {"batches": int(len(outs)), "rule_fired_share": pct(outs.rule_stopped.mean()),
                       "costlier_than_end_by_0.5pct": int(costlier.sum()),
                       "share": pct(costlier.mean()), "pass": bool(costlier.mean() <= 0.10)}
        out[f"b={b:g}"] = d
    return out


def t3(r: pd.DataFrame) -> dict:
    out = {}
    for exp, b in BO_B.items():
        g = r[(r.exp == exp) & (r.b == b)]
        if g.empty:
            continue
        ins = g[g.inside]
        j_rule = np.where(g.rule_stopped, g.J_rule, g.J_end)        # where the rule would run on: J at the end
        d = {"b": b, "batches": int(len(g)), "inside": int(len(ins)),
             "BO_ended_while_cost_still_falling": pct(g.falling_at_end.mean()),
             "mean_J_BO": pct(g.J_end.mean()), "mean_J_rule_bound": pct(np.mean(j_rule)),
             "rule_vs_BO_mean_J_pct": pct(100 * (np.mean(j_rule) / g.J_end.mean() - 1)),
             "c_end_BO_median": pct(g.c_end.median()), "c_rule_median": pct(g.c_rule.median())}
        if len(ins):
            d["inside_regret_median_pct"] = {"BO": pct(ins.regret_end_pct.median()), "rule": pct(ins.regret_rule_pct.median())}
            d["inside_regret_mean_pct"] = {"BO": pct(ins.regret_end_pct.mean()), "rule": pct(ins.regret_rule_pct.mean())}
            for cf in (10, 15, 20, 25):
                v = ins[f"J_c{cf}"].dropna()
                if len(v):
                    d.setdefault("fixed_end_regret_median_pct", {})[f"{cf} mS/cm"] = {
                        "batches": int(len(v)), "median": pct((100 * (v / ins.loc[v.index, "J_best"] - 1)).median())}
        out[exp] = d
    return out


def main():
    commit = declaration_commit()
    cfg = CCROConfig()
    batches = pd.read_parquet(OUT / "ccro_batches.parquet")
    r = score_datasets(cfg.test, cfg, batches)
    r.to_parquet(OUT / "step3_test_scores.parquet")
    r = r[r.evaluable & r.constant_setpoints]
    summary = {
        "declaration_commit": commit,
        "test": list(cfg.test),
        "batches_scored": int(r[r.b == 0].shape[0]),
        "by_b": t1_t2_t4(r),
        "T3_against_BO": t3(r),
    }
    (OUT / "step3_ccro_rule_test.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
