"""Step 7 (post-hoc) - do the conclusions survive a simulator that looks more
like the deposit?

Step 3 showed the simulator's cycles are about 2.8x longer than the deposit's and
its day-to-day jumps about 2.2x larger. That comparison was made after the
external test, so nothing here changes a setting: it only asks whether the
step-5 conclusions hold in a quieter world with earlier cleanings.

One variant, written down before it was scored and not tuned afterwards:
day-to-day noise and outliers cut, operators cleaning earlier and more often.
New plants, seed 5005.

    python projects/p03_ro_fouling/step7_matched_simulator.py
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

from chemai.config import RODepositConfig, ROConfig, ROSimConfig
from chemai.data.ro_deposit import download_deposit, load_daily, split_cycles
from chemai.data.ro_sim import generate_plants
from chemai.evaluation.fouling_eval import score_sim, sim_cycles, summarise

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from step3_simulator import cycle_stats, q  # noqa: E402
from step5_forecast_test import curves, precision_at_recall  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"
LEVELS = (1.15, 1.20)
MATCHED = dict(dp_noise_rel=(0.004, 0.012), outlier_probability=0.02, operator_level=(1.10, 1.25),
               max_days_without_cip=(45, 120), min_days_between_cip=14, other_reason_cip_per_year=2.0)


def union_check(sc, horizon: int = 28, margin: float = 0.03) -> dict:
    """Recall and precision of the three plan triggers for 'crosses within 28 days':
    forecast alone (P50 <= 28), margin alone (7-day median within 0.03), and both."""
    g = sc[(sc["truth"] > 0) & (sc["status"] != "insufficient")]
    y = (g["truth"] <= horizon).to_numpy()
    fc = ((g["status"] == "crossing") & (g["p50"].fillna(np.inf) <= horizon)).to_numpy()
    mg = (g["obs_level_now"] >= g["level"] - margin).to_numpy()

    def pr(flag):
        tp, fp, fn = (flag & y).sum(), (flag & ~y).sum(), (~flag & y).sum()
        return dict(recall=round(tp / max(tp + fn, 1), 3), precision=round(tp / max(tp + fp, 1), 3))
    return dict(forecast=pr(fc), margin=pr(mg), union=pr(fc | mg))


def main():
    t0 = time.time()
    sim_cfg = replace(ROSimConfig(), **MATCHED)
    dcfg = RODepositConfig()
    real = [c for c in split_cycles(load_daily(download_deposit(cfg=dcfg) / dcfg.daily_file), dcfg)
            if c.kind == "complete"]
    runs = generate_plants(30, seed=5005, cfg=sim_cfg)
    rl, rs, rn, rj = cycle_stats(real)
    sl, ss, sn, sj = cycle_stats([c for i, r in enumerate(runs) for c in sim_cycles(r, i) if c.closed])
    res = dict(variant=MATCHED, plants=30, seed=5005,
               shape=dict(real=dict(length=q(rl), jump=q(rj), scatter=q(rn), slope=q(rs)),
                          matched=dict(length=q(sl), jump=q(sj), scatter=q(sn), slope=q(ss))))
    cfg = ROConfig()
    cache: dict = {}
    for norm, fe, ve in (("own fit", "fit", None), ("vendor 1.5", 1.5, 0.25)):
        sc = score_sim(runs, LEVELS, cfg.law, cfg, flow_exponent=fe, visc_exponent=ve, cache=cache)
        s = summarise(sc)
        c = curves(sc)
        s["precision_at_recall_0.5"] = dict(forecast=precision_at_recall(c["forecast_p50"], 0.5),
                                            margin=precision_at_recall(c["margin"], 0.5))
        res[norm] = dict(summary=s, curves=c)
        if norm == "own fit":
            res[norm]["coverage_by_law"] = {k: summarise(g)["coverage"] for k, g in sc.groupby("law_true")}
            res["plan_triggers"] = {"matched (own fit)": union_check(sc)}
        print(norm, json.dumps(s), flush=True)
    # the same three triggers on the frozen development and test runs (own fit)
    for name, f in (("development (own fit)", "step4_dev_scores.pkl"), ("test (own fit)", "step5_test_scores.pkl")):
        p = OUT / f
        if p.exists():
            import pandas as pd
            res["plan_triggers"][name] = union_check(pd.read_pickle(p))
    print(json.dumps(res["plan_triggers"]))
    (OUT / "step7_matched_simulator.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["shape"]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
