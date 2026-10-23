"""Step 5 - the frozen forecast on TEST plants (never used to choose anything).

Thirty new simulated plants, seed 2002. The settings are the ones frozen after
step 4: linear law, 42-day window, interval inflation 2.0, flag beyond 2x the
observed window, "plan a CIP" when the P50 date is within 28 days.

    python projects/p03_ro_fouling/step5_forecast_test.py
"""
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from chemai.config import ROConfig
from chemai.data.ro_sim import generate_plants
from chemai.evaluation.fouling_eval import score_sim, summarise

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
LEVELS = (1.15, 1.20)


def curves(sc: pd.DataFrame, horizon: int = 28) -> dict:
    g = sc[(sc["truth"] > 0) & (sc["status"] != "insufficient")]
    y = (g["truth"] <= horizon).to_numpy()
    p50 = g["p50"].fillna(np.inf).to_numpy()
    crossing = (g["status"] == "crossing").to_numpy()
    obs, lvl = g["obs_level_now"].to_numpy(), g["level"].to_numpy()

    def pr(flag):
        tp, fp, fn = (flag & y).sum(), (flag & ~y).sum(), (~flag & y).sum()
        return [round(tp / max(tp + fn, 1), 3), round(tp / max(tp + fp, 1), 3)]
    return dict(n=int(len(g)), cycles=int(g["cycle"].nunique()), positive_share=round(float(y.mean()), 3),
                forecast_p50=[[k] + pr(crossing & (p50 <= k)) for k in (7, 14, 21, 28, 35, 42, 56)],
                margin=[[d] + pr(obs >= lvl - d) for d in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.12)])


def precision_at_recall(curve, recall):
    """Precision of a rule interpolated at a given recall (curves are monotone here)."""
    pts = sorted((r, p) for _, r, p in curve)
    rs, ps = np.array([r for r, _ in pts]), np.array([p for _, p in pts])
    if recall < rs.min() or recall > rs.max():
        return None
    return round(float(np.interp(recall, rs, ps)), 3)


def main():
    t0 = time.time()
    test = generate_plants(30, seed=2002)
    cfg = ROConfig()
    cache: dict = {}
    res = {"plants": 30, "seed": 2002, "frozen": dict(law=cfg.law, window=cfg.fit_window_days,
                                                      inflation=cfg.interval_inflation,
                                                      ratio_max=cfg.extrapolation_ratio_max)}
    for norm, fe, ve in (("vendor 1.5", 1.5, 0.25), ("own fit", "fit", None),
                         ("linear 1.0", 1.0, 0.0), ("physics 1.75", 1.75, 0.25)):
        sc = score_sim(test, LEVELS, cfg.law, cfg, flow_exponent=fe, visc_exponent=ve, cache=cache)
        s = summarise(sc)
        c = curves(sc)
        s["precision_at_recall_0.5"] = dict(forecast=precision_at_recall(c["forecast_p50"], 0.5),
                                            margin=precision_at_recall(c["margin"], 0.5))
        res[norm] = dict(summary=s, curves=c)
        print(f"{norm:12s} {json.dumps(s)}", flush=True)
        if norm == "own fit":
            sc.to_pickle(OUT / "step5_test_scores.pkl")
            res["own fit"]["by_law"] = {law: summarise(g) for law, g in sc.groupby("law_true")}
            g = sc[(sc["status"] == "crossing") & (sc["truth"] > 0) & (sc["truth"] <= 180)].copy()
            g["err"] = (g["p50"] - g["truth"]).abs()
            g["ratio_bin"] = pd.cut(g["ratio"], [0, 0.5, 1, 1.5, 2, 3, 5, 1e9])
            res["own fit"]["error_by_ratio"] = {str(k): [int(len(v)), round(float(v["err"].median()), 1)]
                                                for k, v in g.groupby("ratio_bin", observed=True)}
    # instrumentation: dP from a differential transmitter vs two pressure transmitters
    sc_pt = score_sim(test, LEVELS, cfg.law, cfg, flow_exponent="fit", sensor="pt", cache=cache)
    res["two pressure transmitters"] = dict(summary=summarise(sc_pt), curves=curves(sc_pt))
    print("PT difference", json.dumps(res["two pressure transmitters"]["summary"]), flush=True)
    # secondary, declared before the run: the gap law, same everything else
    from dataclasses import replace
    sc_b = score_sim(test, LEVELS, "blockage", replace(cfg, law="blockage"), flow_exponent="fit", cache=cache)
    res["blockage (secondary)"] = dict(summary=summarise(sc_b), curves=curves(sc_b))
    print("blockage", json.dumps(res["blockage (secondary)"]["summary"]), flush=True)
    (OUT / "step5_forecast_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
