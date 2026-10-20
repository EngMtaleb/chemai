"""Step 4 - choosing the forecast on DEVELOPMENT plants only.

Thirty simulated plants (ten per fouling law), seed 1001. Everything chosen here
is frozen before the test plants (step 5) and the real deposit (step 6) are
scored. Run from the repository root:

    python projects/p03_ro_fouling/step4_forecast_dev.py
"""
from dataclasses import replace
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
OUT.mkdir(exist_ok=True)
LEVELS = (1.15, 1.20)
NORMS = {                      # flow exponent, viscosity exponent
    "linear 1.0": (1.0, 0.0),
    "vendor 1.5": (1.5, 0.25),
    "physics 1.75": (1.75, 0.25),
    "own fit": ("fit", None),
}


def decisions(sc: pd.DataFrame, horizon: int = 28) -> dict:
    """Will the train reach the level within `horizon` days? Rules compared at the
    same weekly decision points. Only decisions not already at the level count."""
    g = sc[(sc["truth"] > 0) & (sc["status"] != "insufficient")]
    y = (g["truth"] <= horizon).to_numpy()
    p10 = g["p10"].fillna(np.inf).to_numpy()
    p50 = g["p50"].fillna(np.inf).to_numpy()
    crossing = (g["status"] == "crossing").to_numpy()
    obs = g["obs_level_now"].to_numpy()
    lvl = g["level"].to_numpy()

    def pr(flag):
        tp, fp, fn = (flag & y).sum(), (flag & ~y).sum(), (~flag & y).sum()
        return [round(tp / max(tp + fn, 1), 3), round(tp / max(tp + fp, 1), 3)]
    return dict(
        n=int(len(g)), positive_share=round(float(y.mean()), 3),
        forecast_p50=[[k] + pr(crossing & (p50 <= k)) for k in (7, 14, 21, 28, 35, 42, 56)],
        forecast_p10=[[k] + pr(crossing & (p10 <= k)) for k in (7, 14, 21, 28, 35)],
        margin=[[d] + pr(obs >= lvl - d) for d in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)],
    )


def recall_at_precision(curve: list, target: float) -> float:
    ok = [r for _, r, p in curve if p >= target]
    return max(ok) if ok else 0.0


def main():
    t0 = time.time()
    dev = generate_plants(30, seed=1001)
    cache: dict = {}
    res = {"plants": 30, "seed": 1001, "grid": {}}
    base = ROConfig()

    # 1. law x window x normalisation
    for norm, (fe, ve) in NORMS.items():
        for law in ("linear", "blockage"):
            for win in (28, 42, 63):
                cfg = replace(base, law=law, fit_window_days=win)
                sc = score_sim(dev, LEVELS, law, cfg, flow_exponent=fe, visc_exponent=ve, cache=cache)
                s = summarise(sc)
                dec = decisions(sc)
                s["recall_at_p70"] = dict(forecast=recall_at_precision(dec["forecast_p50"], 0.70),
                                          margin=recall_at_precision(dec["margin"], 0.70))
                res["grid"][f"{norm} | {law} | {win}"] = s
                print(f"{norm:13s} {law:9s} {win:3d}  med err {s['median_abs_err_days']:5}  "
                      f"cov {s['coverage']:.3f}  R@P70 forecast {s['recall_at_p70']['forecast']:.2f} "
                      f"margin {s['recall_at_p70']['margin']:.2f}   ({time.time() - t0:.0f}s)", flush=True)
                if norm in ("vendor 1.5", "physics 1.75", "own fit") and law == "linear" and win == 42:
                    res.setdefault("decisions", {})[norm] = dec
    # 2. interval calibration on the chosen settings (own fit, linear, 42 days)
    res["calibration"] = {}
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        cfg = replace(base, law="linear", fit_window_days=42, interval_inflation=k)
        sc = score_sim(dev, LEVELS, "linear", cfg, flow_exponent="fit", cache=cache)
        s = summarise(sc)
        res["calibration"][k] = s
        print(f"inflation {k}: coverage {s['coverage']:.3f}  flag ratio {s.get('flag_ratio')}", flush=True)
        if k == 1.0:
            sc.to_pickle(OUT / "step4_dev_scores.pkl")
    (OUT / "step4_forecast_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
