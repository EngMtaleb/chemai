"""Step 4 - Part D: when to clean an RO train, judged by energy alone.

Runs exactly the analysis in DECLARED_BEFORE_RUN_D.md, once. For each complete
cleaning cycle of the RO plant deposit (doi:10.5281/zenodo.21055948, the data of
Project 3), the live cost of fouling is the extra energy the train spends to
hold its production; ending the cycle costs a cleaning. The rule, the
square-root law and the implied cleaning cost are computed on a declared grid
of prices and plant constants.

    python projects/p05_when_to_stop/step4_ro_cip_economics.py
"""
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chemai.data.ro_deposit import load_daily, load_ledger, recovery_by_campaign, split_cycles
from chemai.features.ro_physics import fouling_energy_penalty, specific_energy_per_bar
from chemai.models.stopping import interval_economics

HERE = Path(__file__).parent
OUT = HERE / "results"
DECLARATION = HERE / "DECLARED_BEFORE_RUN_D.md"

MID = {"price": 0.10, "ndp0": 20.0, "dp0": 2.0, "erd": True, "F": 1.0}
CORNER = {"price": 0.15, "ndp0": 25.0, "dp0": 3.0, "erd": False, "F": 0.5}   # most in favour of cleaning
GRID = {"price": (0.05, 0.10, 0.15), "ndp0": (15.0, 20.0, 25.0), "dp0": (1.0, 2.0, 3.0),
        "erd": (True, False), "F": (0.5, 1.0, 2.0, 5.0)}
SKIP_DAYS = 3        # flushing transient after a cleaning
SMOOTH_DAYS = 7      # trailing median of the live cost
MIN_DAYS = 7         # the rule does not fire in a cycle's first week


def declaration_commit() -> str:
    root = HERE.parents[1]
    rel = str(DECLARATION.relative_to(root))
    log = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode
    if not log or dirty:
        raise SystemExit(f"{rel} must be committed, unchanged, before this step is run")
    return log


def cycle_inputs(daily: pd.DataFrame):
    """Complete cycles: days since the opening cleaning, ndp_pct, and dp_pct above its start."""
    out = []
    for c in split_cycles(daily):
        if c.kind != "complete":
            continue
        g = (daily[(daily.train == c.train) & daily.day_index.isin(c.abs_day)]
             .set_index("day_index").loc[c.abs_day])
        day = c.day
        keep = day >= SKIP_DAYS
        start = (day >= SKIP_DAYS) & (day <= SKIP_DAYS + 7)
        dp_start = float(np.median(g.dp_pct.to_numpy()[start])) if start.any() else float(g.dp_pct.iloc[0])
        dpn = pd.Series(g.dp_norm.to_numpy(), index=day).rolling(7, min_periods=1).median()
        hit = dpn[dpn >= 1.20]
        out.append({"id": c.id, "train": c.train, "length": float(c.length),
                    "day": day[keep], "ndp": g.ndp_pct.to_numpy()[keep],
                    "d_dp": g.dp_pct.to_numpy()[keep] - dp_start,
                    "trigger_120_day": float(hit.index[0]) if len(hit) else None,
                    "dp_start": dp_start,
                    "ndp_end": float(np.median(g.ndp_pct.to_numpy()[-7:])),
                    "dp_rise_end": float(np.median(g.dp_pct.to_numpy()[-7:]) - dp_start)})
    return out


def judge(cycles, s) -> pd.DataFrame:
    kwh_bar = specific_energy_per_bar(erd_eff=0.95 if s["erd"] else 0.0)
    rows = []
    for c in cycles:
        m = s["price"] * fouling_energy_penalty(c["ndp"], c["d_dp"], s["ndp0"], s["dp0"], kwh_bar)   # $/m3
        e = interval_economics(c["day"], m, s["F"], smooth=SMOOTH_DAYS, min_t=MIN_DAYS)
        rows.append({"id": c["id"], "train": c["train"], "length_d": c["length"], "slope_usd_m3_d": e.slope,
                     "t_star_d": e.t_star, "rule_day": e.rule_time,
                     "rule_before_operators": e.rule_time is not None,
                     "implied_F": e.implied_fixed, "penalty_end_usd_m3": float(np.median(m[-7:])),
                     "trigger_120_day": c["trigger_120_day"]})
    return pd.DataFrame(rows)


def ratio_summary(j: pd.DataFrame) -> dict:
    """T*/T_actual: cycles whose penalty never rose have an infinite T* and are counted apart."""
    r = j.t_star_d / j.length_d
    fin = r[np.isfinite(r)]
    return {"no_rise_share": round(float((~np.isfinite(r)).mean()), 3), "finite": int(len(fin)),
            "q10_q50_q90_of_finite": [round(float(fin.quantile(q)), 2) for q in (0.1, 0.5, 0.9)] if len(fin) else None}


def main():
    commit = declaration_commit()
    daily, ledger = load_daily(), load_ledger()
    cycles = cycle_inputs(daily)
    mid, corner = judge(cycles, MID), judge(cycles, CORNER)
    mid.to_csv(OUT / "step4_cycles_mid.csv", index=False)
    corner.to_csv(OUT / "step4_cycles_corner.csv", index=False)

    # the whole grid: how many cycles the rule would have cleaned before the operators did
    grid = []
    for vals in itertools.product(*GRID.values()):
        s = dict(zip(GRID.keys(), vals))
        j = judge(cycles, s)
        grid.append({**s, "rule_before_operators": int(j.rule_before_operators.sum()),
                     "t_star_over_actual_median": float(np.nanmedian(j.t_star_d / j.length_d)),
                     "implied_F_median": float(np.nanmedian(j.implied_F))})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "step4_grid.csv", index=False)

    # waiting longer -> less recovered? (D4) pressure drop left after the closing campaign
    rec = recovery_by_campaign(daily, ledger)
    lengths, left = [], []
    for c in cycles:
        train, end = c["train"], int(c["id"].split(":")[1].split("-")[1])
        r = rec[(rec.train == train) & (rec.day >= end) & (rec.day <= end + 7) & rec.evaluable]
        if len(r):
            lengths.append(c["length"])
            left.append(float(r.post.iloc[-1]) - c["dp_start"])     # pressure drop the cleaning did not remove
    rho, p = spearmanr(lengths, left) if len(lengths) >= 5 else (np.nan, np.nan)

    n = len(cycles)
    summary = {
        "declaration_commit": commit,
        "complete_cycles": n,
        "mid_case": MID, "corner_case": CORNER,
        "kwh_per_bar": {"with_erd": round(specific_energy_per_bar(), 4),
                        "without_erd": round(specific_energy_per_bar(erd_eff=0.0), 4)},
        "D1": {"rule_before_operators": int(mid.rule_before_operators.sum()), "of": n,
               "pass": bool(mid.rule_before_operators.sum() <= 4)},
        "D2": {"implied_F_median_usd_per_m3d": round(float(np.nanmedian(mid.implied_F)), 4),
               "pass": bool(np.nanmedian(mid.implied_F) < 0.25)},
        "D3": {"rule_before_operators": int(corner.rule_before_operators.sum()), "of": n,
               "pass": bool(corner.rule_before_operators.sum() <= 7)},
        "D4": {"cycles": len(lengths), "spearman_rho": None if np.isnan(rho) else round(float(rho), 3),
               "p": None if np.isnan(p) else round(float(p), 3)},
        "mid_t_star_over_actual": ratio_summary(mid),
        "corner_t_star_over_actual": ratio_summary(corner),
        "grid_rule_before_operators": {"min": int(grid.rule_before_operators.min()),
                                       "median": float(grid.rule_before_operators.median()),
                                       "max": int(grid.rule_before_operators.max()), "cases": int(len(grid))},
        "trigger_120_reached_in": int(sum(c["trigger_120_day"] is not None for c in cycles)),
    }
    (OUT / "step4_ro_cip_economics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(mid.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
