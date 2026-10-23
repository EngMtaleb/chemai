"""Step 5 - Parts B and C: when to backwash and when to clean the UF membrane, by energy alone.

Runs the analysis in DECLARED_BEFORE_RUN_BC.md, once, on the UF pilot record of
Project 4 (doi:10.5068/D1310B, also Zenodo 4630769, CC0).

  B  backwash: each backwash spends filtrate and pump energy (F); between
     backwashes the TMP climbs at the run's fouling rate, so the live energy
     cost per m3 grows (a, per m3 filtered). Energy-only best volume per
     backwash: V* = sqrt(2F/a), against the volume the pilot actually filtered.
  C  chemical cleaning: between cleans the post-backwash resistance climbs;
     the live cost is the extra energy of that resistance at the clean-state
     TMP. The rule, T* and the implied cost of a clean, per interval.

    python projects/p05_when_to_stop/step5_uf_economics.py
"""
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from chemai.data.uf_pilot import read_run, run_root
from chemai.features.uf_cycles import PSI_BAR, cleaning_days, daily_baseline, run_summary
from chemai.models.stopping import interval_economics, sqrt_law

HERE = Path(__file__).parent
OUT = HERE / "results"
P4 = HERE.parent / "p04_tide_fouling" / "results"
DECLARATION = HERE / "DECLARED_BEFORE_RUN_BC.md"

B_MID = {"price": 0.10, "water": 0.05, "eff": 0.7}
B_CORNER = {"price": 0.15, "water": 0.01, "eff": 0.5}       # most in favour of frequent backwash
B_GRID = {"price": (0.05, 0.10, 0.15), "water": (0.01, 0.05, 0.10), "eff": (0.5, 0.7)}
C_MID = {"price": 0.10, "eff": 0.7, "F": 0.3}
C_GRID = {"price": (0.05, 0.10, 0.15), "eff": (0.5, 0.7), "F": (0.05, 0.1, 0.3, 1.0)}


def declaration_commit() -> str:
    root = HERE.parents[1]
    rel = str(DECLARATION.relative_to(root))
    log = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode
    if not log or dirty:
        raise SystemExit(f"{rel} must be committed, unchanged, before this step is run")
    return log


def run_table() -> pd.DataFrame:
    runs = pd.read_parquet(P4 / "runs.parquet")
    seg = pd.read_parquet(P4 / "segments.parquet")
    rate = seg.groupby("run").fouling_pct_h.median()
    root = run_root()
    rows = []
    for r in runs.itertuples():
        s = run_summary(read_run(root / r.split / r.file))
        rows.append({"file": r.file, "year": int(pd.Timestamp(r.t_start).year), **s,
                     "fouling_pct_h": float(rate.get(r.file, np.nan))})
    return pd.DataFrame(rows)


def backwash_economics(t: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Energy-only best volume per backwash against the volume actually filtered."""
    t = t.copy()
    t["v_cycle_m3"] = t.v_in_m3 / t.backwashes
    t["F_usd"] = (t.v_bw_m3 / t.backwashes) * s["water"] + (t.bw_energy_kwh / t.backwashes) / s["eff"] * s["price"]
    # growth of the live energy cost per m3 filtered: price/(36 eff) * dTMP/dV, dTMP/dt = TMP * rate
    t["a_usd_m3_per_m3"] = (s["price"] / (36.0 * s["eff"]) * t.tmp_bar * (t.fouling_pct_h / 100.0) / t.q_m3h)
    t["v_star_m3"] = [sqrt_law(f, a)[0] for f, a in zip(t.F_usd, t.a_usd_m3_per_m3)]
    t["ratio"] = t.v_star_m3 / t.v_cycle_m3
    return t


def chemical_cleaning(daily: pd.DataFrame, cleans: list, s: dict) -> pd.DataFrame:
    rows = []
    for a, b in zip(cleans[:-1], cleans[1:]):
        g = daily[(daily.index >= a) & (daily.index < b)]
        day = (g.index - a).days.to_numpy(float)
        start = g[(day >= 1) & (day <= 7)]
        if len(g) < 20 or start.empty:
            continue
        lnr0, tmp0 = float(start.lnr.median()), float(start.tmp_psi.median()) * PSI_BAR
        keep = day >= 1
        excess_bar = tmp0 * (np.exp(g.lnr.to_numpy(float)[keep] - lnr0) - 1.0)    # at the clean-state flow
        m = s["price"] / (36.0 * s["eff"]) * np.clip(excess_bar, 0, None)          # $/m3
        e = interval_economics(day[keep], m, s["F"], smooth=7, min_t=7)
        rows.append({"from": str(a.date()), "to": str(b.date()), "length_d": float((b - a).days),
                     "logged_days": int(keep.sum()), "tmp_clean_psi": tmp0 / PSI_BAR,
                     "lnr_rise": float(np.median(g.lnr.to_numpy(float)[-7:]) - lnr0),
                     "slope_usd_m3_d": e.slope, "t_star_d": e.t_star, "rule_day": e.rule_time,
                     "implied_F": e.implied_fixed})
    return pd.DataFrame(rows)


def main():
    commit = declaration_commit()
    t = run_table()
    t.to_parquet(OUT / "step5_uf_runs.parquet")
    used = t[(t.backwashes >= 3) & (t.v_in_m3 > 1.0) & (t.fouling_pct_h > 0) & t.tmp_bar.notna()]
    mid, corner = backwash_economics(used, B_MID), backwash_economics(used, B_CORNER)
    grid = []
    for vals in itertools.product(*B_GRID.values()):
        s = dict(zip(B_GRID.keys(), vals))
        grid.append({**s, "median_ratio": float(backwash_economics(used, s).ratio.median())})

    seg = pd.read_parquet(P4 / "segments.parquet")
    daily = daily_baseline(seg)
    cleans = cleaning_days(daily)
    cc = chemical_cleaning(daily, cleans, C_MID)
    cgrid = []
    for vals in itertools.product(*C_GRID.values()):
        s = dict(zip(C_GRID.keys(), vals))
        c = chemical_cleaning(daily, cleans, s)
        cgrid.append({**s, "t_star_over_actual_median": float(np.median(c.t_star_d / c.length_d)) if len(c) else None})
    mid.to_csv(OUT / "step5_backwash_mid.csv", index=False)
    cc.to_csv(OUT / "step5_chemical_cleaning_mid.csv", index=False)
    daily.to_parquet(OUT / "step5_uf_daily.parquet")

    summary = {
        "declaration_commit": commit,
        "runs": int(len(t)), "runs_used_for_backwash": int(len(used)),
        "backwash_share_of_feed_median": round(float((used.v_bw_m3 / used.v_in_m3).median()), 3),
        "B_mid": B_MID, "B_corner": B_CORNER,
        "B1": {"median_ratio": round(float(mid.ratio.median()), 2),
               "q10_q90": [round(float(mid.ratio.quantile(q)), 2) for q in (0.1, 0.9)],
               "pass": bool(mid.ratio.median() >= 5)},
        "B2": {"median_ratio": round(float(corner.ratio.median()), 2), "pass": bool(corner.ratio.median() >= 2)},
        "B_grid_median_ratio": {"min": round(min(g["median_ratio"] for g in grid), 2),
                                "max": round(max(g["median_ratio"] for g in grid), 2)},
        "B_by_year": mid.groupby("year").agg(runs=("ratio", "size"), median_ratio=("ratio", "median"),
                                             v_cycle_m3=("v_cycle_m3", "median"), F_usd=("F_usd", "median"))
                        .round(3).reset_index().to_dict("records"),
        "C_cleaning_days": [str(d.date()) for d in cleans],
        "C_mid": C_MID,
        "C_intervals": cc.round(4).to_dict("records"),
        "C_t_star_over_actual_mid": [round(float(x), 2) for x in (cc.t_star_d / cc.length_d)] if len(cc) else [],
        "C_grid_median": {"min": min(g["t_star_over_actual_median"] for g in cgrid),
                          "max": max(g["t_star_over_actual_median"] for g in cgrid)} if len(cc) else None,
    }
    (OUT / "step5_uf_economics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
