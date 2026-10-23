"""Step 7 - size a seawater UF -> RO plant that runs on the sun.

Uses part 2's result (no daylight or restart fouling penalty at design flux) to
plan with modules switched on and off at design flux, so fouling per cubic
metre stays what the pilot measured. Sweeps production capacity, storage and PV
size over a synthetic year; for each design, the best possible hour-by-hour
plan (LP, perfect foresight) and a simple rule with no forecast.

    python projects/p04_tide_fouling/step7_solar_plan.py

Writes results/step7_solar_plan.parquet (every design) and
results/step7_solar_plan.json (the summary).
"""
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.models.solar_plan import SolarPlanConfig, flux_penalty, plan_lp, plan_rule, pv_year

OUT = Path(__file__).parent / "results"
CAPACITY = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
STORAGE_H = (2, 4, 8, 12, 16, 24)
PV_KWP = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
TARGETS = (0.50, 0.75, 0.90, 0.95)
WEATHER = {
    "sunny": SolarPlanConfig(),
    "cloudier": replace(SolarPlanConfig(), cloud=((0.60, 0.85, 1.00), (0.30, 0.45, 0.85), (0.10, 0.15, 0.45))),
}


def sweep(cfg: SolarPlanConfig) -> pd.DataFrame:
    pv = pv_year(cfg)
    rows = []
    for c, s, p in product(CAPACITY, STORAGE_H, PV_KWP):
        if c == 1.0 and s > 4:
            continue                    # capacity 1 cannot use a large tank: it must run all day
        lp = plan_lp(pv, c, s, p, cfg).as_dict()
        rule = plan_rule(pv, c, s, p, cfg)
        lp["rule_solar_fraction"] = rule.solar_fraction
        lp["rule_starts_per_day"] = rule.starts_per_day
        rows.append(lp)
    return pd.DataFrame(rows)


def smallest_design(df: pd.DataFrame, target: float) -> dict | None:
    """Fewest PV kWp, then least capacity, then least storage, reaching the target."""
    ok = df[df.solar_fraction >= target].sort_values(["pv_kwp", "capacity", "storage_h"])
    return None if ok.empty else ok.iloc[0][["capacity", "storage_h", "pv_kwp", "solar_fraction",
                                             "curtailed_share", "run_hours_per_day"]].to_dict()


def main():
    out = {"sec_kwh_m3": SolarPlanConfig().sec, "weather": {}}
    frames = []
    for name, cfg in WEATHER.items():
        df = sweep(cfg)
        df["weather"] = name
        frames.append(df)
        pv = pv_year(cfg)
        out["weather"][name] = {
            "mean_daily_yield_kwh_per_kwp": float(pv.reshape(-1, 24).sum(1).mean()),
            "smallest_design": {f"{t:.0%}": smallest_design(df, t) for t in TARGETS},
            "rule_vs_plan_max_gap": float((df.solar_fraction - df.rule_solar_fraction).max()),
            "rule_vs_plan_median_gap": float((df.solar_fraction - df.rule_solar_fraction).median()),
        }
        print(name, json.dumps(out["weather"][name], indent=1, default=float), flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUT / "step7_solar_plan.parquet")
    out["run_faster_instead"] = {f"capacity {c}x by flux": {f"n={n}": flux_penalty(c, n) for n in (1.2, 1.5, 2.0)}
                                 for c in (1.5, 2.0, 2.5)}
    (OUT / "step7_solar_plan.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["run_faster_instead"], indent=1))


if __name__ == "__main__":
    main()
