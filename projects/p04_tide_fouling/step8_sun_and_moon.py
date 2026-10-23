"""Step 8 - the moon as a power source: a tidal turbine beside the PV.

Part 1 found the tide does not foul the membrane; part 2 that the membrane
does not mind when it runs. So the plant can follow any renewable supply. A
tidal-stream turbine delivers at night and on cloudy days, on a timetable known
years ahead. How much tank and capacity does a little of it save?

    python projects/p04_tide_fouling/step8_sun_and_moon.py
"""
import json
from itertools import product
from pathlib import Path

import pandas as pd

from chemai.models.solar_plan import SolarPlanConfig, plan_lp, pv_year, tidal_year

OUT = Path(__file__).parent / "results"
TARGETS = (0.90, 0.95, 0.99)
TIDAL_KW = (0.0, 0.05, 0.10, 0.15)       # kW rated per m3/day of demand


def main():
    cfg = SolarPlanConfig()
    pv, tide = pv_year(cfg), tidal_year()
    rows = []
    for c, s, p, k in product((1.5, 2.0, 2.5, 3.0), (4, 8, 12, 16, 24), (0.3, 0.5, 0.7, 0.9), TIDAL_KW):
        r = plan_lp(pv, c, s, p, cfg, other=tide * k).as_dict()
        r["tidal_kw"] = k
        rows.append(r)
    df = pd.DataFrame(rows).rename(columns={"solar_fraction": "renewable_fraction"})
    df.to_parquet(OUT / "step8_sun_and_moon.parquet")
    out = {"tidal_capacity_factor": float(tide.mean()), "smallest_tank": {}}
    for t, k in product(TARGETS, TIDAL_KW):
        ok = df[(df.tidal_kw == k) & (df.renewable_fraction >= t)].sort_values(["storage_h", "capacity", "pv_kwp"])
        out["smallest_tank"][f"{t:.0%} with tidal {k:.2f} kW"] = (
            None if ok.empty else ok.iloc[0][["capacity", "storage_h", "pv_kwp", "renewable_fraction"]].to_dict())
    (OUT / "step8_sun_and_moon.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
