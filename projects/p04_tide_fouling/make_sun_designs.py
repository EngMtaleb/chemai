"""Regenerate chemai/api/sun_designs.py from the step 7 and step 8 results.

    python projects/p04_tide_fouling/make_sun_designs.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
TARGET = HERE.parents[1] / "chemai" / "api" / "sun_designs.py"
COLS = ["weather", "capacity", "storage_h", "pv_kwp", "tidal_kw", "renewable_fraction", "grid_kwh_m3",
        "curtailed_share", "run_hours_per_day", "rule_solar_fraction"]


def main():
    a = pd.read_parquet(HERE / "results" / "step7_solar_plan.parquet").rename(
        columns={"solar_fraction": "renewable_fraction"})
    a["tidal_kw"] = 0.0
    b = pd.read_parquet(HERE / "results" / "step8_sun_and_moon.parquet")
    b = b[b.tidal_kw > 0].assign(weather="sunny", rule_solar_fraction=float("nan"))
    d = pd.concat([a[COLS], b[COLS]], ignore_index=True)
    rows = [(r.weather, float(r.capacity), float(r.storage_h), float(r.pv_kwp), float(r.tidal_kw),
             round(float(r.renewable_fraction), 4), round(float(r.grid_kwh_m3), 4),
             round(float(r.curtailed_share), 4), round(float(r.run_hours_per_day), 2),
             None if pd.isna(r.rule_solar_fraction) else round(float(r.rule_solar_fraction), 4))
            for r in d.itertuples(index=False)]
    head = TARGET.read_text().split("DESIGNS = [")[0]
    TARGET.write_text(head + "DESIGNS = [\n" + "".join(f"    {r!r},\n" for r in rows) + "]\n")
    print(len(rows), "designs written to", TARGET)


if __name__ == "__main__":
    main()
