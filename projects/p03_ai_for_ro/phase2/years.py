"""One design, every year 2015-2024, every strategy: how much the answer moves from year to year.

    python years.py <capacity> <kWp per m3/day> <tank hours>
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import schedule as Sc  # noqa: E402

C, P, S_h = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
perf = Sc.load_perf([HERE / "out" / "perf_table_part0.csv", HERE / "out" / "perf_table_part1.csv"])
other = Sc.other_energy(perf)
cv1, cv = Sc.curves(perf, other, 1.0), Sc.curves(perf, other, C)
rows = []
for year in range(2015, 2025):
    pv, T = Sc.load_sun_sea(year)
    kwp, S_max = P * 1000.0, S_h * Sc.DEMAND
    for name, fn, curve in (("steady", lambda: Sc.run_steady(pv, T, cv1, 0), cv1),
                            ("rule", lambda: Sc.run_rule(pv * kwp, T, cv, S_max), cv),
                            ("plan-fcst", lambda: Sc.run_plan(pv * kwp, T, cv, S_max, known=False), cv),
                            ("plan-known", lambda: Sc.run_plan(pv * kwp, T, cv, S_max, known=True), cv)):
        s = Sc.score(fn(), pv, T, curve, kwp)
        s.update(year=year, strategy=name)
        rows.append(s)
    print(year, "done", flush=True)
t = pd.DataFrame(rows)
t.to_csv(HERE / "out" / f"years_C{C}_P{P}_S{S_h}.csv", index=False)
cols = ["kWh_per_m3", "solar_share_pct", "grid_kWh_per_m3", "curtailed_pct", "permeate_mgL_avg", "train_starts_per_day", "short_hours"]
print(t.groupby("strategy")[cols].agg(["mean", "min", "max"]).round(2).T.to_string())
