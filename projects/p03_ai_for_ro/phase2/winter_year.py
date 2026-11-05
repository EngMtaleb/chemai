"""Winter cycle for RO, over whole years (2015-2024): warm the winter feed and see what changes.

A plant beside a power station can mix warm condenser outlet water into its intake. The condenser warms its cooling
water by 5.6-13.9 C (10-25 F; J. P. Guyer, An Introduction to Condensers and Auxiliary Equipment for Steam Power
Plants, 2023, after the US Unified Facilities Criteria). So the feed can be brought up to a target temperature, but
never above sea + the condenser rise. In summer the sea is already warmer than the target: nothing is mixed.

Feed temperature  T_feed = max(T_sea, min(T_target, T_sea + dT_max))
Cases             base (no warming); target 25 C with dT_max 5.6 C (low end) and 13.9 C (high end); target 29 C, 13.9 C
Plants            steady (1x demand, no tank) and follow-the-sun rule (2x demand, 0.8 kWp per m3/day, 12 h tank)
Heat              the heat the warming takes = feed volume x density x heat capacity x (T_feed - T_sea);
                  seawater properties from TEOS-10 (gsw), absolute salinity 43 g/L / 1.03 kg/L

    python winter_year.py
"""
import json
import sys
from pathlib import Path

import gsw
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import schedule as Sc  # noqa: E402

R = 0.40
SA = 43.0 / 1.03                                    # g/kg
CASES = {"base": None, "25C, +5.6": (25.0, 5.6), "25C, +13.9": (25.0, 13.9), "29C, +13.9": (29.0, 13.9)}
WINTER = (12, 1, 2, 3)

perf = Sc.load_perf([HERE / "out" / "perf_table_part0.csv", HERE / "out" / "perf_table_part1.csv"])
other = Sc.other_energy(perf)
cv1, cv2 = Sc.curves(perf, other, 1.0), Sc.curves(perf, other, 2.0)
kwp, S_max = 0.8 * 1000.0, 12 * Sc.DEMAND
rows = []
IDX = pd.to_datetime(pd.read_csv(HERE / "out" / "pv_hourly_per_kwp.csv", index_col=0).index, utc=True).tz_convert("Asia/Kuwait")
for year in range(2015, 2025):
    pv, T_sea = Sc.load_sun_sea(year)
    month = IDX[IDX.year == year].month.to_numpy()
    assert len(month) == len(pv)
    win = np.isin(month, WINTER)
    for case, spec in CASES.items():
        T = T_sea.copy() if spec is None else np.maximum(T_sea, np.minimum(spec[0], T_sea + spec[1]))
        for name, q, cv in (("steady", Sc.run_steady(pv, T, cv1, 0), cv1),
                            ("rule", Sc.run_rule(pv * kwp, T, cv2, S_max), cv2)):
            s = Sc.score(q, pv, T, cv, kwp)
            E = np.array([Sc.energy(Sc.nearest(cv, T[t]), q[t]) for t in range(len(q))])
            dT = T - T_sea
            rho_cp = gsw.rho_t_exact(SA, T_sea, 0) * gsw.cp_t_exact(SA, T_sea, 0)       # J/(m3 K)
            heat_kwh = float(np.sum(q / R * rho_cp * dT) / 3.6e6)
            s.update(year=year, case=case, strategy=name,
                     ro_kWh_per_m3=s["kWh_per_m3"] - other,
                     winter_ro_kWh_per_m3=float(E[win].sum() / q[win].sum() - other),
                     winter_permeate_mgL=float((np.array([Sc.permeate(Sc.nearest(cv, T[t]), q[t]) for t in range(len(q))])[win] * q[win]).sum() / q[win].sum()),
                     feed_T_winter_mean=float(T[win].mean()), sea_T_winter_mean=float(T_sea[win].mean()),
                     warmed_hours=int((dT > 0.05).sum()), heat_kWh_th_per_m3=heat_kwh / s["water_m3"])
            rows.append(s)
    print(year, "done", flush=True)
t = pd.DataFrame(rows)
t.to_csv(HERE / "out" / "winter_year.csv", index=False)
cols = ["kWh_per_m3", "ro_kWh_per_m3", "winter_ro_kWh_per_m3", "solar_share_pct", "grid_kWh_per_m3", "permeate_mgL_avg",
        "permeate_mgL_max", "winter_permeate_mgL", "train_starts_per_day", "feed_T_winter_mean", "sea_T_winter_mean",
        "warmed_hours", "heat_kWh_th_per_m3", "short_hours"]
summary = t.groupby(["strategy", "case"], sort=False)[cols].mean()
print(summary.round(3).T.to_string())
(HERE / "out" / "winter_year_summary.json").write_text(summary.round(4).reset_index().to_json(orient="records", indent=1))
