"""Phase 2, step 1 - the Kuwait sun, hour by hour, 2015-2024 (NASA POWER), and what one kWp of panels gives.

PV model (declared): fixed panels, tilt 29 deg (= latitude), facing south; Hay-Davies sky model; SAPM cell temperature
(open rack, glass/polymer) with the 10 m wind; PVWatts DC model, power temperature coefficient -0.4 %/C (pvlib's
typical range -0.2 to -0.5); 14 % system losses (PVWatts default); inverter 96 % (PVWatts default), DC/AC 1.1.
NASA POWER hour HR covers HR:00-HR+1:00 UTC (checked against sunrise and sunset), so the sun is placed at HR:30.

    python sun_step1.py <POWER csv> <out folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LAT, LON, TILT, AZ, ALBEDO = 29.37, 47.98, 29.0, 180.0, 0.25
GAMMA, LOSSES, ETA_INV, DC_AC = -0.004, 0.14, 0.96, 1.1


def read_power(src):
    head = 0
    with open(src) as f:
        for i, line in enumerate(f):
            if line.startswith("-END HEADER-"):
                head = i + 1
                break
    d = pd.read_csv(src, skiprows=head)
    d.index = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY, hour=d.HR), utc=True)
    return d.rename(columns={"ALLSKY_SFC_SW_DWN": "ghi", "ALLSKY_SFC_SW_DNI": "dni", "ALLSKY_SFC_SW_DIFF": "dhi",
                             "CLRSKY_SFC_SW_DWN": "ghi_clear", "T2M": "t_air", "WS10M": "wind"})


def pv_per_kwp(d):
    """Hourly AC output of 1 kWp (kW, = kWh in the hour), plus the DC output and the DC output at 25 C cells."""
    sp = pvlib.solarposition.get_solarposition(d.index + pd.Timedelta(minutes=30), LAT, LON)
    sp.index = d.index
    poa = pvlib.irradiance.get_total_irradiance(TILT, AZ, sp.apparent_zenith, sp.azimuth, d.dni, d.ghi, d.dhi,
                                                dni_extra=pvlib.irradiance.get_extra_radiation(d.index),
                                                albedo=ALBEDO, model="haydavies")
    poa_g = poa["poa_global"].fillna(0).clip(lower=0)
    tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = pvlib.temperature.sapm_cell(poa_g, d.t_air, d.wind, **tp)
    pdc = pvlib.pvsystem.pvwatts_dc(poa_g, t_cell, pdc0=1.0, gamma_pdc=GAMMA) * (1 - LOSSES)
    pac = pvlib.inverter.pvwatts(pdc, pdc0=1.0 / DC_AC / ETA_INV, eta_inv_nom=ETA_INV).clip(lower=0).fillna(0)
    pdc_25 = pvlib.pvsystem.pvwatts_dc(poa_g, pd.Series(25.0, index=d.index), pdc0=1.0, gamma_pdc=GAMMA) * (1 - LOSSES)
    h = pd.DataFrame({"ghi": d.ghi / 1000, "ghi_clear": d.ghi_clear / 1000, "poa": poa_g / 1000, "pv": pac,
                      "pv_dc": pdc, "pv_dc_25C": pdc_25, "t_air": d.t_air, "t_cell": t_cell})
    h.index = h.index.tz_convert("Asia/Kuwait")                 # local clock, UTC+3
    return h


def main():
    d = read_power(SRC)
    missing = {k: int(v) for k, v in (d[["ghi", "dni", "dhi", "ghi_clear", "t_air", "wind"]] <= -999).sum().items()}
    lit = d.loc["2024-06-21"]
    lit = lit[lit.ghi > 0]
    h = pv_per_kwp(d)
    daily = h.resample("D").agg({"ghi": "sum", "ghi_clear": "sum", "poa": "sum", "pv": "sum", "pv_dc": "sum",
                                 "pv_dc_25C": "sum", "t_air": "max"})
    daily = daily[(daily.index.year >= 2015) & (daily.index.year <= 2024)]
    daily["month"] = daily.index.month
    monthly = daily.groupby("month").agg(ghi=("ghi", "mean"), ghi_clear=("ghi_clear", "mean"), pv=("pv", "mean"),
                                         pv_min_year=("pv", lambda s: s.groupby(s.index.year).mean().min()),
                                         pv_max_year=("pv", lambda s: s.groupby(s.index.year).mean().max()),
                                         t_max=("t_air", "mean"))
    monthly["cloud_loss_pct"] = 100 * (1 - monthly.ghi / monthly.ghi_clear)
    monthly["heat_loss_pct"] = daily.groupby("month").apply(lambda g: 100 * (1 - g.pv_dc.sum() / g.pv_dc_25C.sum()),
                                                            include_groups=False)
    yearly = daily.groupby(daily.index.year).pv.sum()
    med = daily.groupby("month").pv.transform("median")
    bad = daily.pv < 0.5 * med
    res = {"rows": int(len(d)), "missing": missing,
           "stamp_check_2024_06_21": {"first_sunlit_hour_UTC": int(lit.HR.iloc[0]), "last_sunlit_hour_UTC": int(lit.HR.iloc[-1])},
           "annual_kwh_per_kwp": {"mean": float(yearly.mean()), "min": float(yearly.min()), "max": float(yearly.max()),
                                  "by_year": {int(k): round(float(v), 1) for k, v in yearly.items()}},
           "daily_kwh_per_kwp": {"mean": float(daily.pv.mean()),
                                 "summer_JJA": float(daily[daily.month.isin([6, 7, 8])].pv.mean()),
                                 "winter_DJF": float(daily[daily.month.isin([12, 1, 2])].pv.mean())},
           "ghi_kwh_m2_day_mean": float(daily.ghi.mean()),
           "cloud_loss_pct_year": float(100 * (1 - daily.ghi.sum() / daily.ghi_clear.sum())),
           "heat_loss_pct_year": float(100 * (1 - daily.pv_dc.sum() / daily.pv_dc_25C.sum())),
           "bad_days_per_year": float(bad.sum() / 10),
           "bad_days_by_month": daily[bad].groupby("month").size().div(10).round(1).to_dict(),
           "longest_bad_run_days": int(bad.astype(int).groupby((~bad).cumsum()).sum().max()),
           "worst_day": {"date": str(daily.pv.idxmin().date()), "kwh_per_kwp": float(daily.pv.min())},
           "monthly": monthly.round(3).reset_index().to_dict("records")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sun_step1.json").write_text(json.dumps(res, indent=1, default=str))
    h.to_csv(OUT / "pv_hourly_per_kwp.csv")
    daily.to_csv(OUT / "pv_daily_per_kwp.csv")
    print(json.dumps({k: v for k, v in res.items() if k != "monthly"}, indent=1, default=str))
    print(monthly.round(2).to_string())


if __name__ == "__main__":
    main()
