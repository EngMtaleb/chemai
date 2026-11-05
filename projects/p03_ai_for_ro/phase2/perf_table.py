"""Performance table for scheduling: one train (7 elements, 40 % recovery, high-rejection membrane, 43 g/L) at each
sea temperature and production level -> feed pressure, RO energy, permeate salinity, limits.

    python perf_table.py <part: 0 or 1>
"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V
fit = json.loads((Path(__file__).parent / "out" / "hrle_fit.json").read_text())
V.A20, V.B20 = fit["A20_LMH_per_bar"], fit["B20_LMH"]
temps = [15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0]
part = int(sys.argv[1])
rows = []
for T in temps[part::2]:
    for flux in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        v = V.operate(float(flux), 0.40, 43.0, T, 7)
        if v is None:
            rows.append(dict(T=T, flux=flux, f=flux / 14, note="needs > 83 bar")); continue
        rows.append(dict(T=T, flux=flux, f=flux / 14, P_f=v["P_f"], sec_ro=v["sec_ro"], c_p=float(v["c_p_mgL"]),
                         lead_flux=float(v["lead_flux"]), lead_rec=100 * float(v["lead_recovery"]),
                         Q_brine=float(v["Q_brine"]), ndp_tail=float(v["ndp_tail"]), note=V.flags(v)))
pd.DataFrame(rows).to_csv(Path(__file__).parent / "out" / f"perf_table_part{part}.csv", index=False)
print("done", part)
