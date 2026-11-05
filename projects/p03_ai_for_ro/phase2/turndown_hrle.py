"""The same design and turndown sweep, with the high-rejection membrane fitted to the SW30HRLE-440 datasheet."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V
fit = json.loads((Path(__file__).parent / "out" / "hrle_fit.json").read_text())
V.A20, V.B20 = fit["A20_LMH_per_bar"], fit["B20_LMH"]
rows = []
for R in (0.40, 0.35):
    for T in (11.0, 25.0, 36.0):
        for flux in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
            v = V.operate(float(flux), R, 43.0, T, 7)
            if v is None:
                rows.append(dict(n=7, R=R, T=T, flux=flux, production_pct=100 * flux / 14, note="needs > 83 bar"))
                continue
            rows.append(dict(n=7, R=R, T=T, flux=flux, production_pct=100 * flux / 14, P_f=v["P_f"], Q_brine=float(v["Q_brine"]),
                             c_brine=float(v["c_brine"]), ndp_tail=float(v["ndp_tail"]), lead_flux=float(v["lead_flux"]),
                             lead_rec=100 * float(v["lead_recovery"]), c_p=float(v["c_p_mgL"]), sec_ro=v["sec_ro"], note=V.flags(v)))
t = pd.DataFrame(rows)
t.to_csv(Path(__file__).parent / "out" / "turndown_hrle.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
