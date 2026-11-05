"""Turndown of one design: how far can production drop (same recovery) before a limit is crossed?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import vessel as V
rows = []
for n, R in ((7, 0.40), (7, 0.35), (6, 0.35)):
    for T in (11.0, 25.0, 36.0):
        for flux in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
            v = V.operate(float(flux), R, 43.0, T, n)
            if v is None:
                continue
            rows.append(dict(n=n, R=R, T=T, flux=flux, production_pct=100 * flux / 14, P_f=v["P_f"], Q_brine=float(v["Q_brine"]),
                             ndp_tail=float(v["ndp_tail"]), lead_flux=float(v["lead_flux"]), lead_rec=100 * float(v["lead_recovery"]),
                             c_p=float(v["c_p_mgL"]), sec_ro=v["sec_ro"], note=V.flags(v)))
t = pd.DataFrame(rows)
t.to_csv(Path(__file__).parent / "out" / "turndown.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
