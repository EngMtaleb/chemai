"""Design check before the v15 declaration: how the simulated reading's day-to-day statistics move with the ΔP
transmitter noise alone (outliers off, wander off). No model, no forecast: the fingerprint's noise statistics only."""
import dataclasses, json, sys
from pathlib import Path
import numpy as np, pandas as pd
SP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SP / "phase3b")); sys.path.insert(0, str(SP / "phase3b" / "poc"))
import sim_v2 as S, fingerprint as F, make_table as MT
from chemai.config import ROConfig, ROSimConfig
sea = S.load_sea_temperature(SP / "solar/data/POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv")
KEYS = ["noise_pp", "mean_abs_change_1d_pp", "mean_abs_change_3d_pp", "mean_abs_change_7d_pp", "lag1_autocorr_of_daily_change"]
def fp_for(cfg, seeds=((4101, "linear"), (4102, "biofilm"), (4103, "asymptotic")), V=None):
    if V is not None: MT.V = V
    rows = []
    for i, (seed, law) in enumerate(seeds):
        tab, cl, *_ = MT.one_plant(i, seed, law, sea, cfg, ROConfig())
        d = tab.rename(columns={"day": "day_index"})[["train", "day_index", "dp_pct"]]
        legs = {tr: sorted(cl[cl.train == tr].day.astype(int).tolist()) for tr in "ABC"}
        v = F.fingerprint(d, legs, boot=False); v = v.get("value", v)
        rows.append([v[k] for k in KEYS])
    return np.array(rows)
real = json.loads((SP / "phase3b/out/fingerprint_real.json").read_text())["value"]
print("real            " + " ".join(f"{real[k]:6.2f}" for k in KEYS))
for dn, po in [((0.0, 0.0), 0.0), ((0.008, 0.03), 0.05), ((0.004, 0.012), 0.02), ((0.002, 0.006), 0.0), ((0.004, 0.004), 0.0)]:
    cfg = dataclasses.replace(ROSimConfig(days=S.N_DAYS), dp_noise_rel=dn, outlier_probability=po)
    a = fp_for(cfg)
    print(f"dp {dn} out {po:.2f} " + " ".join(f"{x:6.2f}" for x in a.mean(0)) + "   per plant noise " + " ".join(f"{x:.2f}" for x in a[:, 0]))
