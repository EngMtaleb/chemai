"""Design check 2 (before the declaration): where the reading's movement comes from when the gauge is perfect —
the flow normalisation (screen exponent 1.5 vs the physics' 1.70-1.81) and the trains' flow swings. Fingerprint only."""
import dataclasses, json, sys
from pathlib import Path
import numpy as np
SP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SP / "phase3b")); sys.path.insert(0, str(SP / "phase3b" / "poc"))
import sim_v2 as S, fingerprint as F, make_table as MT
from chemai.config import ROConfig, ROSimConfig
sea = S.load_sea_temperature(SP / "solar/data/POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv")
KEYS = ["noise_pp", "mean_abs_change_1d_pp", "mean_abs_change_3d_pp", "mean_abs_change_7d_pp", "lag1_autocorr_of_daily_change"]
PL = ((4101, "linear"), (4102, "biofilm"), (4103, "asymptotic"))
def fp(cfg, ro):
    out = []
    for i, (seed, law) in enumerate(PL):
        tab, cl, *_ = MT.one_plant(i, seed, law, sea, cfg, ro)
        d = tab.rename(columns={"day": "day_index"})[["train", "day_index", "dp_pct"]]
        legs = {tr: sorted(cl[cl.train == tr].day.astype(int).tolist()) for tr in "ABC"}
        v = F.fingerprint(d, legs, boot=False); v = v.get("value", v); out.append([v[k] for k in KEYS])
    return np.array(out).mean(0)
real = json.loads((SP / "phase3b/out/fingerprint_real.json").read_text())["value"]
print("real                         " + " ".join(f"{real[k]:6.2f}" for k in KEYS))
base = dataclasses.replace(ROSimConfig(days=S.N_DAYS), dp_noise_rel=(0.0, 0.0), outlier_probability=0.0)
for fe, ve in [(1.5, 0.25), (1.75, 0.245), (2.0, 0.25)]:
    ro = dataclasses.replace(ROConfig(), vendor_flow_exponent=fe, dp_visc_exponent=ve)
    print(f"perfect gauge, screen exp {fe:4.2f}  " + " ".join(f"{x:6.2f}" for x in fp(base, ro)))
