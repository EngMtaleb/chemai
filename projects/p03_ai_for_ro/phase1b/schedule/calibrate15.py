"""declaration_v15.md, calibration: x (transmitter noise), p (outliers), w (slow movement) against the real record's
fingerprint; screen flow exponent 1.75; nine calibration plants never used again. No model, no forecast."""
import dataclasses, itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
SP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SP / "phase3b")); sys.path.insert(0, str(SP / "phase3b" / "poc"))
import sim_v2 as S, fingerprint as F, make_table as MT
from chemai.config import ROConfig, ROSimConfig
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
sea = S.load_sea_temperature(SP / "solar/data/POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv")
KEYS = ["noise_pp", "mean_abs_change_1d_pp", "mean_abs_change_3d_pp", "mean_abs_change_7d_pp", "lag1_autocorr_of_daily_change"]
OTHER = ["cycle_length_days", "share_of_cycles_over_60_days", "rise_pp_per_week", "cip_drop_pp", "level_at_cleaning_pp",
         "mean_abs_change_14d_pp", "level_p10_pp", "level_p50_pp", "level_p90_pp"]
CAL = [(4301, "linear"), (4302, "linear"), (4303, "linear"), (4311, "biofilm"), (4312, "biofilm"), (4313, "biofilm"),
       (4321, "asymptotic"), (4322, "asymptotic"), (4323, "asymptotic")]
RO = dataclasses.replace(ROConfig(), vendor_flow_exponent=1.75)
real = json.loads((SP / "phase3b/out/fingerprint_real.json").read_text())
R, I = real["value"], real["interval_90"]
HW = {k: (I[k][1] - I[k][0]) / 2 for k in KEYS}


def setting(x, p, w):
    return dataclasses.replace(ROSimConfig(days=S.N_DAYS), dp_noise_rel=(0.5 * x, 1.5 * x), outlier_probability=p), \
        S.V2(sigma_z=0.75, wander_sd=w)


def score_plants(cfg, v2, plants):
    MT.V = v2
    rows = []
    for i, (seed, law) in enumerate(plants):
        tab, cl, *_ = MT.one_plant(i, seed, law, sea, cfg, RO)
        d = tab.rename(columns={"day": "day_index"})[["train", "day_index", "dp_pct"]]
        legs = {tr: sorted(cl[cl.train == tr].day.astype(int).tolist()) for tr in "ABC"}
        v = F.fingerprint(d, legs, boot=False); v = v.get("value", v)
        rows.append({k: v[k] for k in KEYS + OTHER})
    return pd.DataFrame(rows).mean(numeric_only=True)


if __name__ == "__main__":
    t0 = time.time()
    out = []
    for x, p, w in itertools.product([0.002, 0.003, 0.004, 0.005, 0.006, 0.008], [0.0, 0.02, 0.05], [0.0, 0.005, 0.01, 0.015, 0.02]):
        cfg, v2 = setting(x, p, w)
        m = score_plants(cfg, v2, CAL)
        obj = float(sum(((m[k] - R[k]) / HW[k]) ** 2 for k in KEYS))
        out.append(dict(x=x, p=p, w=w, objective=obj, inside=int(sum(I[k][0] <= m[k] <= I[k][1] for k in KEYS)), **m.to_dict()))
    df = pd.DataFrame(out).sort_values("objective").reset_index(drop=True)
    df.to_csv(OUT / "calibration15.csv", index=False)
    best = df.iloc[0]
    res = {"chosen": {"x": float(best.x), "p": float(best.p), "w": float(best.w), "objective": float(best.objective)},
           "five": {k: {"sim": round(float(best[k]), 3), "real": round(R[k], 3), "interval": [round(I[k][0], 3), round(I[k][1], 3)],
                        "inside": bool(I[k][0] <= best[k] <= I[k][1])} for k in KEYS},
           "other": {k: {"sim": round(float(best[k]), 3), "real": round(R[k], 3), "interval": [round(I[k][0], 3), round(I[k][1], 3)]}
                     for k in OTHER},
           "settings": len(df), "plants": CAL, "screen_flow_exponent": 1.75, "seconds": round(time.time() - t0)}
    (OUT / "calibration15.json").write_text(json.dumps(res, indent=1))
    print(df.head(8)[["x", "p", "w", "objective", "inside"] + KEYS].round(3).to_string())
    print(json.dumps(res["five"], indent=0))
