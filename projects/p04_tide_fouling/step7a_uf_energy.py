"""Step 7a - what the UF pre-treatment costs in energy, per m3 of product water.

The design study (step 7) needs the whole plant's energy per m3 of water
delivered. Reverse osmosis is taken from the literature; the pre-treatment
comes from the record:

  hydraulic energy per m3 of UF feed
      = [ integral of  p_uf_in * q_uf   (pushing the feed through the membrane)
        + integral of  p_bw_filt * q_bw (pushing backwash water back through it) ]
        / integral of  q_uf

Pressures in kPa (gauge), flows in m3/s, negatives clipped to zero. Then, per
m3 of RO product: divide by a pump efficiency (0.7) and by the RO recovery
(0.45: 2.2 m3 of filtered seawater per m3 of product). Backwash water is
assumed not to come out of the RO feed (the pilot kept filtering at full flow
while one element backwashed); if it did, the UF share rises by 1/(1 - share).

    python projects/p04_tide_fouling/step7a_uf_energy.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.data.uf_pilot import read_run, run_root
from chemai.features.uf_segments import seconds

OUT = Path(__file__).parent / "results"
GAL_M3 = 3.785411784e-3
PSI_KPA = 6.894757
PUMP_EFFICIENCY = 0.7
RO_RECOVERY = 0.45


def run_energy(df: pd.DataFrame) -> dict:
    t = seconds(df.index)
    dt = np.diff(t, append=t[-1])                               # forward step; last sample carries none
    q = df["q_uf"].clip(lower=0).to_numpy(float) * GAL_M3 / 60
    qb = df["q_bw"].clip(lower=0).to_numpy(float) * GAL_M3 / 60
    p_in = df["p_uf_in"].clip(lower=0).to_numpy(float) * PSI_KPA
    p_bw = df["p_bw_filt"].clip(lower=0).to_numpy(float) * PSI_KPA
    ok = np.isfinite(q) & np.isfinite(qb) & np.isfinite(p_in) & np.isfinite(p_bw)
    v_in, v_bw = float(np.sum((q * dt)[ok])), float(np.sum((qb * dt)[ok]))
    e_filter = float(np.sum((p_in * q * dt)[ok])) / 3600          # kJ -> kWh
    e_backwash = float(np.sum((p_bw * qb * dt)[ok])) / 3600
    on = df["q_uf"] > 1.0
    return {"hours": float((t[-1] - t[0]) / 3600), "v_in_m3": v_in, "v_bw_m3": v_bw,
            "bw_share": v_bw / v_in if v_in > 0 else np.nan,
            "kwh_m3_hyd": (e_filter + e_backwash) / v_in if v_in > 0 else np.nan,
            "backwash_energy_share": e_backwash / (e_filter + e_backwash) if e_filter + e_backwash > 0 else np.nan,
            "p_in_psi": float(df.loc[on, "p_uf_in"].median()) if on.any() else np.nan,
            "q_gpm": float(df.loc[on, "q_uf"].median()) if on.any() else np.nan}


def main():
    runs = pd.read_parquet(OUT / "runs.parquet")
    root = run_root()
    rows = []
    for r in runs.itertuples():
        e = run_energy(read_run(root / r.split / r.file))
        rows.append({"file": r.file, "period": str(r.t_start.year), **e})
    v = pd.DataFrame(rows)
    v.to_parquet(OUT / "uf_volumes.parquet")
    ok = v[(v.v_in_m3 > 1.0) & (v.p_in_psi > 10)]          # runs that actually filtered at pressure
    hyd = ok.kwh_m3_hyd
    per_product = hyd.median() / PUMP_EFFICIENCY / RO_RECOVERY
    out = {
        "runs": int(len(v)), "runs_used": int(len(ok)),
        "kwh_per_m3_uf_feed_hydraulic": {"median": float(hyd.median()),
                                         "iqr": [float(hyd.quantile(0.25)), float(hyd.quantile(0.75))]},
        "backwash_volume_share": {"median": float(ok.bw_share.median()),
                                  "iqr": [float(ok.bw_share.quantile(0.25)), float(ok.bw_share.quantile(0.75))]},
        "backwash_energy_share_median": float(ok.backwash_energy_share.median()),
        "assumed": {"pump_efficiency": PUMP_EFFICIENCY, "ro_recovery": RO_RECOVERY},
        "kwh_per_m3_product": float(per_product),
        "kwh_per_m3_product_if_backwash_from_ro_feed": float(per_product / (1 - ok.bw_share.median())),
    }
    (OUT / "step7a_uf_energy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
