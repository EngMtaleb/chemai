"""Step 6 - after the test: what the batch rule means at full scale (model-based, labelled so).

In a closed loop of volume V_L, salt in = feed conductivity c_f times permeate
volume, so the loop concentrates linearly, c = c_0 + c_f V / V_L, and the feed
pressure follows the osmotic pressure, dp/dc = k_pi. The live specific energy
then rises in a straight line,

    s = k_pi * c_f / (36 * eta * V_L)        kWh/m3 per m3 of permeate

and ending a batch costs F = b * V_brine (+ drain and start-up energy). The
square-root law gives the best permeate volume per batch,

    V* / V_L = sqrt( 72 * beta * eta * b / (k_pi * c_f) ),     beta = V_brine / V_L

a dimensionless rule: brine price against the osmotic slope. The testbed
measures k_pi, the loop volume and beta (development batches); the testbed's
pumps are what do NOT transfer - its marginal efficiency is ~13 %, a full-scale
pump's ~80 %. Nothing here is a test; it is the arithmetic the tested rule
implies once the pumps are replaced.

    python projects/p05_when_to_stop/step6_ccro_full_scale.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import read_dataset
from chemai.features.ccro_batches import COLUMNS, batch_curve, phase_of

HERE = Path(__file__).parent
OUT = HERE / "results"


def development_constants(cfg: CCROConfig) -> pd.DataFrame:
    """Per development batch: dc/dV (mS/cm per L), dp/dc (bar per mS/cm), live slope s (kWh/m3 per m3)."""
    batches = pd.read_parquet(OUT / "ccro_batches.parquet")
    rows = []
    for name in cfg.development:
        df = read_dataset(name, columns=COLUMNS)
        bd = batches[batches.dataset == name].set_index(["exp", "file"])
        for (exp, file), g in df.groupby(["exp", "file"], observed=True, sort=False):
            if (exp, file) not in bd.index:
                continue
            row = bd.loc[(exp, file)]
            if not row.constant_setpoints or row.filt_min < 15 or not np.isfinite(row.brine_l):
                continue
            g = g.sort_values("row")
            cv = batch_curve(g, cfg)
            ff = cv.phase.to_numpy() == "filt"
            w = ff & (cv.elapsed_min.to_numpy() >= 5) & np.isfinite(cv.live_kwh_m3.to_numpy())
            if w.sum() < 100:
                continue
            in_batch = g["state"].map(phase_of).isin(["startup", "stab", "filt"]).to_numpy()
            p_bar = g["feed_pressure_psi"].to_numpy(float)[in_batch] * 0.0689476      # same rows as the curve
            V = cv.V_m3.to_numpy() * 1000.0
            c = cv.c_ms_cm.to_numpy()
            rows.append({"dataset": name, "exp": exp, "seq": int(row.seq), "brine_l": float(row.brine_l),
                         "dc_dV": float(np.polyfit(V[w], c[w], 1)[0]),
                         "dp_dc": float(np.polyfit(c[w], p_bar[w], 1)[0]) if len(p_bar) == len(c) else np.nan,
                         "s": float(np.polyfit(V[w] / 1000.0, cv.live_kwh_m3.to_numpy()[w], 1)[0])})
    return pd.DataFrame(rows)


def best_volume_ratio(b: float, eta: float, k_pi: float, c_f: float, beta: float) -> float:
    return float(np.sqrt(72.0 * beta * eta * b / (k_pi * c_f)))


def main():
    cfg = CCROConfig()
    k = development_constants(cfg)
    k.to_csv(OUT / "step6_dev_constants.csv", index=False)
    feed = {"Exp1": 1.8, "Exp2": 1.8, "Exp2_continued/Data": 1.8, "Exp3": 2.7}      # nominal feed, ESC readme
    esc = k[k.dataset == "dataset3"].copy()
    esc["c_f"] = esc.exp.map(feed)
    esc["V_L_l"] = esc.c_f / esc.dc_dV                                             # near-total rejection
    k_pi = float(k.dp_dc.median())
    v_l = float(esc.V_L_l.median())
    beta = float(esc.brine_l.median() / v_l)
    ideal_s = k_pi * float(k.dc_dV.median()) * 1000.0 / 36.0                        # kWh/m3 per m3 at eta = 1
    eta_pilot = ideal_s / float(k.s.median())

    table = []
    for c_f in (1.0, 2.0, 3.5):
        for b in (1.0, 3.0, 10.0, 30.0):
            r = best_volume_ratio(b, 0.80, k_pi, c_f, beta)
            table.append({"feed_ms_cm": c_f, "b_kwh_per_m3_brine": b, "V_star_over_V_loop": round(r, 2),
                          "end_conductivity_ms_cm": round(c_f * (1 + r), 1),
                          "batch_recovery": round(r / (r + beta), 3)})
    summary = {
        "kind": "model-based translation, after the test; no pass mark",
        "development_batches": int(len(k)),
        "k_pi_bar_per_ms_cm": round(k_pi, 3),
        "loop_volume_l": round(v_l, 1), "beta_brine_over_loop": round(beta, 2),
        "live_slope_kwh_m3_per_m3": {"testbed_median": round(float(k.s.median()), 2),
                                     "at_eta_1": round(ideal_s, 2)},
        "testbed_marginal_efficiency": round(eta_pilot, 3),
        "full_scale": {"eta": 0.80, "first_batch_conductivity": "equal to the feed (a complete drain)",
                       "table": table},
        "caveat": ("NaCl-only feed: real brackish water reaches a scaling limit (gypsum, silica) first; "
                   "the table gives the energy-and-brine optimum to compare with that limit"),
    }
    (OUT / "step6_ccro_full_scale.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
