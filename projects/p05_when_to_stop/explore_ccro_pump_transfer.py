"""Exploration, development data only: do the testbed's pump-power curves carry from one month to
the next? Kept to show why Project 5 builds no model of the live cost for setpoints a batch did not
run (VALIDATION.md 9): fitted on the June-July ESC batches, they miss the August open-loop batches.

    python projects/p05_when_to_stop/explore_ccro_pump_transfer.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.data.nawi_ccro import read_dataset

HERE = Path(__file__).parent
COLS = ["dataset", "state", "reject_conductivity_ms_per_cm", "reject_flow_l_per_min", "permeate_flow_l_per_min",
        "feed_pressure_psi", "feed_pump_power_draw_kw", "recirculation_pump_power_draw_kw",
        "feed_temperature_c", "flux_lmh"]


def load() -> pd.DataFrame:
    parts = []
    for name in ("dataset3", "dataset4"):
        d = read_dataset(name, columns=COLS)
        d = d[d.state.astype(str) == "Experiment"].iloc[::20]           # ~10 s apart
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return d[(d.permeate_flow_l_per_min > 0.5) & (d.flux_lmh > 3)]


def main():
    d = load()
    one = np.ones(len(d))
    T = d.feed_temperature_c.to_numpy(float)
    tcf = np.exp(2640 * (1 / 298.15 - 1 / (T + 273.15)))
    c, J = d.reject_conductivity_ms_per_cm.to_numpy(float), d.flux_lmh.to_numpy(float)
    Qr, Qp = d.reject_flow_l_per_min.to_numpy(float), d.permeate_flow_l_per_min.to_numpy(float)
    p = d.feed_pressure_psi.to_numpy(float) * 0.0689476
    models = {
        "feed_pressure": (np.column_stack([one, c, J / tcf, Qr ** 1.75]), p),
        "feed_pump_power": (np.column_stack([one, p * Qp, Qp, p]), d.feed_pump_power_draw_kw.to_numpy(float)),
        "recirculation_pump_power": (np.column_stack([one, Qr ** 3, Qr]), d.recirculation_pump_power_draw_kw.to_numpy(float)),
    }
    train = (d.dataset == "dataset3").to_numpy()
    out = {"samples_june_july": int(train.sum()), "samples_august": int((~train).sum())}
    for name, (X, y) in models.items():
        b, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        err = (y[~train] - X[~train] @ b) / y[~train]
        fit = (y[train] - X[train] @ b) / y[train]
        out[name] = {"mape_in_sample_pct": round(float(100 * np.mean(np.abs(fit))), 1),
                     "mape_august_pct": round(float(100 * np.mean(np.abs(err))), 1),
                     "bias_august_pct": round(float(100 * np.mean(err)), 1)}
    (HERE / "results" / "explore_ccro_pump_transfer.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
