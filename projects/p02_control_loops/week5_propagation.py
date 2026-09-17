"""Project 2 - Week 5: plant-wide propagation.

Run:  python projects/p02_control_loops/week5_propagation.py

  1. Eastman - find the loops sharing one oscillation and rank the candidates
     against the published root cause (tag 22).
  2. Simulation - a known source and three victims, to measure what the ranking
     can and cannot do.
  3. The three coherent horch loops - where the ranking gets it wrong, and why.

Teaching version: notebook p02_week5_propagation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import detrend, lfilter

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import load_sacac, sample_loop_spec, simulate_loop
from chemai.features import band_share, find_oscillation_cluster, rank_source_candidates

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week5")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week5_propagation.json"

EASTMAN_TS = 20.0
EASTMAN_ROOT_CAUSE = "tag22"          # Thornhill, Cox & Paulonis (2003): LC2, sticking valve
TRIO = ["stiction-F-paper-horch-2003", "tuning-Q-paper-horch-2003", "other-F-paper-horch-2003"]
TRIO_LABELS = ["stiction", "tight tuning", "external disturbance"]


def eastman() -> dict:
    m = sio.loadmat(next((DATA_DIR / "sacac").rglob("EastmanDatasetFromNFThornhill_Data.mat")))
    names = [f"tag{j + 1}" for j in range(m["errmat"].shape[1])]
    event = find_oscillation_cluster(m["errmat"], names, fs=1 / EASTMAN_TS)
    table = rank_source_candidates(event)
    log.info("EASTMAN: %d loops share a %.1f min oscillation\n%s",
             len(event.members), event.period / 60, table.to_string(index=False))
    first = table.iloc[0]
    log.info("published root cause: %s | leading candidate: %s (%.2f vs %.2f for the next)",
             EASTMAN_ROOT_CAUSE, first.loop, first.band_share, table.iloc[1].band_share)
    return {"period_min": round(event.period / 60, 1), "members": event.members,
            "candidates": table.to_dict("records"),
            "leading": first.loop, "correct": first.loop == EASTMAN_ROOT_CAUSE,
            "victims_not_to_work_on": event.victims()}


def simulated(cfg: ControlLoopConfig) -> dict:
    """A known source and three victims - the oscillation arrives filtered, with
    local noise added, as it does through real process units."""
    spec = sample_loop_spec("F", "stiction", np.random.default_rng(4), cfg)
    run = simulate_loop(spec, 3000, seed=3, cfg=cfg)
    source = detrend((run.sp - run.pv).to_numpy())

    def victim(tau, noise_ratio, seed):
        a = np.exp(-1 / tau)
        return (lfilter([1 - a], [1, -a], source)
                + noise_ratio * np.std(source) * np.random.default_rng(seed).normal(len(source)))

    names = ["source", "victim_near", "victim_mid", "victim_far"]
    signals = np.column_stack([source, victim(3, 0.2, 1), victim(10, 0.4, 2), victim(25, 0.8, 3)])
    event = find_oscillation_cluster(signals, names, fs=1.0, nperseg=1024, min_share=0.005)
    table = rank_source_candidates(event)
    log.info("SIMULATION (source known)\n%s", table.to_string(index=False))
    gap = float(table.iloc[0].band_share - table.iloc[1].band_share)
    log.info("leading candidate: %s, margin over the next loop: %.2f", table.iloc[0].loop, gap)
    return {"candidates": table.to_dict("records"), "leading": table.iloc[0].loop,
            "correct": table.iloc[0].loop == "source", "margin": round(gap, 3)}


def horch_trio() -> dict:
    """Three loops, one 29 s oscillation, three different published labels."""
    recs = {r.name: r for r in load_sacac(DATA_DIR / "sacac")}
    signals = np.column_stack([detrend(recs[n].data.pv.to_numpy()[:1196]) for n in TRIO])
    shares = band_share(signals, 29.2, fs=1.0, nperseg=256)
    table = pd.DataFrame({"loop": TRIO, "label": TRIO_LABELS,
                          "band_share": shares.round(3)}).sort_values(
        "band_share", ascending=False)
    log.info("HORCH TRIO - no published root cause\n%s", table.to_string(index=False))
    leading = table.iloc[0]
    log.info("band share leads to '%s'; the waveform shapes pointed at the stiction loop "
             "(section 11.6) - a non-linear source is penalised by its own harmonics",
             leading.label)
    return {"table": table.to_dict("records"), "leading_label": leading.label,
            "agrees_with_shape_evidence": bool(leading.label == "stiction")}


def main() -> dict:
    cfg = ControlLoopConfig()
    results = {"eastman": eastman(), "simulated": simulated(cfg), "horch_trio": horch_trio()}
    results["adopted"] = {
        "clustering": "adopted - loops sharing one period are one event",
        "ranking": "adopted as a FIRST FILTER only, never as a verdict",
        "nonlinearity_ranking": "rejected - the simple distortion measure tracked the "
                                "noise floor and ranked the known root cause last",
    }
    log.info("verdict: clustering adopted; ranking is a filter; "
             "non-linearity ranking rejected (see VALIDATION.md 20)")
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
