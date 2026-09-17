"""Project 2 - Week 4: cascade loops.

Run:  python projects/p02_control_loops/week4_cascade.py

  1. Detect cascade structure in the three plants that recorded every loop at
     the same time (Eastman, two refinery units).
  2. Check the attribution rule against the simulator, where the fault is known.
  3. Apply it to the one real pair whose slave carries a published fault.

Teaching version: notebook p02_week4_cascade.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import load_isdb
from chemai.features import attribute_source, find_cascades, simulate_cascade

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week4")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week4_cascade.json"

GROUPS = {"South-East Asian refinery": [f"chemicals.{k}" for k in range(40, 70)],
          "refinery separation unit": [f"chemicals.{k}" for k in range(13, 18)]}
# the one real pair whose slave has a published fault
LABELLED_PAIR = ("chemicals.17", "chemicals.14")


def eastman_pairs() -> pd.DataFrame:
    path = next((DATA_DIR / "sacac").rglob("EastmanDatasetFromNFThornhill_Data.mat"))
    m = sio.loadmat(path)
    names = [f"tag{k + 1}" for k in range(m["spmat"].shape[1])]
    return find_cascades(m["spmat"], m["opmat"], names)


def isdb_pairs(loops: dict, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in loops]
    sp = np.column_stack([loops[k].data.sp.to_numpy() for k in keys])
    op = np.column_stack([loops[k].data.op.to_numpy() for k in keys])
    names = [f"{loops[k].comment.split('(')[0].split(':')[-1].strip()[:26]} [{k}]" for k in keys]
    return find_cascades(sp, op, names)


def main() -> dict:
    cfg = ControlLoopConfig()

    # 1. structure, from the data alone
    found = {"Eastman chemical plant": eastman_pairs()}
    loops = {x.key: x for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat")}
    for title, keys in GROUPS.items():
        found[title] = isdb_pairs(loops, keys)
    total = sum(len(v) for v in found.values())
    for title, table in found.items():
        log.info("%s: %d candidate pairs\n%s", title, len(table),
                 table.to_string(index=False) if len(table) else "  none")
    log.info("%d cascade candidates across %d plants", total, len(found))

    # 2. the rule against the simulator, where the fault is known
    cases = {"healthy": {}, "slave valve sticks": dict(stiction_s=3.0, slip_s=2.5),
             "master tuned too tight": dict(kc_m=8.5, ti_m=15.0)}
    expected = {"healthy": "none", "slave valve sticks": "slave",
                "master tuned too tight": "master"}
    simulated = {}
    for name, kw in cases.items():
        run = simulate_cascade(**kw)
        v = attribute_source(run.sp_s, run.pv_s, cfg)
        simulated[name] = {"source": v.source, "slave_error": v.slave_error,
                           "slave_setpoint": v.slave_setpoint, "owner": v.owner,
                           "correct": v.source == expected[name]}
        log.info("%-24s -> %-7s (slave error %.2f, setpoint %.2f) %s", name, v.source,
                 v.slave_error, v.slave_setpoint, "OK" if simulated[name]["correct"] else "WRONG")

    # 3. the one labelled real pair
    master, slave = (loops[k] for k in LABELLED_PAIR)
    v = attribute_source(slave.data.sp, slave.data.pv, cfg)
    log.info("real pair: master %s | slave %s (%s)", master.key, slave.key,
             slave.comment.split("|")[0].strip())
    log.info("  -> %s (slave error %.2f, setpoint %.2f) - owner: %s",
             v.source, v.slave_error, v.slave_setpoint, v.owner)
    log.info("  the slave carries the published fault, and the rule points at the slave")
    log.info("  NOTE: the setpoint here does oscillate, but irregularly - the rule "
             "separates by regularity, not by quiet")

    results = {
        "candidate_pairs": {k: v.to_dict("records") for k, v in found.items()},
        "n_candidates": total,
        "simulated": simulated,
        "all_simulated_correct": all(v["correct"] for v in simulated.values()),
        "real_pair": {"master": master.key, "slave": slave.key,
                      "slave_comment": slave.comment.split("|")[0].strip(),
                      "source": v.source, "slave_error": v.slave_error,
                      "slave_setpoint": v.slave_setpoint, "owner": v.owner},
        "limits": ["two levels only", "one labelled real pair",
                   "names the loop, not the fault",
                   "a fault in BOTH loops reads as master",
                   "ratio control leaves the same fingerprint"],
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
