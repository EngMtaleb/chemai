"""Project 2 - Week 2, part 1: data-quality checks on every real loop.

Run:  python projects/p02_control_loops/week2_data_quality.py
      (needs data/sacac/ and data/isdb/isdb10.mat)

Reproduces the evidence behind the thresholds in DataQualityConfig and checks
that each trap is caught on the real file that illustrates it.
Teaching version: notebook p02_week2_data_quality.
"""
from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import DATA_DIR, DataQualityConfig
from chemai.data import load_isdb, load_sacac
from chemai.features import assess

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week2")
log.setLevel(logging.INFO)
OUT = Path(__file__).parent / "results_week2_dq.json"

# each trap, and the real file that shows it
EXPECTED = {
    "quantisation-Q-paper-horch-2003": "quantised_pv",
    "quantisation-T-chemicals-Thornhill-2003": "quantised_pv",
    "saturation-L-minerals-bauer-2017": "saturated",
    "saturation-T-oilgas-thornhill-2002": "saturated",
}


def all_loops():
    """(source, name, sp, pv, op) for every single-loop record; ISDB loops that
    are already in SACAC are kept once, under their SACAC name."""
    rows = []
    for r in load_sacac(DATA_DIR / "sacac"):
        d = r.data
        rows.append(("SACAC", r.name, d.sp.to_numpy(), d.pv.to_numpy(), d.op.to_numpy()))
    for x in load_isdb(DATA_DIR / "isdb" / "isdb10.mat"):
        if x.in_sacac:
            continue
        d = x.data
        rows.append(("ISDB", x.key, d.sp.to_numpy(), d.pv.to_numpy(), d.op.to_numpy()))
    return rows


def main() -> dict:
    cfg = DataQualityConfig()
    loops = all_loops()
    recs = []
    for src, name, sp, pv, op in loops:
        if np.isnan(sp).all():
            sp = np.zeros_like(pv)                      # PV-only file: no SP recorded
        rep = assess(sp, pv, op, cfg)
        recs.append({"source": src, "name": name, "n": rep.n, "pv_flat_run": rep.pv_flat_run,
                     "op_flat_run": rep.op_flat_run, "pv_levels": rep.pv_levels,
                     "op_levels": rep.op_levels, "saturation": rep.saturation,
                     "pi_r2": rep.pi_r2, "dq_flags": rep.flags,
                     "ok_for_harris": rep.ok_for_harris})
    df = pd.DataFrame(recs)
    log.info("assessed %d distinct real loops", len(df))

    evidence = {
        "pv_flat_run_quantiles": df.pv_flat_run.quantile([.5, .95, .99]).round(0).to_dict(),
        "pv_levels_quantiles": df.pv_levels.quantile([.01, .05, .5]).round(0).to_dict(),
        "pi_r2_quantiles": df.pi_r2.dropna().quantile([.1, .25, .5]).round(3).to_dict(),
    }
    log.info("threshold evidence: %s", evidence)

    counts = collections.Counter(f for fl in df.dq_flags for f in fl)
    log.info("flags raised: %s", dict(counts))
    for flag in sorted(counts):
        names = df[df.dq_flags.apply(lambda f: flag in f)].name.tolist()
        log.info("  %-24s %s", flag, ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))

    checks = {}
    for name, flag in EXPECTED.items():
        row = df[df.name.str.lower() == name.lower()]
        caught = bool(len(row)) and flag in row.iloc[0].dq_flags
        checks[name] = {"expected": flag, "caught": caught}
        log.info("%-44s expected %-14s -> %s", name, flag, "caught" if caught else "MISSED")

    manual = df[df.name == "unknown-F-paper-horch-2003"].iloc[0]
    log.info("file described as MANUAL: flags %s, PI R2 %.2f", manual.dq_flags, manual.pi_r2)

    results = {"n_loops": int(len(df)), "thresholds": cfg.__dict__, "evidence": evidence,
               "flag_counts": dict(counts), "expected_cases": checks,
               "ok_for_harris": int(df.ok_for_harris.sum())}
    OUT.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
