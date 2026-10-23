"""Step 1 - the closed-circuit RO record: check, convert, cut into batches.

Checks the seven zips (size and sha256), converts every CSV to parquet (text
only - nothing inside the zips is executed), and writes one row per batch:
phases, energy, permeate, brine, the conductivity it ran to, its setpoints.
No stopping rule is scored here.

    python projects/p05_when_to_stop/step1_ccro_audit.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from chemai.config import CCROConfig
from chemai.data.nawi_ccro import check_files, convert_all, read_dataset
from chemai.features.ccro_batches import COLUMNS, batch_table

OUT = Path(__file__).parent / "results"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = CCROConfig()
    files = check_files(cfg)
    print(files[["file", "present", "size_ok", "sha256_ok"]].to_string(index=False))
    if not files[["present", "size_ok", "sha256_ok"]].all().all():
        raise SystemExit("download the zips listed above into data/nawi_ccro/ (urls in check_files)")
    convert_all(cfg)

    parts, per_dataset = [], {}
    for name in ("dataset1", "dataset2", "dataset3", "dataset4", "dataset5", "dataset6"):
        cols = [c for c in COLUMNS]
        df = read_dataset(name)
        df = df[[c for c in cols if c in df.columns]]
        per_dataset[name] = {
            "rows": int(len(df)), "files": int(df.groupby(["exp", "file"], observed=True).ngroups),
            "first": str(df.ts.min()), "last": str(df.ts.max()),
            "median_dt_s": round(float(np.nanmedian(np.diff(df.runtime_hr.to_numpy(float)) * 3600)), 3)
            if "runtime_hr" in df else None,
            "has_states": bool("state" in df and df.state.notna().mean() > 0.5),
        }
        if "state" not in df or df.state.notna().mean() < 0.5:
            continue                       # the oldest logger version has no phase column
        for c in COLUMNS:
            if c not in df:
                df[c] = np.nan
        parts.append(batch_table(df, cfg))
    b = pd.concat(parts, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    b.to_parquet(OUT / "ccro_batches.parquet")

    groups = b.groupby(["dataset", "exp"])
    summary = {
        "files": files[["file", "url", "sha256_ok"]].to_dict("records"),
        "datasets": per_dataset,
        "batches": int(len(b)),
        "batches_with_own_brine": int(b.brine_l.notna().sum()),
        "constant_setpoint_batches": int(b.constant_setpoints.sum()),
        "by_experiment": groups.agg(
            batches=("seq", "size"), first=("t_first", "min"), last=("t_last", "max"),
            filt_min=("filt_min", "median"), c_start=("c_start", "median"), c_end=("c_end", "median"),
            flux=("flux", "median"), q_recirc=("q_recirc", "median"), constant=("constant_setpoints", "mean"),
            V_l=("V_l", "median"), brine_l=("brine_l", "median"), sec=("sec", "median"), sbc=("sbc", "median"),
            temp_c=("temp_c", "median"),
        ).round(3).reset_index().astype({"first": str, "last": str}).to_dict("records"),
        "drain_kwh_dev_median": round(float(b[b.dataset.isin(cfg.development)].own_drain_kwh.median()), 4),
        "brine_l_dev_median": round(float(b[b.dataset.isin(cfg.development)].brine_l.median()), 2),
    }
    (OUT / "step1_ccro_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_experiment"}, indent=2, default=str))
    print(pd.DataFrame(summary["by_experiment"]).to_string(index=False))


if __name__ == "__main__":
    main()
