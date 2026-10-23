"""Step 3 - how big a tide could this record see? (before the test is run)

An M2 wave of known size is added to noise shaped like the data (AR(1) per
run, with the residuals' own spread and persistence) and the full test is run
on it. The data's real alignment with the tide is never used here, so this can
be - and was - run before the declaration was committed.

    python projects/p04_tide_fouling/step3_power.py
"""
import json
from pathlib import Path

import pandas as pd

from chemai.evaluation.tide_test import TideTestConfig, power_curve, prepare

OUT = Path(__file__).parent / "results"
SIZES = (0.02, 0.05, 0.10, 0.20)          # 5-95 % range of the wave / median response


def main():
    seg = pd.read_parquet(OUT / "segments.parquet")
    cfg = TideTestConfig()
    out = {}
    jobs = {"fouling_pct_h": seg,
            "turbidity": seg[seg.turbidity_reads == "feed"],
            "pump_speed": seg, "p_mf_in": seg}
    for col, data in jobs.items():
        prep = prepare(data, col, cfg)
        out[col] = {"n": int(len(prep.y)), "median": float(pd.Series(prep.y).median()),
                    "detection_rate_at_p_le_0.01": power_curve(prep, SIZES, n_sim=200, cfg=cfg)}
        print(col, json.dumps(out[col]), flush=True)
    (OUT / "step3_power.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
