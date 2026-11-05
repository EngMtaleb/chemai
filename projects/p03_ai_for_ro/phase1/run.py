"""Phase 3, steps 2-4 - run the declared tests (declaration.md) and save every prediction.

    python run.py <data folder> <out folder>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3lib as L  # noqa: E402

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SHA = {"derived_daily_indicator_table.csv": "50ad81cb632dfc74e710bad70ce9bf08a014061d0ef7b602be25fc465f1f6374",
       "cip_campaign_ledger.csv": "4f65351d8aaa96d8c804c63d3257978dd22628553e7ab4e12d08f1aa0a375ca7"}


def main():
    t0 = time.time()
    for name, sha in SHA.items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == sha, name
    d = pd.read_csv(SRC / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
    led = pd.read_csv(SRC / "cip_campaign_ledger.csv")
    days = L.build_days(d, led)
    pairs = days[days.valid].reset_index(drop=True)
    print("pairs per horizon:", pairs.groupby("h").size().to_dict())
    feats = L.features(days)
    ro, lo, wd, cal = [], [], [], []
    for c in L.TARGETS:
        for h in L.HORIZONS:
            P, D = pairs[pairs.h == h], days[days.h == h]
            for b, (start, _) in enumerate(L.BLOCKS):
                train = P[P.t + h < start].sort_values(["t", "train"])
                test, alld = P[P.block == b], D[D.block == b]
                f_test, f_days = L.forecasts(train, [test, alld], c, h, feats)
                # 90 % range: errors on the last quarter of the training pairs, by a fit on the first three quarters
                k = int(round(0.75 * len(train)))
                fit, calib = train.iloc[:k], train.iloc[k:]
                (f_cal,) = L.forecasts(fit, [calib], c, h, feats)
                q = {m: float(np.quantile(np.abs(calib[f"y_{c}"].to_numpy() - f_cal[m]), 0.9)) for m in L.METHODS}
                cal.append({"target": c, "h": h, "block": b, "n_train": len(train), "n_calib": len(calib), **q})
                base = test[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"})
                ro.append(base.assign(target=c, h=h, block=b, **{m: f_test[m] for m in L.METHODS},
                                      **{f"q90 {m}": q[m] for m in L.METHODS}))
                if c == "dp_pct" and h in (7, 14):
                    wd.append(alld[["train", "t", c]].rename(columns={c: "now"}).assign(h=h, block=b, **{m: f_days[m] for m in L.METHODS}))
                for k_out in "ABC":                                     # an unseen train, in the future
                    tr_k, te_k = train[train.train != k_out], test[test.train == k_out]
                    if te_k.empty:
                        continue
                    (f_k,) = L.forecasts(tr_k, [te_k], c, h, feats)
                    base = te_k[["train", "t", c, f"y_{c}"]].rename(columns={c: "now", f"y_{c}": "y"})
                    lo.append(base.assign(target=c, h=h, block=b, **{m: f_k[m] for m in L.METHODS}))
                print(f"{c} h={h} block {b}: train {len(train)}, test {len(test)}  [{time.time() - t0:.0f} s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(ro).to_csv(OUT / "pred_rolling.csv", index=False)
    pd.concat(lo).to_csv(OUT / "pred_unseen_train.csv", index=False)
    pd.concat(wd).to_csv(OUT / "pred_warning_days.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
