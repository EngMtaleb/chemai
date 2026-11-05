"""Phase 3 addition - Prophet (method 7, as approved) and Prophet + level shift (7b, declared variant).

    <python with prophet> -B run_prophet.py            (writes pred_prophet.csv and prophet_log.json next to this file)

One fit per train, indicator and forecast origin t, on the train's records with day <= t; changepoints exactly at the
train's ledger cleaning days <= t; weekly seasonality on, yearly and daily off; Prophet's defaults otherwise; point
forecasts only. 7b adds a step regressor (0 up to day c, 1 after) for each ledger cleaning c < t. Forecast = yhat at
t + h, clipped to 0-1; a failed fit falls back to no change and is counted. Origins: every (train, t) that starts at least
one test pair (day 550-771). Prophet uses only the forecast train's own history, so the same forecasts serve both tests.
"""
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE3 = HERE.parent
sys.path.insert(0, str(PHASE3))
import p3lib as L  # noqa: E402  (unchanged phase-3 functions: the pairs are built exactly as in phase 3)
from prophet import Prophet  # noqa: E402

for name in ("cmdstanpy", "prophet"):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).disabled = True
warnings.simplefilter("ignore")

DATA = PHASE3 / "data"
SHA = {"derived_daily_indicator_table.csv": "50ad81cb632dfc74e710bad70ce9bf08a014061d0ef7b602be25fc465f1f6374",
       "cip_campaign_ledger.csv": "4f65351d8aaa96d8c804c63d3257978dd22628553e7ab4e12d08f1aa0a375ca7"}
BASE = pd.Timestamp("2000-01-01")                       # day 0; only the spacing of the days matters
VARIANTS = {"Prophet": False, "Prophet + level shift": True}


def date(days):
    return BASE + pd.to_timedelta(np.asarray(days, dtype=float), unit="D")


def forecast(days_hist, y_hist, cips, t, horizons, level_shift):
    cps = [int(c) for c in cips if c <= t]
    m = Prophet(changepoints=list(date(cps)), yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False, uncertainty_samples=0)
    df = pd.DataFrame({"ds": date(days_hist), "y": y_hist})
    fut = pd.DataFrame({"ds": date([t + h for h in horizons])})
    if level_shift:
        for c in cips[cips < t]:
            name = f"after_cip_{int(c)}"
            df[name] = (days_hist > c).astype(float)
            fut[name] = 1.0
            m.add_regressor(name, standardize=False)
    m.fit(df)
    return np.clip(m.predict(fut).yhat.to_numpy(float), 0, 1)


def main():
    t0 = time.time()
    for n, sha in SHA.items():
        assert hashlib.sha256((DATA / n).read_bytes()).hexdigest() == sha, n
    d = pd.read_csv(DATA / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
    led = pd.read_csv(DATA / "cip_campaign_ledger.csv")
    days = L.build_days(d, led)
    pairs = days[days.valid & (days.block >= 0)]
    origins = pairs[["train", "t"]].drop_duplicates().sort_values(["train", "t"])
    print(f"{len(origins)} origins x {len(L.TARGETS)} indicators x {len(VARIANTS)} forms", flush=True)
    rows, failures, n_fit, fit_s = [], [], 0, 0.0
    for tr, g in d.groupby("train"):
        cips = np.sort(led[led.train == tr].day_index.values)
        for t in origins[origins.train == tr].t:
            hist = g[g.day_index <= t]
            row = {"train": tr, "t": int(t), "n_hist": len(hist), "cips_known": int((cips <= t).sum())}
            for c in L.TARGETS:
                for vname, lvl in VARIANTS.items():
                    t1 = time.time()
                    try:
                        f = forecast(hist.day_index.to_numpy(), hist[c].to_numpy(float), cips, int(t), L.HORIZONS, lvl)
                    except Exception as e:                       # declared fallback: no change, counted
                        f = np.repeat(float(hist[c].iloc[-1]), len(L.HORIZONS))
                        failures.append({"train": tr, "t": int(t), "target": c, "form": vname, "error": repr(e)[:300]})
                    fit_s += time.time() - t1
                    n_fit += 1
                    for h, v in zip(L.HORIZONS, f):
                        row[f"{vname}|{c}|{h}"] = float(v)
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"  {len(rows)} origins done [{time.time() - t0:.0f} s]", flush=True)
    wide = pd.DataFrame(rows)
    long = []
    for c in L.TARGETS:
        for h in L.HORIZONS:
            part = wide[["train", "t"]].assign(target=c, h=h)
            for vname in VARIANTS:
                part[vname] = wide[f"{vname}|{c}|{h}"]
            long.append(part)
    out = pd.concat(long, ignore_index=True)
    out.to_csv(HERE / "pred_prophet.csv", index=False)
    log = {"origins": int(len(origins)), "fits": n_fit, "failed_fits": len(failures), "failures": failures,
           "seconds_fitting": round(fit_s, 1), "seconds_total": round(time.time() - t0, 1),
           "prophet_version": __import__("prophet").__version__, "pandas": pd.__version__, "numpy": np.__version__}
    (HERE / "prophet_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
