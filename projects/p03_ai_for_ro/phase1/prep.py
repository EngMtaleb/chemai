"""Phase 3, step 1 - build the forecasting tables for every horizon and list the events, before any model is run.

    python prep.py <folder with derived_daily_indicator_table.csv and cip_campaign_ledger.csv> <out folder>

Pairs     (train, day t) -> the same train on day t + h, h = 1, 3, 7, 14; both days must have data and no cleaning may
          fall in (t, t + h]: we forecast fouling, not the operators' decision to clean.
Inputs    only what is known on day t: today's four indicators, their values 1, 3, 7 and 14 days back (nearest earlier
          day within 2 days), the 14-day slope of dP and flux inside the current cycle, days since the last cleaning,
          the number of 10-min samples behind today's value, the smoothed state, and the fouling-law forecast.
Blocks    future test blocks by day t: [550, 600), [600, 650), [650, 700), [700, 772); a block's model trains on pairs
          whose target day t + h is before the block starts (all earlier data, all trains).
Events    dP onset: the first day a train's 7-day median dP rise reaches 15 % after at least 14 days below it.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SHA = {"derived_daily_indicator_table.csv": "50ad81cb632dfc74e710bad70ce9bf08a014061d0ef7b602be25fc465f1f6374",
       "cip_campaign_ledger.csv": "4f65351d8aaa96d8c804c63d3257978dd22628553e7ab4e12d08f1aa0a375ca7"}
IND = ["dp_pct", "flux_pct", "sp_pct", "ndp_pct"]
TARGETS = ["dp_pct", "flux_pct"]
HORIZONS = (1, 3, 7, 14)
BLOCKS = [(550, 600), (600, 650), (650, 700), (700, 772)]
THRESH = 0.15


def near(s, day, tol=2):
    for k in range(tol + 1):
        if day - k in s.index:
            return s[day - k]
    return np.nan


def cycle_view(g, cips, t):
    """the current cleaning cycle's observations up to and including day t."""
    last = cips[cips <= t].max() if (cips <= t).any() else -1
    return g[(g.index > last) & (g.index <= t)], last


def law_forecast(cyc, c, t, h, min_pts=5):
    """fouling law: linear growth since the last cleaning, fitted on the cycle so far, extrapolated to t + h."""
    if len(cyc) < min_pts:
        return np.nan
    b, a = np.polyfit(cyc.index.values.astype(float), cyc[c].values, 1)
    return float(np.clip(a + b * (t + h), 0, 1))


def build(d, led):
    rows = []
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        cips = np.sort(led[led.train == tr].day_index.values)
        for t in g.index:
            cyc, last = cycle_view(g, cips, t)
            base = {"train": tr, "t": int(t), "days_since_cip": float(t - last) if last >= 0 else np.nan,
                    "n_samples": float(g.at[t, "n_10min_samples"])}
            for c in IND:
                base[c] = float(g.at[t, c])
                for lag in (1, 3, 7, 14):
                    base[f"{c}_lag{lag}"] = near(g[c], t - lag)
            recent = cyc[cyc.index > t - 14]
            recent3 = cyc[cyc.index > t - 7].tail(3)
            for c in TARGETS:
                base[f"{c}_slope14"] = float(np.polyfit(recent.index, recent[c], 1)[0]) if len(recent) >= 4 else np.nan
                base[f"{c}_state"] = float(recent3[c].median()) if len(recent3) else float(g.at[t, c])
            for h in HORIZONS:
                if t + h not in g.index or ((cips > t) & (cips <= t + h)).any():
                    continue
                r = dict(base, h=h)
                for c in TARGETS:
                    r[f"y_{c}"] = float(g.at[t + h, c])
                    r[f"{c}_law"] = law_forecast(cyc, c, t, h)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["block"] = -1
    for i, (a, b) in enumerate(BLOCKS):
        df.loc[(df.t >= a) & (df.t < b), "block"] = i
    return df


def onsets(d, led):
    out, m7 = [], {}
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        med = pd.Series({t: g.dp_pct[(g.index > t - 7) & (g.index <= t)].median()
                         for t in g.index if ((g.index > t - 7) & (g.index <= t)).sum() >= 3})
        m7[tr] = med
        for t, v in med.items():
            if v < THRESH:
                continue
            before = med[(med.index >= t - 14) & (med.index < t)]
            if len(before) and (before < THRESH).all():
                out.append({"train": tr, "day": int(t)})
    return pd.DataFrame(out), m7


def main():
    for name, sha in SHA.items():
        got = hashlib.sha256((SRC / name).read_bytes()).hexdigest()
        assert got == sha, f"{name}: not the published file"
    d = pd.read_csv(SRC / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"]).reset_index(drop=True)
    led = pd.read_csv(SRC / "cip_campaign_ledger.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    df = build(d, led)
    df.to_csv(OUT / "pairs.csv", index=False)
    ev, m7 = onsets(d, led)
    ev.to_csv(OUT / "onsets.csv", index=False)
    pd.concat({tr: s for tr, s in m7.items()}).rename("dp_median7").to_csv(OUT / "dp_median7.csv")
    counts = {}
    for h in HORIZONS:
        x = df[df.h == h]
        counts[h] = {"pairs": int(len(x)), **{f"block {i}": int((x.block == i).sum()) for i in range(len(BLOCKS))},
                     "before 550": int((x.t < 550).sum()), **{f"train {k}": int((x.train == k).sum()) for k in "ABC"}}
    c = pd.DataFrame(counts)
    print(c.to_string())
    print("\nonsets of dP >= +15 % (7-day median, after >= 14 days below):")
    print(ev.groupby("train").day.apply(list).to_string())
    print("onsets from day 550:", int((ev.day >= 550).sum()), "| before 550:", int((ev.day < 550).sum()))
    (OUT / "prep_summary.json").write_text(json.dumps({"counts": counts, "onsets": ev.to_dict("records")}, indent=1))


if __name__ == "__main__":
    main()
