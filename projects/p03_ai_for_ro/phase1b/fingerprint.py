"""Phase 3b, step 1 - the real plant's fingerprint: how its normalised feed-channel dP behaves, in numbers a simulator
must reproduce before it is trusted. Descriptive only: no model, no forecast.

The same function will score simulated plants in step 3, so it takes a plain table:
    daily  - columns train, day_index, dp_pct (fixed-baseline rise, 0-1), one row per recorded train-day
    legs   - {train: [cleaning-leg days]}; legs closer than 7 days form one campaign (the deposit's own rule)

    python fingerprint.py <folder with the two deposit csv files> <out folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FROM_DAY = 300          # before day 300 the record is sparse (50-53 days per train) and holds the start-up
HORIZONS = (1, 3, 7, 14)
CAMPAIGN_GAP = 7
N_BOOT, SEED = 2000, 2026


def campaigns(legs, gap=CAMPAIGN_GAP):
    legs = sorted(legs)
    out = []
    for d in legs:
        if out and d - out[-1][1] <= gap:
            out[-1][1] = d
        else:
            out.append([d, d])
    return [tuple(c) for c in out]


def cycles(daily, legs):
    """Stretches between campaigns: (train, kind, days, dp). A campaign's own days belong to no cycle."""
    out = []
    for tr, g in daily.groupby("train"):
        g = g.sort_values("day_index")
        camps = campaigns(legs.get(tr, []))
        first, last = int(g.day_index.min()), int(g.day_index.max())
        bounds = [(first - 1, camps[0][0], "initial")] if camps else [(first - 1, last + 1, "initial")]
        bounds += [(a[1], b[0], "complete") for a, b in zip(camps[:-1], camps[1:])]
        if camps:
            bounds.append((camps[-1][1], last + 1, "open"))
        for s, e, kind in bounds:
            seg = g[(g.day_index > s) & (g.day_index < e)]
            out.append(dict(train=tr, kind=kind, start=s, end=e, day=seg.day_index.to_numpy(int),
                            dp=seg.dp_pct.to_numpy(float)))
    return out


def lag_pairs(day, x, h):
    """All (x[t], x[t+h]) with both days recorded."""
    pos = {d: i for i, d in enumerate(day)}
    i = [pos[d] for d in day if d + h in pos]
    j = [pos[d + h] for d in day if d + h in pos]
    return x[i], x[j]


def per_cycle(c):
    """The numbers one cycle contributes. Percentage points throughout."""
    m = c["day"] >= FROM_DAY
    day, dp = c["day"][m], 100 * c["dp"][m]
    r = {"kind": c["kind"], "n": int(m.sum())}
    if c["kind"] == "complete":
        r["length"] = c["end"] - c["start"]
        w = (c["day"] - c["start"] > 3) & (c["day"] - c["start"] <= 45)
        if w.sum() >= 8:
            r["rise_pp_per_week"] = 7 * 100 * stats.theilslopes(c["dp"][w], c["day"][w])[0]
    a, b = lag_pairs(day, dp, 1)
    if len(a) >= 5:
        dx = b - a
        r["noise_pp"] = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2)
    # consecutive one-day changes, for their lag-1 autocorrelation
    pos = {d: i for i, d in enumerate(day)}
    r["chg_pairs"] = [(dp[pos[d + 1]] - dp[pos[d]], dp[pos[d + 2]] - dp[pos[d + 1]])
                      for d in day if d + 1 in pos and d + 2 in pos]
    for h in HORIZONS:
        a, b = lag_pairs(day, dp, h)
        r[f"abs_change_{h}"] = list(np.abs(b - a))
    r["levels"] = list(dp)
    return r


def campaign_numbers(daily, legs):
    """Per campaign: the drop it made (median 14 days before - median 14 days after, at least 3 days each side; the
    deposit's published construction) and the level the operator saw (trailing 7-day median, at least 3 days, on the
    last recorded day before the campaign). Percentage points."""
    rows = []
    for tr, g in daily.groupby("train"):
        s = g.set_index("day_index").dp_pct.sort_index() * 100
        for a, b in campaigns(legs.get(tr, [])):
            pre, post = s[(s.index >= a - 14) & (s.index < a)], s[(s.index > b) & (s.index <= b + 14)]
            seen = s[s.index < a]
            lvl = np.nan
            if len(seen):
                d_last = seen.index.max()
                w = seen[seen.index > d_last - 7]
                lvl = float(w.median()) if len(w) >= 3 else np.nan
            rows.append(dict(train=tr, first_leg=a, last_leg=b,
                             drop=float(pre.median() - post.median()) if len(pre) >= 3 and len(post) >= 3 else np.nan,
                             level_seen=lvl))
    return pd.DataFrame(rows)


def gaps(daily):
    out = {}
    for tr, g in daily.groupby("train"):
        d = g.day_index[g.day_index >= FROM_DAY].to_numpy()
        span = d.max() - d.min() + 1
        holes = np.diff(d) - 1
        out[tr] = dict(missing_share=float(1 - len(d) / span), gaps_of_3_days_or_more=int((holes >= 3).sum()))
    return out


def summarise(cyc, camp):
    """Point values from per-cycle and per-campaign numbers."""
    get = lambda k, kinds=None: np.array([c[k] for c in cyc if k in c and (kinds is None or c["kind"] in kinds)], float)
    v = {"cycle_length_days": np.median(get("length")),
         "share_of_cycles_over_60_days": np.mean(get("length") > 60),
         "rise_pp_per_week": np.median(get("rise_pp_per_week")),
         "noise_pp": np.median(get("noise_pp")),
         "cip_drop_pp": np.nanmedian(camp["drop"]) if len(camp) else np.nan,
         "level_at_cleaning_pp": np.nanmedian(camp["level_seen"]) if len(camp) else np.nan}
    pairs = np.array([p for c in cyc for p in c["chg_pairs"]], float)
    v["lag1_autocorr_of_daily_change"] = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 10 else np.nan
    for h in HORIZONS:
        v[f"mean_abs_change_{h}d_pp"] = float(np.mean([x for c in cyc for x in c[f"abs_change_{h}"]]))
    lv = np.array([x for c in cyc for x in c["levels"]], float)
    v["level_p10_pp"], v["level_p50_pp"], v["level_p90_pp"] = np.percentile(lv, [10, 50, 90])
    return {k: float(x) for k, x in v.items()}


def fingerprint(daily, legs, boot=True):
    cyc = [per_cycle(c) for c in cycles(daily, legs)]
    cyc = [c for c in cyc if c["n"] > 0]
    camp = campaign_numbers(daily, legs)
    camp = camp[camp.first_leg >= FROM_DAY]
    point = summarise(cyc, camp)
    res = {"value": point, "counts": {"cycles": len(cyc), "complete_cycles": sum(c["kind"] == "complete" for c in cyc),
                                      "campaigns": int(len(camp)), "evaluable_drops": int(camp["drop"].notna().sum())},
           "gaps": gaps(daily),
           "missing_share_all_trains": float(np.mean([v["missing_share"] for v in gaps(daily).values()]))}
    if boot:
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(N_BOOT):            # resample whole cycles and whole campaigns: days inside one are not independent
            cb = [cyc[i] for i in rng.integers(0, len(cyc), len(cyc))]
            kb = camp.iloc[rng.integers(0, len(camp), len(camp))]
            draws.append(summarise(cb, kb))
        draws = pd.DataFrame(draws)
        res["interval_90"] = {k: [float(draws[k].quantile(0.05)), float(draws[k].quantile(0.95))] for k in point}
    return res


def deposit(folder):
    d = pd.read_csv(Path(folder) / "derived_daily_indicator_table.csv").sort_values(["train", "day_index"])
    legs = {}
    for t, g in d.groupby("train"):
        legs[t] = sorted({int(round(v)) for v in pd.concat([(g.day_index - g.days_since_cip).dropna(),
                                                            (g.day_index + g.days_to_next_cip).dropna()])})
    return d[["train", "day_index", "dp_pct"]].reset_index(drop=True), legs


if __name__ == "__main__":
    daily, legs = deposit(sys.argv[1])
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(daily, legs)
    fp["legs"] = legs
    fp["campaigns"] = {t: campaigns(v) for t, v in legs.items()}
    (out / "fingerprint_real.json").write_text(json.dumps(fp, indent=1))
    print(json.dumps({k: fp[k] for k in ("counts", "gaps")}, indent=1))
    for k, v in fp["value"].items():
        lo, hi = fp["interval_90"][k]
        print(f"{k:34s} {v:8.3f}   90 % interval {lo:8.3f} - {hi:8.3f}")
