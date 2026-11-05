"""Phase 1B proof of concept - shared harness (declaration_poc.md).

Phase 1's functions (phase3/p3lib.py) are imported unchanged: the fouling law, the methods, XGBoost's settings, the scores
and the verdict. This file adds only what the simulated record needs: its pairs with the extra columns, the two records'
input lists, and the folds per plant (phase 1's lines).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
sys.path.insert(0, str(SCRATCH / "phase3"))
import p3lib as L  # noqa: E402  (unchanged phase-1 functions)

DATA = HERE / "data"
SHA = {"table.csv": "6dcb3b7734916a48b1cd8b4a39f936ce91e09c67be43e884e6e6af0930173eec",
       "cleanings.csv": "f0fbcfe3ecfa0ddee6d3bd0852720551d6b72b34acbe085fff67d7286f9baef2"}
PLANTS = {"P1": 4101, "P2": 4102, "P3": 4103}
CAUSES = ["temperature_c", "flux_lmh", "sdi", "turbidity_ntu"]
OPERATION = ["feed_flow_m3h", "perm_flow_m3h", "conc_flow_m3h", "recovery", "feed_tds_mgl", "perm_pressure_bar"]
RESULTS_RAW = ["feed_pressure_bar", "conc_pressure_bar", "dp_bar", "perm_tds_mgl"]
NO_LINK = ["ph", "orp_mv", "free_chlorine_mgl", "sbs_mgl", "antiscalant_mgl", "coagulant_mgl_fe", "cf_dp_bar"]
EXTRA = CAUSES + OPERATION + RESULTS_RAW + NO_LINK               # record 2 adds these to phase 1's inputs
SUFFIX = ["", "_m7", "_lag1", "_lag3", "_lag7", "_lag14"]
INSTRUMENTS = ["sdi", "turbidity_ntu"]                           # the control shifts or removes these two
METHODS = ["no change", "fouling law", "XGBoost", "hybrid"]      # the declared ones among p3lib's six


def load():
    for name, sha in SHA.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == sha, name
    t = pd.read_csv(DATA / "table.csv")
    c = pd.read_csv(DATA / "cleanings.csv")
    hidden = {"phi", "true_dp", "dp_true", "true_temperature", "z", "g"}
    assert not hidden & set(t.columns), "a hidden quantity is in the record"
    return t, c


def plant(t, c, p, shift_instruments=False):
    """one plant's record in phase 1's shape (day_index), and its cleaning log; optionally the control's shifted SDI and
    turbidity: the plant-wide daily series moved in time, circularly, by a random 60-300 days (default_rng([seed, 5]))."""
    d = t[t.plant == p].rename(columns={"day": "day_index"}).sort_values(["train", "day_index"]).reset_index(drop=True)
    led = c[c.plant == p].rename(columns={"day": "day_index"})
    if shift_instruments:
        k = int(np.random.default_rng([PLANTS[p], 5]).integers(60, 301))
        pw = pd.read_csv(DATA / "plantwide.csv")
        daily = pw[pw.plant == p].set_index("day").sort_index()[INSTRUMENTS]
        assert len(daily) == 772 and (daily.index == np.arange(772)).all()
        moved = pd.DataFrame(np.roll(daily.to_numpy(), k, axis=0), index=daily.index, columns=INSTRUMENTS)
        # check: before the shift, the plant-wide series is exactly what the record holds on its days
        chk = d[["day_index"] + INSTRUMENTS].merge(daily, left_on="day_index", right_index=True, suffixes=("", "_pw"))
        assert np.allclose(chk[INSTRUMENTS].to_numpy(), chk[[f"{c}_pw" for c in INSTRUMENTS]].to_numpy(), rtol=1e-5)
        d = d.drop(columns=INSTRUMENTS).merge(moved, left_on="day_index", right_index=True, how="left")
        d = d.sort_values(["train", "day_index"]).reset_index(drop=True)
        return d, led, k
    return d, led, 0


def build_days(d, led):
    """p3lib.build_days line for line, without the count of 10-min samples (not simulated), plus record 2's columns:
    today, the 7-day mean (recorded days in t-6..t) and 1, 3, 7, 14 days back (the nearest earlier day within 2)."""
    rows = []
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        cips = np.sort(led[led.train == tr].day_index.values)
        for t in g.index:
            last = cips[cips <= t].max() if (cips <= t).any() else -1
            cyc = g[(g.index > last) & (g.index <= t)]
            base = {"train": tr, "t": int(t), "days_since_cip": float(t - last) if last >= 0 else np.nan}
            for c in L.IND:
                base[c] = float(g.at[t, c])
                for lag in (1, 3, 7, 14):
                    base[f"{c}_lag{lag}"] = L.near(g[c], t - lag)
            recent = cyc[cyc.index > t - 14]
            recent3 = cyc[cyc.index > t - 7].tail(3)
            for c in L.TARGETS:
                base[f"{c}_slope14"] = float(np.polyfit(recent.index, recent[c], 1)[0]) if len(recent) >= 4 else np.nan
                base[f"{c}_state"] = float(recent3[c].median()) if len(recent3) else float(g.at[t, c])
            week = g[(g.index > t - 7) & (g.index <= t)]
            for c in EXTRA:
                base[c] = float(g.at[t, c])
                base[f"{c}_m7"] = float(week[c].mean())
                for lag in (1, 3, 7, 14):
                    base[f"{c}_lag{lag}"] = L.near(g[c], t - lag)
            for h in L.HORIZONS:
                valid = (t + h in g.index) and not ((cips > t) & (cips <= t + h)).any()
                r = dict(base, h=h, valid=valid)
                for c in L.TARGETS:
                    r[f"y_{c}"] = float(g.at[t + h, c]) if valid else np.nan
                    r[f"{c}_law"] = L.law_forecast(cyc, c, t, h)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["block"] = -1
    for i, (a, b) in enumerate(L.BLOCKS):
        df.loc[(df.t >= a) & (df.t < b), "block"] = i
    return df


def extra_feats(cols):
    return [f"{c}{s}" for c in cols for s in SUFFIX]


def feature_sets(days):
    allf = L.features(days)
    ex = set(extra_feats(EXTRA))
    rec1 = [f for f in allf if f not in ex]
    assert len(rec1) + len(ex) == len(allf) and "n_samples" not in allf
    return {"record 1": rec1, "record 2": allf}


def group_of(feature):
    """the declared column groups; record 1's inputs are the indicators' history (results), days since cleaning is the
    cleaning log (operation)."""
    for name, cols in (("causes", CAUSES), ("operation", OPERATION), ("results", RESULTS_RAW), ("no link", NO_LINK)):
        if any(feature == c or feature.startswith(c + "_") for c in cols):
            return name
    return "operation" if feature == "days_since_cip" else "results"


def folds(pairs):
    """phase 1's folds (run.py's lines): (test, h, block, held-out train, train rows, test rows)."""
    for h in L.HORIZONS:
        P = pairs[pairs.h == h]
        for b, (start, _) in enumerate(L.BLOCKS):
            train = P[P.t + h < start].sort_values(["t", "train"])
            test = P[P.block == b]
            yield "rolling", h, b, "", train, test
            for k_out in "ABC":
                tr_k, te_k = train[train.train != k_out], test[test.train == k_out]
                if te_k.empty:
                    continue
                yield "unseen_train", h, b, k_out, tr_k, te_k


def score_table(pred, methods, by=("record", "target", "h")):
    """pooled over plants: MAE (pp), skill against no change, the four block skills and phase 1's verdict."""
    out = []
    for key, g in pred.groupby(list(by)):
        nc = np.abs(g.y - g["no change"]).mean()
        for m in methods:
            if m not in g or g[m].isna().all():
                continue
            mae = np.abs(g.y - g[m]).mean()
            blocks = []
            for b in range(4):
                gb = g[g.block == b]
                blocks.append(100 * (1 - np.abs(gb.y - gb[m]).mean() / np.abs(gb.y - gb["no change"]).mean()))
            skill = 100 * (1 - mae / nc)
            out.append({**dict(zip(by, key)), "method": m, "n": len(g), "MAE_pp": 100 * mae, "MAE_no_change_pp": 100 * nc,
                        "skill_pct": skill, **{f"skill_b{b}": s for b, s in enumerate(blocks)},
                        "passes": L.verdict(skill, blocks)})
    return pd.DataFrame(out)
