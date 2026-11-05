"""declaration_v15.md: phase 1B's harness (phase3b/poc/poc_lib.py, imported unchanged) on the new table, with the three
plants as one dataset. A train is named by plant and letter (P1A ... P3C); nothing else in the pairs changes."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SP = HERE.parent
sys.path.insert(0, str(SP / "phase3b" / "poc"))
import poc_lib as P  # noqa: E402

L = P.L
P.DATA = HERE / "data"                                   # poc_lib.plant() reads plantwide.csv from here
SHA = dict(line.split()[::-1] for line in (P.DATA / "table.sha256").read_text().splitlines())
H, C = 7, "dp_pct"
METHODS = ["no change", "fouling law", "XGBoost", "hybrid"]


def load():
    for name, sha in SHA.items():
        assert hashlib.sha256((P.DATA / name).read_bytes()).hexdigest() == sha, name
    t = pd.read_csv(P.DATA / "table.csv")
    c = pd.read_csv(P.DATA / "cleanings.csv")
    assert not {"phi", "true_dp", "dp_true", "true_temperature", "z", "g"} & set(t.columns)
    return t, c


def plant_shifted(t, c, p):
    """the control: SDI and turbidity, each column's yearly cycle taken out (log x = a + b sin + c cos + r, fitted on the
    plant's 772 days), the rest moved circularly by the declared k (default_rng([seed, 5]), 60-300 days), the cycle put
    back in its own place: x'(d) = exp(a + b sin(2 pi d/365) + c cos(2 pi d/365) + r(d - k))."""
    k = int(np.random.default_rng([P.PLANTS[p], 5]).integers(60, 301))
    d = t[t.plant == p].rename(columns={"day": "day_index"}).sort_values(["train", "day_index"]).reset_index(drop=True)
    led = c[c.plant == p].rename(columns={"day": "day_index"})
    pw = pd.read_csv(P.DATA / "plantwide.csv")
    daily = pw[pw.plant == p].set_index("day").sort_index()[P.INSTRUMENTS]
    assert len(daily) == 772 and (daily.index == np.arange(772)).all()
    x = np.arange(772)
    X = np.column_stack([np.ones(772), np.sin(2 * np.pi * x / 365.0), np.cos(2 * np.pi * x / 365.0)])
    moved = {}
    for col in P.INSTRUMENTS:
        y = np.log(daily[col].to_numpy())
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        moved[col] = np.exp(X @ beta + np.roll(y - X @ beta, k))
    chk = d[["day_index"] + P.INSTRUMENTS].merge(daily, left_on="day_index", right_index=True, suffixes=("", "_pw"))
    assert np.allclose(chk[P.INSTRUMENTS].to_numpy(), chk[[f"{q}_pw" for q in P.INSTRUMENTS]].to_numpy(), rtol=1e-5)
    d = d.drop(columns=P.INSTRUMENTS).merge(pd.DataFrame(moved, index=daily.index), left_on="day_index",
                                            right_index=True, how="left")
    return d.sort_values(["train", "day_index"]).reset_index(drop=True), led, k


def merged(t, c, shifted=False):
    """the three plants' pairs as one dataset (every horizon kept; the run uses 7 days): (days, feats, per-plant pieces)."""
    parts, feats, pieces = [], None, {}
    for p in P.PLANTS:
        d, led, k = plant_shifted(t, c, p) if shifted else (*P.plant(t, c, p)[:2], 0)
        days = P.build_days(d, led)
        f = P.feature_sets(days)["record 2"]
        assert feats is None or f == feats
        feats = f
        days = days.assign(train=p + days.train)
        parts.append(days)
        pieces[p] = (d, led, k)
    return pd.concat(parts, ignore_index=True), feats, pieces


def folds(pairs):
    """phase 1's rolling origin on the merged pairs, 7 days ahead: (block, train rows, test rows)."""
    P7 = pairs[pairs.h == H]
    for b, (start, _) in enumerate(L.BLOCKS):
        yield b, P7[P7.t + H < start].sort_values(["t", "train"]), P7[P7.block == b]
