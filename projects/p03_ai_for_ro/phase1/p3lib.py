"""Phase 3 - the declared tests, as functions (see declaration.md). Used by run.py and copied into the notebook."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

IND = ["dp_pct", "flux_pct", "sp_pct", "ndp_pct"]
TARGETS = ["dp_pct", "flux_pct"]
HORIZONS = (1, 3, 7, 14)
BLOCKS = [(550, 600), (600, 650), (650, 700), (700, 772)]
METHODS = ["no change", "smoothed no change", "trend", "fouling law", "XGBoost", "hybrid"]
SEEDS = (0, 1, 2, 3, 4)
XGB = dict(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
           n_jobs=2, verbosity=0)


# ------------------------------------------------------------------ tables
def near(s, day, tol=2):
    for k in range(tol + 1):
        if day - k in s.index:
            return s[day - k]
    return np.nan


def law_forecast(cyc, c, t, h, min_pts=5):
    """fouling law: linear growth since the last cleaning, fitted on the cycle so far, extrapolated to t + h."""
    if len(cyc) < min_pts:
        return np.nan
    b, a = np.polyfit(cyc.index.values.astype(float), cyc[c].values, 1)
    return float(np.clip(a + b * (t + h), 0, 1))


def build_days(d, led):
    """one row per (train, day t, horizon h) with the inputs known on day t; y and `valid` where a pair exists."""
    rows = []
    for tr, g in d.groupby("train"):
        g = g.set_index("day_index").sort_index()
        cips = np.sort(led[led.train == tr].day_index.values)
        for t in g.index:
            last = cips[cips <= t].max() if (cips <= t).any() else -1
            cyc = g[(g.index > last) & (g.index <= t)]
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
                valid = (t + h in g.index) and not ((cips > t) & (cips <= t + h)).any()
                r = dict(base, h=h, valid=valid)
                for c in TARGETS:
                    r[f"y_{c}"] = float(g.at[t + h, c]) if valid else np.nan
                    r[f"{c}_law"] = law_forecast(cyc, c, t, h)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["block"] = -1
    for i, (a, b) in enumerate(BLOCKS):
        df.loc[(df.t >= a) & (df.t < b), "block"] = i
    return df


def features(df):
    return [c for c in df.columns if c not in ("train", "t", "h", "valid", "block") and not c.startswith("y_")]


# ------------------------------------------------------------------ methods
def simple(df, c, h):
    """the four methods that need no training: arrays aligned with df."""
    now = df[c].to_numpy()
    trend = np.clip(now + h * np.nan_to_num(df[f"{c}_slope14"].to_numpy()), 0, 1)
    law = df[f"{c}_law"].to_numpy()
    return {"no change": now, "smoothed no change": df[f"{c}_state"].to_numpy(),
            "trend": trend, "fouling law": np.where(np.isnan(law), now, law)}


def fit_xgb(X, y, seeds=SEEDS):
    from xgboost import XGBRegressor
    return [XGBRegressor(random_state=s, **XGB).fit(X, y) for s in seeds]


def predict_xgb(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def learned(train, tests, c, h, feats):
    """XGBoost (on the change) and hybrid (on the fouling law's error): fit on `train`, predict every frame in `tests`."""
    X = train[feats].to_numpy(float)
    now = train[c].to_numpy(); y = train[f"y_{c}"].to_numpy()
    law = simple(train, c, h)["fouling law"]
    m_x, m_h = fit_xgb(X, y - now), fit_xgb(X, y - law)
    out = []
    for te in tests:
        Xt = te[feats].to_numpy(float); s = simple(te, c, h)
        out.append({"XGBoost": np.clip(s["no change"] + predict_xgb(m_x, Xt), 0, 1),
                    "hybrid": np.clip(s["fouling law"] + predict_xgb(m_h, Xt), 0, 1)})
    return out


def forecasts(train, tests, c, h, feats):
    """all six methods for each test frame."""
    lr = learned(train, tests, c, h, feats)
    return [{**simple(te, c, h), **l} for te, l in zip(tests, lr)]


# ------------------------------------------------------------------ scores
def scores(y, now, preds):
    base = np.mean(np.abs(y - preds["no change"]))
    out = {}
    for m, p in preds.items():
        mae = np.mean(np.abs(y - p))
        out[m] = {"MAE_pp": 100 * mae, "skill_pct": 100 * (1 - mae / base), "R2_change": r2_score(y - now, p - now)}
    return out


def verdict(pooled_skill, block_skills):
    return bool(pooled_skill >= 5.0 and sum(s > 0 for s in block_skills) >= 3)
