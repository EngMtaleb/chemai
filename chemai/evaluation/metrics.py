"""Evaluation - including the parts that are usually skipped.

Three things are reported here that a bare R2 does not tell you:

  * error inside vs outside the operating envelope
  * whether the confidence intervals are actually calibrated
  * error on the high tail, where the model matters most and is worst
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate(model, df: pd.DataFrame, label: str = "") -> dict:
    """Headline metrics, plus the high-tail check.

    R2 is relative to the variance of the target, so comparing it across
    sets with different variance is misleading - always read MAE alongside.
    This bit the project once already; see docs/VALIDATION.md section 3.
    """
    y = np.asarray(df[model.cfg.target], dtype=float)
    p = model.predict(df)
    resid = y - p

    hi = y > np.quantile(y, 0.90)
    out = {
        "label": label,
        "n": int(len(df)),
        "target_std": round(float(np.std(y, ddof=1)), 3),
        "r2": round(float(r2_score(y, p)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 3),
        "mae_high_decile": round(float(mean_absolute_error(y[hi], p[hi])), 3),
        "mae_rest": round(float(mean_absolute_error(y[~hi], p[~hi])), 3),
        "residual_mean": round(float(resid.mean()), 3),
        "residual_pred_corr": round(float(np.corrcoef(p, resid)[0, 1]), 3),
    }
    out["high_tail_penalty"] = round(out["mae_high_decile"] / out["mae_rest"], 2)
    return out


def envelope_report(model, df: pd.DataFrame) -> dict:
    """Does the envelope detector flag the right samples?

    A safety mechanism that flags the wrong points is worse than none.
    If error outside the envelope is not clearly worse than inside, the
    detector is not doing its job and should not be trusted.
    """
    from chemai.features import build_features
    X = build_features(df, model.cfg)
    d = model.mahalanobis(X)
    inside = d <= model.envelope_threshold_

    y = np.asarray(df[model.cfg.target], dtype=float)
    p = model.predict(df)

    rep = {
        "threshold": round(float(model.envelope_threshold_), 3),
        "flagged_fraction": round(float((~inside).mean()), 4),
        "n_inside": int(inside.sum()),
        "n_outside": int((~inside).sum()),
        "mae_inside": round(float(mean_absolute_error(y[inside], p[inside])), 3)
        if inside.any() else None,
    }
    if (~inside).any():
        rep["mae_outside"] = round(float(mean_absolute_error(y[~inside], p[~inside])), 3)
        rep["ratio"] = round(rep["mae_outside"] / rep["mae_inside"], 2)
    return rep


def interval_coverage(model, df: pd.DataFrame) -> dict:
    """Are the stated confidence intervals honest?

    An interval claiming 95% while covering 70% is worse than no interval,
    because it invites false confidence.
    """
    y = np.asarray(df[model.cfg.target], dtype=float)
    p = model.predict(df)
    half = 1.96 * model.residual_sigma_
    covered = (y >= p - half) & (y <= p + half)
    return {
        "nominal": model.cfg.confidence,
        "empirical": round(float(covered.mean()), 3),
        "mean_width": round(float(2 * half), 3),
        "residual_sigma": round(float(model.residual_sigma_), 3),
    }
