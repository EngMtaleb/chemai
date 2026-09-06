"""Soft sensor: an estimate, its uncertainty, and an envelope check.

A regression model returns a number. A soft sensor returns a number, how
confident it is, and whether the question was inside the range it was
validated on. The last part is what stops it being confidently wrong.
"""
from dataclasses import dataclass, asdict
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemai.config import SoftSensorConfig
from chemai.features import build_features, FEATURE_NAMES

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    """What the sensor returns. Three fields, not one."""
    value: float
    lower: float
    upper: float
    in_envelope: bool
    distance: float
    warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SoftSensor:
    """Ridge on physics-formed features, with bootstrap intervals and a
    Mahalanobis operating envelope.

    The scaler is not optional. The inverse-temperature feature has ~165x
    less variance than pressure; without scaling, Ridge suppresses it and
    performance drops from 0.98 to 0.52. See docs/EDA.md section 7.
    """

    def __init__(self, cfg: SoftSensorConfig | None = None):
        self.cfg = cfg or SoftSensorConfig()
        self.pipeline = None
        self.residual_sigma_: float | None = None
        self.envelope_mean_: np.ndarray | None = None
        self.envelope_inv_cov_: np.ndarray | None = None
        self.envelope_threshold_: float | None = None

    # ---------------------------------------------------------------- fit

    def _new_pipeline(self):
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.cfg.ridge_alpha),
        )

    def fit(self, df: pd.DataFrame) -> "SoftSensor":
        X = build_features(df, self.cfg)
        y = np.asarray(df[self.cfg.target], dtype=float)

        self.pipeline = self._new_pipeline().fit(X, y)
        self._fit_uncertainty(X, y)
        self._fit_envelope(X)

        log.info("fitted on %d rows, residual sigma %.3f",
                 len(df), self.residual_sigma_)
        return self

    def _fit_uncertainty(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Bootstrap the residual scatter.

        Two sources of error: how much the fit itself could differ
        (model uncertainty) and irreducible scatter (residual noise).
        On this dataset the second dominates - more data would help little.
        """
        rng = np.random.default_rng(self.cfg.random_state)
        sigmas = []
        n = len(X)
        for _ in range(self.cfg.n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            m = self._new_pipeline().fit(X.iloc[idx], y[idx])
            resid = y[idx] - m.predict(X.iloc[idx])
            sigmas.append(np.std(resid, ddof=2))
        self.residual_sigma_ = float(np.mean(sigmas))

    def _fit_envelope(self, X: pd.DataFrame) -> None:
        """Mahalanobis envelope over the training features.

        Mahalanobis rather than per-feature ranges because it accounts for
        the correlation between temperature and pressure - it flags
        implausible COMBINATIONS, not only individually extreme values.
        """
        A = np.asarray(X, dtype=float)
        self.envelope_mean_ = A.mean(axis=0)
        self.envelope_inv_cov_ = np.linalg.inv(np.cov(A.T))
        self.envelope_threshold_ = float(
            np.sqrt(chi2.ppf(self.cfg.envelope_percentile, df=A.shape[1]))
        )

    # ------------------------------------------------------------ predict

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("call fit() before predicting")

    def mahalanobis(self, X) -> np.ndarray:
        self._check_fitted()
        D = np.asarray(X, dtype=float) - self.envelope_mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.envelope_inv_cov_, D))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(build_features(df, self.cfg))

    def predict_one(self, temperature: float, pressure: float) -> Prediction:
        """Single estimate with interval and envelope flag."""
        self._check_fitted()
        df = pd.DataFrame({
            self.cfg.temperature_col: [temperature],
            self.cfg.pressure_col: [pressure],
        })
        X = build_features(df, self.cfg)
        value = float(self.pipeline.predict(X)[0])
        dist = float(self.mahalanobis(X)[0])
        inside = dist <= self.envelope_threshold_

        z = 1.96 if abs(self.cfg.confidence - 0.95) < 1e-9 else None
        if z is None:
            from scipy.stats import norm
            z = float(norm.ppf(0.5 + self.cfg.confidence / 2))
        half = z * self.residual_sigma_

        return Prediction(
            value=round(value, 2),
            lower=round(value - half, 2),
            upper=round(value + half, 2),
            in_envelope=bool(inside),
            distance=round(dist, 2),
            warning=None if inside else
            "input outside validated operating envelope - estimate unreliable",
        )
