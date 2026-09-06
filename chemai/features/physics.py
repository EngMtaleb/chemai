"""Physics-derived features.

The principle behind this module: before fitting anything, ask what
mathematical form the physics imposes on the relationship. Feeding raw
measurements to a model when the governing equation says otherwise
throws away information that costs nothing to keep.
"""
import numpy as np
import pandas as pd

from chemai.config import SoftSensorConfig

FEATURE_NAMES = ["invT", "pressure"]


def antoine_form(temperature: np.ndarray | pd.Series, scale: float = 1000.0):
    """Return 1000/T - the form Antoine's equation requires.

        log10(Psat) = A - B / (C + T)

    Vapour pressure depends on the INVERSE of temperature. The same
    structure appears in the Wilson K-value correlation and in most of
    the vapour-pressure equations published since 1841.

    On the industrial dataset this form reduces residual error by roughly
    40% against raw temperature, with no additional variable.

    Note: this feature has a very small variance (~0.08 against ~14 for
    pressure). Any penalised regressor MUST be preceded by a scaler, or
    the penalty will suppress it. See docs/EDA.md section 7.
    """
    t = np.asarray(temperature, dtype=float)
    if np.any(t <= 0):
        raise ValueError("temperature must be positive and absolute (K or R)")
    return scale / t


def build_features(df: pd.DataFrame, cfg: SoftSensorConfig | None = None) -> pd.DataFrame:
    """Build the two-feature matrix: inverse temperature and pressure.

    Two features match the performance of seven on this dataset - the extra
    five contribute under 0.005. Fewer sensors means less calibration, less
    maintenance and fewer failure points. See docs/EDA.md section 6.
    """
    cfg = cfg or SoftSensorConfig()
    for col in (cfg.temperature_col, cfg.pressure_col):
        if col not in df.columns:
            raise ValueError(f"required column '{col}' missing")
    return pd.DataFrame({
        "invT": antoine_form(df[cfg.temperature_col]),
        "pressure": np.asarray(df[cfg.pressure_col], dtype=float),
    }, index=df.index)
