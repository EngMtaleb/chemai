"""Loading and splitting - with the checks that should never be skipped."""
from pathlib import Path
import numpy as np
import pandas as pd

from chemai.config import DATA_DIR, SoftSensorConfig


def load_distillation_tower(path: Path | str | None = None,
                            cfg: SoftSensorConfig | None = None) -> pd.DataFrame:
    """Load the industrial distillation dataset, sorted by date.

    Sorting matters: the split is temporal, and an unsorted frame would
    make the period boundaries meaningless.
    """
    cfg = cfg or SoftSensorConfig()
    path = Path(path) if path else DATA_DIR / cfg.data_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Data is not committed to the repository - "
            "see README for the source."
        )
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    if cfg.target not in df.columns:
        raise ValueError(f"target column '{cfg.target}' missing from {path.name}")
    if df.isna().any().any():
        raise ValueError("unexpected missing values - this dataset should have none")
    return df


def check_derived_columns(df: pd.DataFrame, tol: float = 1e-6) -> dict[str, str]:
    """Find columns that are exact transformations of other columns.

    Returns a mapping {derived_column: source_column} for anything matching
    within `tol`. Run this on any new dataset before modelling: with both
    forms present, feature attribution names whichever the model happened
    to pick, and any physical interpretation built on it is meaningless.
    """
    found: dict[str, str] = {}
    num = df.select_dtypes("number")
    for col in num.columns:
        for src in num.columns:
            if col == src:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1000.0 / num[src]
            if np.isfinite(inv).all() and np.abs(num[col] - inv).max() < tol:
                found[col] = f"1000/{src}"
                break
    return found


def add_periods(df: pd.DataFrame, cfg: SoftSensorConfig | None = None) -> pd.DataFrame:
    """Label operating campaigns.

    The boundaries are gaps in the record, not arbitrary cuts. Each campaign
    is a different operating regime - see docs/EDA.md section 3.
    """
    cfg = cfg or SoftSensorConfig()
    out = df.copy()
    out["period"] = 0
    for i, brk in enumerate(cfg.period_breaks, start=1):
        out.loc[brk:, "period"] = i
    return out


def split_by_period(df: pd.DataFrame, cfg: SoftSensorConfig | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test split by operating campaign - never randomly.

    A random split on this data would place samples from the same campaign
    on both sides, and the reported score would not reflect performance on
    a new operating regime.
    """
    cfg = cfg or SoftSensorConfig()
    if "period" not in df.columns:
        df = add_periods(df, cfg)
    train = df[df.period == cfg.train_period]
    test = df[df.period == cfg.test_period]
    if train.empty or test.empty:
        raise ValueError("train or test period is empty - check period_breaks")
    return train, test
