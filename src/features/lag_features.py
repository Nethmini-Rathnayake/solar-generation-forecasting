"""
src/features/lag_features.py
------------------------------
Lag features and 24h-ahead multi-step target construction for the
XGBoost direct multi-step forecaster.

Forecasting strategy: Direct Multi-Step (DMS)
----------------------------------------------
For 24h-ahead forecasting we use the "Direct" strategy: train one
XGBoost model per forecast horizon h ∈ {1, 2, …, 24}.

Each model at horizon h predicts:
    ŷ(t+h) = f_h(X_t)

where X_t is the feature vector at time t (all features known at t,
including lags of pv_ac_W up to the present).

Why Direct and not Recursive?
    Recursive: use ŷ(t+1) as a lag input to predict ŷ(t+2), etc.
        → Error accumulates: each prediction feeds into the next.
    Direct:    each horizon gets its own model with clean, observed lags.
        → No error propagation; each model is independently optimal.
    For solar (strong periodicity, well-behaved physics), Direct
    consistently outperforms Recursive in the literature.

Lag groups chosen
-----------------
  Recent (short memory):   t-1  … t-6   (last 6 hours)
  Daily lag:               t-24          (same hour yesterday)
  Two-day lag:             t-48          (same hour two days ago)
  Weekly lag:              t-168         (same hour last week — captures
                                         weekly cloud patterns / rain cycles)

These cover the dominant autocorrelation structure of tropical solar data
without creating hundreds of redundant features.

Target columns
--------------
  target_h1  … target_h24 : pv_ac_W values at t+1 … t+24
  Only rows where ALL 24 targets are non-NaN are kept (no target leakage).

Usage
-----
    from src.features.lag_features import add_lag_features, build_target_matrix
    df = add_lag_features(df, target_col="pv_ac_W")
    df = build_target_matrix(df, target_col="pv_ac_W", horizons=24)
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lag steps in hours
_LAGS: list[int] = [1, 2, 3, 4, 5, 6, 24, 48, 168]

# Number of forecast horizons
_N_HORIZONS: int = 24


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "pv_ac_W",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged values of target_col as new columns.

    Each column is named ``{target_col}_lag{k}`` where k is the lag in hours.
    Rows without enough history (first max(lags) rows) will have NaN lags —
    they are dropped later when the full feature matrix is finalised.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a regular hourly UTC DatetimeIndex.
    target_col : str
        Column to lag. Default: "pv_ac_W".
    lags : list[int], optional
        Lag steps in hours. Defaults to _LAGS = [1,2,3,4,5,6,24,48,168].

    Returns
    -------
    pd.DataFrame
        Original columns + lag columns.
    """
    lags = lags or _LAGS
    out  = df.copy()

    for k in lags:
        col_name     = f"{target_col}_lag{k}"
        out[col_name] = out[target_col].shift(k)

    logger.info(f"  Added {len(lags)} lag features for '{target_col}': {lags}")
    return out


def build_target_matrix(
    df: pd.DataFrame,
    target_col: str = "pv_ac_W",
    horizons:   int = _N_HORIZONS,
) -> pd.DataFrame:
    """
    Add forward-shifted target columns for each forecast horizon h=1..horizons.

    Column name: ``target_h{h}`` = pv_ac_W at t+h.

    These are what the models learn to predict. Rows near the end of the
    dataset that cannot form a full 24-step target window are NaN and will
    be dropped by the caller.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    horizons : int
        Number of forecast steps (default 24 → 24h-ahead).

    Returns
    -------
    pd.DataFrame  with ``target_h1`` … ``target_h{horizons}`` appended.
    """
    out = df.copy()
    for h in range(1, horizons + 1):
        out[f"target_h{h}"] = out[target_col].shift(-h)

    logger.info(
        f"  Added {horizons} target columns (target_h1 … target_h{horizons})"
    )
    return out


def get_feature_cols(df: pd.DataFrame, target_col: str = "pv_ac_W") -> list[str]:
    """
    Return the list of input feature column names (excludes target_h* columns
    and the raw target_col itself, which would be data leakage at t).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    target_col : str

    Returns
    -------
    list[str]
    """
    exclude = {target_col} | {c for c in df.columns if c.startswith("target_h")}
    return [c for c in df.columns if c not in exclude]


def get_target_cols(horizons: int = _N_HORIZONS) -> list[str]:
    """Return the list of target column names for h=1..horizons."""
    return [f"target_h{h}" for h in range(1, horizons + 1)]
