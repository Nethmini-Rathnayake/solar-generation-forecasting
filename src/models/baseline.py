"""
src/models/baseline.py
-----------------------
Naive forecast baselines for benchmarking the XGBoost DMS model.

Two baselines are implemented:

1. Day-ahead persistence (standard solar baseline)
   ŷ(t+h) = y(t + h − 24)   →  "same time yesterday"
   This is the most common benchmark in solar forecasting literature.
   For h ≤ 24 this is always the observation from exactly 24h before t+h.

2. Climatological mean (hour-of-day average from the training set)
   ŷ(t+h) = mean(y | hour(t+h) == hour)
   Useful lower bound: a model must beat the hourly mean to add value.

Usage
-----
    from src.models.baseline import day_ahead_persistence, climatological_mean

    pers  = day_ahead_persistence(df_test, target_col="pv_ac_W")
    clim  = climatological_mean(df_train, df_test, target_col="pv_ac_W")
    # Both return DataFrames with columns pred_h1 … pred_h24
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_N_HORIZONS = 24


def day_ahead_persistence(
    df:         pd.DataFrame,
    target_col: str = "pv_ac_W",
    n_horizons: int = _N_HORIZONS,
) -> pd.DataFrame:
    """
    Day-ahead persistence: ŷ(t+h) = y(t + h − 24).

    For each row t in df, the prediction at horizon h is the observed value
    24 hours before t+h — i.e., y(t + h - 24).  When h ≤ 24 this equals
    y(t + (h-24)), which for h=1..24 runs from t-23 to t.

    All predictions clipped to ≥ 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target_col and have a UTC DatetimeIndex.
    target_col : str
    n_horizons : int

    Returns
    -------
    pd.DataFrame
        Shape (len(df), n_horizons). Columns: pred_h1 … pred_h{n_horizons}.
        Index matches df.index. NaN where insufficient history.
    """
    preds = {}
    series = df[target_col]

    for h in range(1, n_horizons + 1):
        # y(t + h - 24): shift by (24 - h) backwards from the target position
        # Equivalent: for the series shifted forward by h, shift it back 24 to get yesterday
        lag = 24 - h            # positive lag = past, negative = future
        if lag >= 0:
            preds[f"pred_h{h}"] = series.shift(lag).values.clip(min=0.0)
        else:
            # h > 24: use 48h back
            preds[f"pred_h{h}"] = series.shift(48 - h).values.clip(min=0.0)

    return pd.DataFrame(preds, index=df.index)


def climatological_mean(
    df_train:   pd.DataFrame,
    df_test:    pd.DataFrame,
    target_col: str = "pv_ac_W",
    n_horizons: int = _N_HORIZONS,
) -> pd.DataFrame:
    """
    Climatological (hour-of-day mean) baseline.

    For each horizon h, the predicted value is the mean of target_col for all
    training rows whose hour equals hour(t + h).

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (used to compute hourly means).
    df_test : pd.DataFrame
        Test data (index used for predictions).
    target_col : str
    n_horizons : int

    Returns
    -------
    pd.DataFrame
        Shape (len(df_test), n_horizons). Columns: pred_h1 … pred_h{n_horizons}.
    """
    # Compute hourly climatology from training set
    hourly_mean = (
        df_train[target_col]
        .groupby(df_train.index.hour)
        .mean()
    )

    preds = {}
    for h in range(1, n_horizons + 1):
        # Hour of t+h
        target_hours = (df_test.index + pd.Timedelta(hours=h)).hour
        preds[f"pred_h{h}"] = hourly_mean.reindex(target_hours).values.clip(min=0.0)

    logger.info(
        f"Climatological baseline: hourly mean from {len(df_train):,} train rows → "
        f"applied to {len(df_test):,} test rows"
    )
    return pd.DataFrame(preds, index=df_test.index)
