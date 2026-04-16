"""
src/features/rolling_stats.py
------------------------------
Rolling window statistics for the 24h-ahead PV forecaster.

These features capture recent trends and variability that lag values alone
cannot express.  For example:
  - A 3-hour rolling mean shows whether output is climbing or falling.
  - A 24-hour rolling std measures yesterday's cloud variability, which
    correlates with today's forecast uncertainty.

Windows chosen
--------------
  3h   — sub-hourly ramp detection (morning ramp-up / afternoon ramp-down)
  6h   — half-day trend
  24h  — same-time-yesterday mean and variance (strong predictor for solar)
  168h — same-time-last-week mean (longer seasonal context)

min_periods
-----------
Set to 50% of window so that the first few days of data aren't entirely NaN.
Rows with NaN features are dropped later during final matrix assembly.

Usage
-----
    from src.features.rolling_stats import add_rolling_features
    df = add_rolling_features(df, target_col="pv_ac_W")
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# (window_hours, min_periods)
_WINDOWS: list[tuple[int, int]] = [
    (3,   2),
    (6,   3),
    (24, 12),
    (168, 84),
]


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "pv_ac_W",
    windows: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, and max of target_col for each window.

    Column naming:  ``{target_col}_roll{w}_{stat}``
    e.g.            ``pv_ac_W_roll24_mean``

    Parameters
    ----------
    df : pd.DataFrame
        Must have a regular hourly UTC DatetimeIndex and contain target_col.
    target_col : str
        Column to compute rolling stats on.
    windows : list of (window_hours, min_periods), optional
        Defaults to _WINDOWS.

    Returns
    -------
    pd.DataFrame  with rolling feature columns appended.
    """
    windows = windows or _WINDOWS
    out     = df.copy()
    series  = out[target_col]
    n_added = 0

    for w, min_p in windows:
        prefix = f"{target_col}_roll{w}"
        roll   = series.rolling(window=w, min_periods=min_p)

        out[f"{prefix}_mean"] = roll.mean().astype("float32")
        out[f"{prefix}_std"]  = roll.std().astype("float32")
        out[f"{prefix}_min"]  = roll.min().astype("float32")
        out[f"{prefix}_max"]  = roll.max().astype("float32")
        n_added += 4

    logger.info(
        f"  Added {n_added} rolling features for '{target_col}' "
        f"(windows: {[w for w, _ in windows]}h)"
    )
    return out


def add_diff_features(
    df: pd.DataFrame,
    target_col: str = "pv_ac_W",
) -> pd.DataFrame:
    """
    Add first-order difference features to capture rate of change.

    Columns added
    -------------
    ``{target_col}_diff1``   — t vs t-1   (1h rate of change)
    ``{target_col}_diff24``  — t vs t-24  (same hour yesterday delta)

    These help the model detect whether output is rising or falling relative
    to the same hour in the previous period.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    out[f"{target_col}_diff1"]  = out[target_col].diff(1).astype("float32")
    out[f"{target_col}_diff24"] = out[target_col].diff(24).astype("float32")
    logger.info(f"  Added 2 diff features for '{target_col}' (diff1, diff24)")
    return out
