"""
src/features/weather_patterns.py
----------------------------------
Identifies and encodes weather patterns in 5-min Solcast data for use
as ML features.

Three pattern types are captured:

1. Sky condition (categorical)
   Based on cloud_opacity and clearness index kt:
     Clear         cloud_opacity < 15  AND  kt > 0.75
     PartlyCloudy  15 ≤ cloud_opacity < 50  OR  0.4 < kt ≤ 0.75
     MostlyCloudy  50 ≤ cloud_opacity < 80  OR  0.15 < kt ≤ 0.4
     Overcast      cloud_opacity ≥ 80  OR  kt ≤ 0.15

2. Cloud variability (continuous — "unpredicted situations")
   Rolling standard deviation of kt over short windows (15 min, 30 min).
   High variability = rapidly fluctuating cloud cover → hard-to-predict
   ramp events that the physics model misses entirely.

3. Clearness trend (continuous)
   kt(t) − kt(t − 15 min): positive = clearing, negative = clouding over.
   Helps the model anticipate ramp direction.

Usage
-----
    from src.features.weather_patterns import add_weather_pattern_features
    df = add_weather_pattern_features(df)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Sky condition thresholds ──────────────────────────────────────────────────
_SKY_CONDITIONS = {
    "Clear":       0,
    "PartlyCloudy": 1,
    "MostlyCloudy": 2,
    "Overcast":    3,
}

# Encoded as int for XGBoost
_SKY_ENC = _SKY_CONDITIONS


def classify_sky_condition(
    cloud_opacity: pd.Series,
    kt:            pd.Series,
) -> pd.Series:
    """
    Classify each 5-min interval into a sky condition category.

    Uses both cloud_opacity (Solcast proprietary, 0-100%) and clearness
    index kt = GHI / clearsky_GHI (0-1) to produce a robust classification.
    When they disagree, a weighted rule prioritises cloud_opacity for night
    and kt for daytime (kt is meaningless at night).

    Returns
    -------
    pd.Series[int]  0=Clear, 1=PartlyCloudy, 2=MostlyCloudy, 3=Overcast
    """
    cond = pd.Series(1, index=cloud_opacity.index, dtype=np.int8)  # default PartlyCloudy

    # Overcast first (most restrictive)
    overcast = (cloud_opacity >= 80) | (kt <= 0.15)
    # Mostly cloudy
    mostly   = ((cloud_opacity >= 50) | (kt <= 0.40)) & ~overcast
    # Clear
    clear    = (cloud_opacity < 15) & (kt > 0.75)
    # Remaining = partly cloudy (default)

    cond[overcast] = 3
    cond[mostly]   = 2
    cond[clear]    = 0

    counts = cond.value_counts().sort_index()
    labels = {0: "Clear", 1: "PartlyCloudy", 2: "MostlyCloudy", 3: "Overcast"}
    logger.info("  Sky condition distribution (daytime):")
    for code, label in labels.items():
        n = int(counts.get(code, 0))
        logger.info(f"    {label:>14}: {n:>7,}  ({100*n/len(cond):.1f}%)")

    return cond


def add_weather_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all weather pattern features to a 5-min Solcast DataFrame.

    Expected input columns (NASA schema names):
        ALLSKY_SFC_SW_DWN_cal   — GHI
        CLRSKY_SFC_SW_DWN_cal   — clear-sky GHI
        cloud_opacity           — Solcast cloud opacity (0-100)
        air_temp / T2M_cal      — air temperature

    New columns added
    -----------------
    clearness_index    : kt = GHI / clearsky_GHI  (0–1.2)
    diffuse_fraction   : DHI / GHI  (0–1)
    sky_condition      : int  0=Clear 1=PartlyCloudy 2=MostlyCloudy 3=Overcast
    sky_is_clear       : bool shorthand
    sky_is_overcast    : bool shorthand
    kt_roll15_std      : 15-min rolling std of kt  (cloud transient proxy)
    kt_roll30_std      : 30-min rolling std of kt
    kt_trend_15min     : kt(t) - kt(t-3)  i.e. change over 15 min
    ghi_roll15_mean    : 15-min rolling mean GHI
    ghi_roll30_mean    : 30-min rolling mean GHI
    ghi_roll15_std     : 15-min rolling std GHI  (irradiance variability)
    cloud_opacity_trend: cloud_opacity(t) - cloud_opacity(t-3)

    Parameters
    ----------
    df : pd.DataFrame
        5-min UTC-indexed DataFrame with Solcast columns.

    Returns
    -------
    pd.DataFrame  with pattern feature columns appended.
    """
    out = df.copy()

    # ── Resolve column names (NASA schema vs raw Solcast) ────────────────────
    ghi_col    = "ALLSKY_SFC_SW_DWN_cal" if "ALLSKY_SFC_SW_DWN_cal" in out.columns else "ghi"
    clrsky_col = "CLRSKY_SFC_SW_DWN_cal" if "CLRSKY_SFC_SW_DWN_cal" in out.columns else "clearsky_ghi"
    dhi_col    = "ALLSKY_SFC_SW_DIFF_cal" if "ALLSKY_SFC_SW_DIFF_cal" in out.columns else "dhi"

    ghi    = out[ghi_col].clip(lower=0)
    clrsky = out[clrsky_col].replace(0, np.nan).clip(lower=1)

    # ── Clearness index ───────────────────────────────────────────────────────
    kt = (ghi / clrsky).clip(0, 1.2).fillna(0)
    out["clearness_index"] = kt.astype(np.float32)

    # ── Diffuse fraction ──────────────────────────────────────────────────────
    if dhi_col in out.columns:
        dhi = out[dhi_col].clip(lower=0)
        out["diffuse_fraction"] = (dhi / ghi.replace(0, np.nan)).clip(0, 1).fillna(0).astype(np.float32)

    # ── Sky condition classification ──────────────────────────────────────────
    if "cloud_opacity" in out.columns:
        cloud_op = out["cloud_opacity"]
    else:
        # Approximate from kt when cloud_opacity not available
        cloud_op = ((1 - kt) * 100).clip(0, 100)

    sky = classify_sky_condition(cloud_op, kt)
    out["sky_condition"]  = sky
    out["sky_is_clear"]   = (sky == 0).astype(np.int8)
    out["sky_is_overcast"] = (sky == 3).astype(np.int8)

    # ── Cloud opacity trend ───────────────────────────────────────────────────
    if "cloud_opacity" in out.columns:
        out["cloud_opacity_trend"] = out["cloud_opacity"].diff(3).fillna(0).astype(np.float32)

    # ── Clearness index variability (cloud transients / ramp events) ──────────
    # Rolling std over 3 (=15 min) and 6 (=30 min) periods
    out["kt_roll15_std"] = kt.rolling(3,  min_periods=2).std().fillna(0).astype(np.float32)
    out["kt_roll30_std"] = kt.rolling(6,  min_periods=3).std().fillna(0).astype(np.float32)

    # ── Clearness index trend (clearing vs clouding) ──────────────────────────
    out["kt_trend_15min"] = kt.diff(3).fillna(0).astype(np.float32)

    # ── GHI rolling statistics ────────────────────────────────────────────────
    out["ghi_roll15_mean"] = ghi.rolling(3,  min_periods=2).mean().fillna(0).astype(np.float32)
    out["ghi_roll30_mean"] = ghi.rolling(6,  min_periods=3).mean().fillna(0).astype(np.float32)
    out["ghi_roll15_std"]  = ghi.rolling(3,  min_periods=2).std().fillna(0).astype(np.float32)

    n_added = len(out.columns) - len(df.columns)
    logger.info(f"  Added {n_added} weather pattern features")
    return out


def add_pv_lag_features_5min(
    df:         pd.DataFrame,
    target_col: str = "pv_ac_kW",
    lags:       list[int] | None = None,
) -> pd.DataFrame:
    """
    Add 5-min-resolution lag features of PV power.

    Default lags (in 5-min periods):
      1  →  5 min ago
      3  → 15 min ago
      6  → 30 min ago
      12 →  1 hour ago
      24 →  2 hours ago
      288 → 24 hours ago (same time yesterday)

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str  column to lag (must exist in df)
    lags : list[int]  lag steps in 5-min periods

    Returns
    -------
    pd.DataFrame  with lag columns appended.
    """
    lags = lags or [1, 3, 6, 12, 24, 288]
    out  = df.copy()

    lag_minutes = {1: 5, 3: 15, 6: 30, 12: 60, 24: 120, 288: 1440}
    for k in lags:
        mins = lag_minutes.get(k, k * 5)
        col  = f"{target_col}_lag{mins}m"
        out[col] = out[target_col].shift(k).astype(np.float32)

    logger.info(
        f"  Added {len(lags)} PV lag features: "
        + ", ".join(f"t-{lag_minutes.get(k, k*5)}m" for k in lags)
    )
    return out
