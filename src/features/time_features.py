"""
src/features/time_features.py
------------------------------
Calendar and solar-position time features for the 24h-ahead PV forecaster.

Why cyclic encoding?
---------------------
XGBoost sees features as independent scalars. Raw hour (0–23) gives the model
no way to know that hour 23 and hour 0 are adjacent. Sine/cosine encoding
maps any periodic quantity θ onto the unit circle:
    sin_hour = sin(2π × hour / 24)
    cos_hour = cos(2π × hour / 24)
so the Euclidean distance between hour 23 and hour 0 is small (as expected).

Features generated
------------------
  Cyclic calendar    : sin/cos of hour-of-day, day-of-year, day-of-week, month
  Raw calendar       : hour, day_of_year, month, weekday, is_weekend
  Solar geometry     : solar_elevation, solar_azimuth, cos_solar_zenith
                       (already in synthetic PV file; re-derived here for
                        full 6-year coverage and for rows without simulation)
  Clearness index    : kt = GHI_cal / CLRSKY_SFC_SW_DWN_cal  (0–1 cloud cover proxy)

Usage
-----
    from src.features.time_features import add_time_features
    df = add_time_features(df, cfg)
"""

import numpy as np
import pandas as pd
import pvlib

from src.utils.logger import get_logger

logger = get_logger(__name__)

_LAT = 6.7912
_LON = 79.9005
_ELEV_M = 20


def add_time_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Add calendar and solar-geometry features to a UTC-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a UTC-aware DatetimeIndex.
    cfg : dict, optional
        Config dict. Not required — site coordinates are hardcoded to match
        the project site.

    Returns
    -------
    pd.DataFrame
        Original columns + time/solar features.
    """
    assert df.index.tz is not None, "DatetimeIndex must be timezone-aware (UTC)"
    out = df.copy()

    # ── Raw calendar ──────────────────────────────────────────────────────────
    out["hour"]        = out.index.hour.astype(np.int8)
    out["day_of_year"] = out.index.day_of_year.astype(np.int16)
    out["month"]       = out.index.month.astype(np.int8)
    out["weekday"]     = out.index.weekday.astype(np.int8)   # 0=Mon … 6=Sun
    out["is_weekend"]  = (out["weekday"] >= 5).astype(np.int8)

    # ── Cyclic calendar ───────────────────────────────────────────────────────
    # Hour of day  (period = 24)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)

    # Day of year  (period = 365.25)
    out["sin_doy"]  = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_doy"]  = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    # Month  (period = 12)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)

    # Day of week  (period = 7)
    out["sin_dow"]   = np.sin(2 * np.pi * out["weekday"] / 7)
    out["cos_dow"]   = np.cos(2 * np.pi * out["weekday"] / 7)

    # ── Solar geometry ────────────────────────────────────────────────────────
    loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV_M)
    solar_pos = loc.get_solarposition(out.index)

    out["solar_elevation_deg"] = solar_pos["elevation"].astype(np.float32)
    out["solar_azimuth_deg"]   = solar_pos["azimuth"].astype(np.float32)
    # cos(zenith) is proportional to clear-sky irradiance — useful nonlinear feature
    out["cos_solar_zenith"]    = np.cos(
        np.deg2rad(solar_pos["apparent_zenith"].clip(upper=90))
    ).astype(np.float32)
    # Night mask (elevation ≤ 0)
    out["is_daytime"] = (solar_pos["elevation"] > 0).astype(np.int8)

    # ── Clearness index ───────────────────────────────────────────────────────
    # kt = GHI / clear-sky GHI — measures cloud cover (0 = overcast, 1 = clear)
    if "ALLSKY_SFC_SW_DWN_cal" in out.columns and "CLRSKY_SFC_SW_DWN_cal" in out.columns:
        clrsky = out["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan)
        out["clearness_index"] = (
            out["ALLSKY_SFC_SW_DWN_cal"] / clrsky
        ).clip(0, 1.2).astype(np.float32)

    n_features = len(out.columns) - len(df.columns)
    logger.info(f"  Added {n_features} time/solar features")
    return out
