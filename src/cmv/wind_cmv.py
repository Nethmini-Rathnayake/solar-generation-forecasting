"""
src/cmv/wind_cmv.py
──────────────────────────────────────────────────────────────────────────────
Wind-based Cloud Motion Vector feature engineering.

Since satellite-derived CMV is unavailable (Himawari-8 data archived at
~4 KB/s), we derive equivalent cloud-motion features from surface wind data
(Open-Meteo ERA5-derived) scaled to the 850 hPa cloud level.

Physics basis
─────────────
In the tropical marine boundary layer over Sri Lanka, the ratio of 850 hPa
wind speed to 10 m wind speed is approximately 1.8 (Hastenrath 1985;
WMO 2012 Marine Surface Wind guide).  Direction is well-coupled (within ±15°)
because the boundary layer over a warm sea is well-mixed.

The shadow arrival features are physically identical to the Himawari-derived
CMV features in shadow_predictor.py — only the wind source changes.

Feature vector (per 5-min interval after upsampling)
─────────────────────────────────────────────────────
    cloud_speed_kmh        — 850 hPa equivalent wind speed [km/h]
    cloud_direction_deg    — direction clouds move toward [° from N, CW]
    shadow_offset_km       — solar-geometry shadow displacement [km]
    solar_zenith_deg       — solar zenith angle [°]
    solar_azimuth_deg      — solar azimuth angle [°]
    shadow_arrival_5km     — minutes until shadow at 5 km arrives
    shadow_arrival_10km    — minutes until shadow at 10 km arrives
    shadow_arrival_20km    — minutes until shadow at 20 km arrives
    shadow_arrival_40km    — minutes until shadow at 40 km arrives
    opacity_lag_5km        — cloud opacity lag corresponding to 5 km upstream
    opacity_lag_10km       — cloud opacity lag corresponding to 10 km upstream
    opacity_lag_20km       — cloud opacity lag corresponding to 20 km upstream
    opacity_lag_40km       — cloud opacity lag corresponding to 40 km upstream
    site_cloud_opacity     — current cloud opacity at site [0–100]
    cloud_opacity_trend    — 30-min change in cloud opacity

Usage
─────
    from src.cmv.wind_cmv import build_wind_cmv_features
    features = build_wind_cmv_features(solcast_df, wind_hourly_df)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pvlib

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Site constants ────────────────────────────────────────────────────────────
_LAT  =  6.7912
_LON  = 79.9005
_ELEV = 20.0

_LOCATION = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV)

# Tropical marine 850 hPa / 10 m wind speed ratio
_SCALE_850 = 1.8

_CLOUD_ALT_M = 1500.0     # representative cumulus base over Sri Lanka

_DISTANCES_KM = [5, 10, 20, 40]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shadow_offset_km(zenith_deg: float, azimuth_deg: float) -> tuple[float, float]:
    """East and north shadow displacement (km) for a cloud at _CLOUD_ALT_M."""
    if zenith_deg >= 90.0:
        return 0.0, 0.0
    tan_z   = math.tan(math.radians(zenith_deg))
    offset  = (_CLOUD_ALT_M / 1000.0) * tan_z
    az_r    = math.radians(azimuth_deg)
    return -offset * math.sin(az_r), -offset * math.cos(az_r)


def _shadow_arrival_min(
    d_km: float,
    speed_kmh: float,
    dir_deg: float,
    sh_e: float,
    sh_n: float,
) -> float:
    """Minutes until shadow from distance d_km arrives at site."""
    if speed_kmh < 0.5:
        return np.nan
    dir_r  = math.radians(dir_deg)
    ux, uy = math.sin(dir_r), math.cos(dir_r)   # unit vector toward
    shadow_proj = sh_e * ux + sh_n * uy
    d_eff  = d_km - shadow_proj
    if d_eff <= 0:
        return 0.0
    return (d_eff / speed_kmh) * 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Upstream opacity from time-lagged cloud_opacity
# ─────────────────────────────────────────────────────────────────────────────

def _opacity_lag_features(
    opacity_5min: pd.Series,
    speed_kmh_5min: pd.Series,
) -> pd.DataFrame:
    """
    For each 5-min timestamp, look up cloud_opacity at
    t - (distance / speed) minutes — this is the opacity of the cloud that
    will arrive in (distance / speed) minutes.

    Uses a variable lag based on instantaneous wind speed.
    Falls back to fixed lags when speed is zero.

    Returns DataFrame with columns opacity_lag_{5,10,20,40}km.
    """
    out = pd.DataFrame(index=opacity_5min.index)
    dt_min = 5.0   # 5-min resolution

    for d_km in _DISTANCES_KM:
        # Lag in 5-min steps = (d_km / speed_kmh) * 60 / 5
        # Vectorised: compute lag index per row
        with np.errstate(divide="ignore", invalid="ignore"):
            lag_min  = np.where(
                speed_kmh_5min.values > 0.5,
                d_km / speed_kmh_5min.values * 60.0,
                np.nan,
            )
        lag_steps_f = np.round(lag_min / dt_min)
        nan_mask  = np.isnan(lag_steps_f)
        lag_steps_f_safe = np.where(nan_mask, 0, lag_steps_f)
        lag_steps = np.where(nan_mask, -1, lag_steps_f_safe.astype(int))

        vals = np.full(len(opacity_5min), np.nan)
        arr  = opacity_5min.values.astype(float)

        for i, lag in enumerate(lag_steps):
            if lag < 0:   # sentinel for NaN
                continue
            j = i - lag
            if j >= 0:
                vals[i] = arr[j]

        out[f"opacity_lag_{d_km}km"] = vals

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openmeteo_wind(
    start_date: str,
    end_date: str,
    lat: float = _LAT,
    lon: float = _LON,
) -> pd.DataFrame:
    """
    Fetch hourly 10 m wind speed and direction from Open-Meteo ERA5 archive.

    Parameters
    ----------
    start_date, end_date : "YYYY-MM-DD" strings.

    Returns
    -------
    pd.DataFrame with UTC DatetimeIndex and columns:
        wind_speed_10m_kmh   (km/h)
        wind_direction_10m   (° from N, CW — direction wind comes FROM)
    """
    import urllib.request
    import json

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=kmh&timezone=UTC"
    )
    logger.info(f"  Fetching Open-Meteo wind: {start_date} → {end_date} …")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)

    h  = data["hourly"]
    df = pd.DataFrame({
        "wind_speed_10m_kmh" : h["wind_speed_10m"],
        "wind_direction_10m" : h["wind_direction_10m"],
    }, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "timestamp_utc"

    logger.info(f"  Wind data: {len(df):,} hours  "
                f"(non-null WS: {df['wind_speed_10m_kmh'].notna().sum():,}  "
                f"WD: {df['wind_direction_10m'].notna().sum():,})")
    return df


def build_wind_cmv_features(
    solcast_5min: pd.DataFrame,
    wind_hourly:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Build CMV-equivalent feature matrix from surface wind + Solcast cloud_opacity.

    Parameters
    ----------
    solcast_5min : 5-min DataFrame with at minimum:
        - cloud_opacity   (0–100)
        - WS10M_cal       (m/s, used as fallback if wind_hourly is empty)
        UTC DatetimeIndex.
    wind_hourly : hourly DataFrame from fetch_openmeteo_wind() with:
        - wind_speed_10m_kmh
        - wind_direction_10m   (direction wind comes FROM)
        UTC DatetimeIndex.

    Returns
    -------
    pd.DataFrame (5-min, UTC index) with all CMV features.
    """
    logger.info("Building wind-based CMV features …")

    # ── 1. Upsample wind to 5-min (forward-fill within each hour) ─────────────
    idx_5min = solcast_5min.index
    wind_5min = (
        wind_hourly
        .reindex(idx_5min, method="ffill", tolerance="59min")
    )

    # Scale 10 m → 850 hPa (~1500 m cloud level)
    cloud_speed = wind_5min["wind_speed_10m_kmh"] * _SCALE_850   # km/h

    # Wind direction: "FROM" direction → "TO" direction (clouds move toward)
    cloud_direction = (wind_5min["wind_direction_10m"] + 180.0) % 360.0

    # ── 2. Solar geometry ─────────────────────────────────────────────────────
    logger.info("  Computing solar positions …")
    sol = _LOCATION.get_solarposition(idx_5min)
    zenith   = sol["apparent_zenith"].values
    azimuth  = sol["azimuth"].values
    elevation = sol["elevation"].values
    is_day   = elevation > 0.0

    # ── 3. Shadow offset ──────────────────────────────────────────────────────
    sh_e_arr  = np.zeros(len(idx_5min))
    sh_n_arr  = np.zeros(len(idx_5min))
    shadow_km = np.zeros(len(idx_5min))

    for i in range(len(idx_5min)):
        if is_day[i]:
            e, n = _shadow_offset_km(float(zenith[i]), float(azimuth[i]))
            sh_e_arr[i]  = e
            sh_n_arr[i]  = n
            shadow_km[i] = math.sqrt(e**2 + n**2)

    # ── 4. Shadow arrival times ───────────────────────────────────────────────
    logger.info("  Computing shadow arrival times …")
    arrivals = {d: np.full(len(idx_5min), np.nan) for d in _DISTANCES_KM}
    spd = cloud_speed.values
    dr  = cloud_direction.values

    for i in range(len(idx_5min)):
        if not is_day[i]:
            continue
        for d_km in _DISTANCES_KM:
            arrivals[d_km][i] = _shadow_arrival_min(
                d_km, float(spd[i]), float(dr[i]),
                sh_e_arr[i], sh_n_arr[i],
            )

    # ── 5. Upstream opacity lags ──────────────────────────────────────────────
    logger.info("  Computing upstream opacity lags …")
    opacity = solcast_5min["cloud_opacity"].copy()
    opacity_lags = _opacity_lag_features(opacity, cloud_speed)

    # ── 6. Opacity trend ──────────────────────────────────────────────────────
    # 30-min change (6 × 5-min steps)
    opacity_trend = opacity.diff(6)

    # ── 7. Assemble ───────────────────────────────────────────────────────────
    feat = pd.DataFrame({
        "cloud_speed_kmh"      : cloud_speed.values,
        "cloud_direction_deg"  : cloud_direction.values,
        "shadow_offset_km"     : shadow_km,
        "solar_zenith_deg"     : zenith,
        "solar_azimuth_deg"    : azimuth,
        "shadow_arrival_5km"   : arrivals[5],
        "shadow_arrival_10km"  : arrivals[10],
        "shadow_arrival_20km"  : arrivals[20],
        "shadow_arrival_40km"  : arrivals[40],
        "opacity_lag_5km"      : opacity_lags["opacity_lag_5km"].values,
        "opacity_lag_10km"     : opacity_lags["opacity_lag_10km"].values,
        "opacity_lag_20km"     : opacity_lags["opacity_lag_20km"].values,
        "opacity_lag_40km"     : opacity_lags["opacity_lag_40km"].values,
        "site_cloud_opacity"   : opacity.values,
        "cloud_opacity_trend"  : opacity_trend.values,
    }, index=idx_5min)

    # Nighttime: zero out arrivals / offsets (already NaN from above)
    night = ~is_day
    for col in ["shadow_offset_km", "shadow_arrival_5km",
                "shadow_arrival_10km", "shadow_arrival_20km", "shadow_arrival_40km"]:
        feat.loc[feat.index[night], col] = np.nan

    n_day   = int(is_day.sum())
    n_valid = int(feat["shadow_arrival_5km"].notna().sum())
    logger.info(
        f"  Done. {len(feat):,} rows  "
        f"({n_day:,} daytime  |  {n_valid:,} with shadow arrival)"
    )
    return feat
