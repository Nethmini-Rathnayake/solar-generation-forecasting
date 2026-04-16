"""
src/cmv/shadow_predictor.py
──────────────────────────────────────────────────────────────────────────────
Shadow arrival time prediction from Cloud Motion Vectors.

The key physics insight is that the cloud shadow is displaced from the cloud
itself due to the solar geometry:

    shadow_offset = cloud_altitude × tan(solar_zenith_angle)

This offset changes continuously as the sun moves, and must be accounted for
when predicting when a cloud's shadow will cross the PV array.

The effective "shadow arrival time" is:

    t_arrival = (d_effective) / cloud_speed

where d_effective is the distance the shadow must travel, not the cloud.

Output feature vector (per 10-min interval)
───────────────────────────────────────────
    cloud_speed_kmh       — magnitude of CMV [km/h]
    cloud_direction_deg   — direction cloud moves toward [°N clockwise]
    shadow_offset_km      — solar-geometry shadow displacement from cloud [km]
    shadow_arrival_5km    — minutes until cloud shadow at 5 km arrives
    shadow_arrival_10km   — minutes until cloud shadow at 10 km arrives
    shadow_arrival_20km   — minutes until cloud shadow at 20 km arrives
    shadow_arrival_40km   — minutes until cloud shadow at 40 km arrives
    upstream_ref_5km      — cloud reflectance 5 km upstream (opacity proxy)
    upstream_ref_10km     — cloud reflectance 10 km upstream
    upstream_ref_20km     — cloud reflectance 20 km upstream
    upstream_ref_40km     — cloud reflectance 40 km upstream
    cmv_confidence        — flow field spatial coherence [0–1]
    site_reflectance      — cloud reflectance at site pixel

Usage
─────
    from src.cmv.shadow_predictor import compute_shadow_features
    features = compute_shadow_features(cmv, upstream, timestamp)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pvlib

if TYPE_CHECKING:
    from src.cmv.optical_flow import CMV

# ── Site constants ────────────────────────────────────────────────────────────
_LAT  =  6.7912
_LON  = 79.9005
_ELEV = 20.0    # m

# Typical cumulus cloud base altitude over Sri Lanka (m).
# Cumulus: 600–2000 m.  Stratus/stratocumulus: 200–600 m.
# We use 1500 m as a representative daytime convective cloud base.
_CLOUD_ALT_M_DEFAULT = 1500.0

_LOCATION = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV)

_UPSTREAM_DISTANCES_KM = [5, 10, 20, 40]


def solar_geometry(timestamp: pd.Timestamp) -> dict:
    """
    Compute solar position at the site for the given UTC timestamp.

    Returns
    -------
    dict with keys:
        zenith_deg    : solar zenith angle (°)
        azimuth_deg   : solar azimuth (° from N, CW)
        elevation_deg : solar elevation (°)
        is_daytime    : bool
    """
    idx = pd.DatetimeIndex([timestamp], tz="UTC")
    sol = _LOCATION.get_solarposition(idx)

    return {
        "zenith_deg"    : float(sol["apparent_zenith"].iloc[0]),
        "azimuth_deg"   : float(sol["azimuth"].iloc[0]),
        "elevation_deg" : float(sol["elevation"].iloc[0]),
        "is_daytime"    : float(sol["elevation"].iloc[0]) > 0.0,
    }


def shadow_offset(
    zenith_deg:     float,
    azimuth_deg:    float,
    cloud_alt_m:    float = _CLOUD_ALT_M_DEFAULT,
) -> tuple[float, float]:
    """
    Compute the ground displacement of the cloud shadow from the cloud.

    The shadow is displaced in the direction opposite to the sun:
        shadow_east  = -cloud_alt × tan(zenith) × sin(azimuth)
        shadow_north = -cloud_alt × tan(zenith) × cos(azimuth)

    Parameters
    ----------
    zenith_deg  : solar zenith angle (°).
    azimuth_deg : solar azimuth (° from N, CW, where sun is).
    cloud_alt_m : cloud base altitude (m).

    Returns
    -------
    offset_east_km, offset_north_km : shadow displacement (km).
        Negative = shadow is west/south of cloud.
    """
    if zenith_deg >= 90.0:
        return 0.0, 0.0

    tan_z = math.tan(math.radians(zenith_deg))
    offset_km = (cloud_alt_m / 1000.0) * tan_z   # horizontal distance (km)

    az_rad = math.radians(azimuth_deg)
    # Sun is at azimuth_deg → shadow is in opposite direction
    shadow_east  = -offset_km * math.sin(az_rad)
    shadow_north = -offset_km * math.cos(az_rad)

    return shadow_east, shadow_north


def shadow_arrival_time(
    distance_km:   float,
    cmv:           "CMV",
    shadow_east:   float,
    shadow_north:  float,
) -> float:
    """
    Compute minutes until the shadow of a cloud at given distance arrives.

    A cloud at distance d_km upstream will have its shadow at the site when:

        cloud_position + shadow_offset = site_position

    This shifts the effective target distance:
        d_effective = d_km + dot(shadow_offset, motion_unit_vector)

    Parameters
    ----------
    distance_km   : distance to the cloud (km, upstream of site).
    cmv           : cloud motion vector.
    shadow_east   : shadow east displacement (km, from shadow_offset()).
    shadow_north  : shadow north displacement (km, from shadow_offset()).

    Returns
    -------
    float  Minutes until shadow arrival.  NaN if cloud is moving away.
    """
    speed = cmv.speed_kmh
    if speed < 0.5:
        # Essentially stationary — shadow arrival is indeterminate
        return np.nan

    # Motion unit vector (direction cloud moves toward)
    dir_rad = math.radians(cmv.direction_deg)
    ux = math.sin(dir_rad)   # eastward component
    uy = math.cos(dir_rad)   # northward component

    # Shadow offset projected onto motion direction
    # The shadow of an upstream cloud arrives later (negative correction)
    # if the shadow is offset in the direction of motion.
    shadow_proj = shadow_east * ux + shadow_north * uy

    d_effective = distance_km - shadow_proj   # corrected distance

    if d_effective <= 0:
        return 0.0   # shadow already at or past the site

    return (d_effective / speed) * 60.0   # minutes


def compute_shadow_features(
    cmv:        "CMV",
    upstream:   dict[float, float],
    timestamp:  pd.Timestamp,
    cloud_alt_m: float = _CLOUD_ALT_M_DEFAULT,
) -> dict:
    """
    Build the full CMV feature dictionary for one 10-min interval.

    Parameters
    ----------
    cmv       : CMV result from optical_flow.extract_cmv().
    upstream  : dict {distance_km: reflectance} from optical_flow.upstream_state().
    timestamp : UTC timestamp of the observation.
    cloud_alt_m : assumed cloud base altitude (m).

    Returns
    -------
    dict  Feature dictionary (all floats, NaN for unavailable).
    """
    sol = solar_geometry(timestamp)

    # Shadow displacement
    if sol["is_daytime"]:
        sh_east, sh_north = shadow_offset(
            sol["zenith_deg"], sol["azimuth_deg"], cloud_alt_m
        )
        shadow_total_km = math.sqrt(sh_east**2 + sh_north**2)
    else:
        sh_east, sh_north, shadow_total_km = 0.0, 0.0, 0.0

    # Shadow arrival times for each upstream distance
    arrivals = {}
    for d_km in _UPSTREAM_DISTANCES_KM:
        if sol["is_daytime"] and cmv.speed_kmh > 0.5:
            t = shadow_arrival_time(d_km, cmv, sh_east, sh_north)
        else:
            t = np.nan
        arrivals[d_km] = t

    # Upstream reflectance
    refs = {d: upstream.get(d, np.nan) for d in _UPSTREAM_DISTANCES_KM}

    feat = {
        "timestamp"            : timestamp,
        "cloud_speed_kmh"      : round(cmv.speed_kmh, 2),
        "cloud_direction_deg"  : round(cmv.direction_deg, 1),
        "shadow_offset_km"     : round(shadow_total_km, 2),
        "solar_zenith_deg"     : round(sol["zenith_deg"], 2),
        "solar_azimuth_deg"    : round(sol["azimuth_deg"], 2),
        "shadow_arrival_5km"   : arrivals[5],
        "shadow_arrival_10km"  : arrivals[10],
        "shadow_arrival_20km"  : arrivals[20],
        "shadow_arrival_40km"  : arrivals[40],
        "upstream_ref_5km"     : refs[5],
        "upstream_ref_10km"    : refs[10],
        "upstream_ref_20km"    : refs[20],
        "upstream_ref_40km"    : refs[40],
        "cmv_confidence"       : round(cmv.confidence, 3),
        "site_reflectance"     : round(cmv.site_reflectance, 3),
    }
    return feat


def build_null_features(timestamp: pd.Timestamp) -> dict:
    """Return an all-NaN feature dict (used when CMV computation fails)."""
    feat = {
        "timestamp"           : timestamp,
        "cloud_speed_kmh"     : np.nan,
        "cloud_direction_deg" : np.nan,
        "shadow_offset_km"    : np.nan,
        "solar_zenith_deg"    : np.nan,
        "solar_azimuth_deg"   : np.nan,
    }
    for d in _UPSTREAM_DISTANCES_KM:
        feat[f"shadow_arrival_{d}km"] = np.nan
        feat[f"upstream_ref_{d}km"]   = np.nan
    feat["cmv_confidence"]  = np.nan
    feat["site_reflectance"] = np.nan
    return feat


def annotate_pv_with_cmv(
    pv_5min:  pd.DataFrame,
    cmv_10min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge CMV features (10-min) into the 5-min PV feature matrix.

    CMV values are forward-filled within each 10-min window so every 5-min
    PV row gets the CMV computed at the start of that window.

    Parameters
    ----------
    pv_5min   : 5-min PV/weather DataFrame (UTC index).
    cmv_10min : 10-min CMV feature DataFrame (UTC index, "timestamp" column
                or DatetimeIndex).

    Returns
    -------
    pd.DataFrame  pv_5min with CMV columns merged in.
    """
    if "timestamp" in cmv_10min.columns:
        cmv_10min = cmv_10min.set_index("timestamp")

    cmv_10min.index = pd.to_datetime(cmv_10min.index, utc=True)

    # Reindex to 5-min, forward-fill within 10-min windows
    cmv_5min = cmv_10min.reindex(pv_5min.index, method="ffill", tolerance="10min")

    cmv_cols = [c for c in cmv_5min.columns if c != "timestamp"]
    return pv_5min.join(cmv_5min[cmv_cols], how="left")
