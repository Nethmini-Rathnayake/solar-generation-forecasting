"""
src/calibration/multistage.py
──────────────────────────────────────────────────────────────────────────────
Multi-stage physical calibration for Solcast satellite → site irradiance.

Stages implemented
──────────────────
1. Polynomial Bias Correction  (2nd and 3rd order)
   Corrects systematic bias in Solcast air_temp, relative_humidity, and wind
   speed by fitting against the 1-year ground-truth weather API data.

2. Clear-Sky Index (kt) GHI Normalisation
   kt = GHI / GHI_clearsky captures the cloud-attenuation fraction.
   Because no on-site pyranometer exists, kt_actual is derived by inverting
   the calibrated physics PV model against measured AC power:

       P_actual ≈ (kt_solcast * F) * GHI_clrsky * η_total
       ⟹  F = P_actual / P_physics_from_solcast_kt

   A monotonic polynomial is then fitted: kt_cal = f(kt_solcast, F)
   and applied to the full 6-year Solcast record.

3. Sky-Regime Residual Statistics
   Computes mean / std / AR(1) φ of (actual - physics) residuals per sky
   regime during the calibration window, ready for noise injection in the
   main pipeline.

Data availability note
──────────────────────
  - No on-site pyranometer → GHI ground-truth inferred via inverse PV model.
  - Ground-truth weather: tempC, humidity, windspeedKmph from local API.
  - Calibration window: April 2022 – March 2023 (1 year overlap).

Usage
─────
    from src.calibration.multistage import (
        fit_polynomial_corrections,
        apply_polynomial_corrections,
        fit_kt_normalisation,
        apply_kt_normalisation,
        compute_regime_noise_params,
    )
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pvlib

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Site constants ────────────────────────────────────────────────────────────
_LAT     =  6.7912
_LON     = 79.9005
_ELEV_M  = 20
_TILT    = 10
_AZIMUTH = 180
_GAMMA   = -0.0037   # °C⁻¹ temperature coefficient
_ETA_INV = 0.96      # inverter efficiency

# ── Calibration variable mapping ──────────────────────────────────────────────
# (solcast_column, local_column, unit_scale_to_solcast_units)
_VAR_MAP = [
    ("air_temp",          "tempC",          1.0),      # °C → °C
    ("relative_humidity", "humidity",        1.0),      # % → %
    ("wind_speed_10m",    "windspeedKmph",   0.27778),  # km/h → m/s
]

_MIN_CAL_POINTS = 80   # minimum daytime pairs to fit a polynomial


# ─────────────────────────────────────────────────────────────────────────────
# 1. Polynomial Bias Correction
# ─────────────────────────────────────────────────────────────────────────────

def _wind_col(solcast: pd.DataFrame) -> str:
    """Find the wind-speed column in a Solcast DataFrame."""
    for c in ("wind_speed_10m", "WS10M_cal", "wind_speed", "windspeed"):
        if c in solcast.columns:
            return c
    raise KeyError("No wind-speed column found in Solcast data")


def fit_polynomial_corrections(
    solcast_overlap: pd.DataFrame,
    local_overlap:   pd.DataFrame,
    poly_order:      int = 3,
) -> dict[str, np.ndarray]:
    """
    Fit polynomial bias corrections between Solcast and local weather.

    For each variable pair (X_sat, X_local) the function fits:

        X_local = p[0] + p[1]*X_sat + p[2]*X_sat² [+ p[3]*X_sat³]

    using least-squares regression on daytime intervals only.

    Parameters
    ----------
    solcast_overlap : pd.DataFrame
        Solcast 5-min data aligned to the calibration window.
    local_overlap   : pd.DataFrame
        Local 5-min data (same index) with columns tempC, humidity,
        windspeedKmph.
    poly_order : int
        Polynomial degree (2 or 3).  Default 3.

    Returns
    -------
    dict[str, np.ndarray]
        {solcast_col: poly_coefficients}  (ascending-power order)
    """
    if poly_order not in (2, 3):
        raise ValueError("poly_order must be 2 or 3")

    # Daytime mask: solar elevation > 5°
    loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV_M)
    sol = loc.get_solarposition(solcast_overlap.index)
    day = sol["elevation"] > 5

    corrections: dict[str, np.ndarray] = {}

    for sat_col, loc_col, scale in _VAR_MAP:
        # Resolve actual column name in solcast dataframe
        if sat_col not in solcast_overlap.columns:
            if sat_col == "wind_speed_10m":
                try:
                    sat_col = _wind_col(solcast_overlap)
                except KeyError:
                    logger.warning(f"  Skipping wind: no matching Solcast column")
                    continue
            else:
                logger.warning(f"  Skipping {sat_col}: not in Solcast columns")
                continue

        if loc_col not in local_overlap.columns:
            logger.warning(f"  Skipping {loc_col}: not in local columns")
            continue

        x = solcast_overlap.loc[day, sat_col].values.astype(float)
        y = local_overlap.loc[day, loc_col].values.astype(float) * scale

        # Drop NaN / inf
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < _MIN_CAL_POINTS:
            logger.warning(
                f"  [{sat_col}] only {len(x)} valid pairs — skipping polynomial fit"
            )
            continue

        # Fit with numpy polyfit (coefficients: highest power first)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(x, y, poly_order)   # highest power first

        p = np.poly1d(coeffs)
        y_pred = p(x)
        r2  = float(1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
        rmse = float(np.sqrt(np.mean((y - y_pred)**2)))

        logger.info(
            f"  [{sat_col}→{loc_col}]  order-{poly_order} poly  "
            f"R²={r2:.4f}  RMSE={rmse:.3f}  n={len(x):,}"
        )

        # Store ascending-power order for easy application
        corrections[sat_col] = coeffs[::-1]   # [const, linear, quad, ...]

    logger.info(f"  Polynomial corrections fitted for {len(corrections)} variable(s)")
    return corrections


def apply_polynomial_corrections(
    solcast_full: pd.DataFrame,
    corrections:  dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Apply fitted polynomial corrections to the full 6-year Solcast record.

    Parameters
    ----------
    solcast_full : pd.DataFrame
        Full multi-year Solcast data.
    corrections  : dict
        Output of ``fit_polynomial_corrections``.

    Returns
    -------
    pd.DataFrame  (copy with corrected columns suffixed _pcal)
    """
    out = solcast_full.copy()

    for sat_col, coeffs in corrections.items():
        if sat_col not in out.columns:
            logger.warning(f"  {sat_col} not in full Solcast — skipping apply")
            continue

        x    = out[sat_col].values.astype(float)
        corr = sum(coeffs[k] * x**k for k in range(len(coeffs)))
        new_col = sat_col + "_pcal"
        out[new_col] = corr.astype(np.float32)

        delta = float(np.nanmean(corr - x))
        logger.info(
            f"  [{sat_col}→{new_col}]  mean correction = {delta:+.3f} units"
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clear-Sky Index (kt) GHI Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _physics_pv_sim(solcast: pd.DataFrame, pdc0_w: float) -> pd.Series:
    """
    Compute physics-based AC power (W) from Solcast GHI using pvlib.

    Uses Erbs decomposition → Perez POA → Faiman cell temperature → PVWatts.
    """
    loc      = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV_M)
    sol_pos  = loc.get_solarposition(solcast.index)
    ghi      = solcast["ghi"].clip(lower=0)
    erbs     = pvlib.irradiance.erbs(ghi, sol_pos["apparent_zenith"], solcast.index)
    dni_e    = pvlib.irradiance.get_extra_radiation(solcast.index)
    airmass  = loc.get_airmass(solar_position=sol_pos)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt    = _TILT,
        surface_azimuth = _AZIMUTH,
        solar_zenith    = sol_pos["apparent_zenith"],
        solar_azimuth   = sol_pos["azimuth"],
        dni             = erbs["dni"].clip(lower=0),
        ghi             = ghi,
        dhi             = erbs["dhi"].clip(lower=0),
        model           = "perez",
        dni_extra       = dni_e,
        airmass         = airmass["airmass_relative"],
    )["poa_global"].fillna(0).clip(lower=0)

    wind  = solcast.get("wind_speed_10m", pd.Series(1.5, index=solcast.index)).clip(lower=0)
    t_air = solcast.get("air_temp", pd.Series(28.0, index=solcast.index))
    t_cell = pvlib.temperature.faiman(poa, t_air, wind)

    p_dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=pdc0_w, gamma_pdc=_GAMMA)
    return (p_dc * _ETA_INV).clip(lower=0)


def fit_kt_normalisation(
    solcast_overlap: pd.DataFrame,
    local_overlap:   pd.DataFrame,
    pv_power_col:    str,
    pdc0_w:          float = 259_000.0,
    poly_order:      int   = 2,
) -> dict:
    """
    Fit a kt-normalisation correction using measured PV power as irradiance proxy.

    Method
    ------
    1. Compute kt_solcast = GHI_solcast / clearsky_GHI_solcast  (0–1.2)
    2. Simulate P_physics from Solcast GHI using pvlib PVWatts
    3. Compute performance ratio:  PR = P_actual / P_physics  (clipped 0.1–2.5)
    4. kt_target ≈ kt_solcast × PR   (corrected clearness index)
    5. Fit polynomial:  kt_target = f(kt_solcast)
    6. Bin-statistics: mean kt_correction per kt_solcast decile

    Parameters
    ----------
    solcast_overlap : pd.DataFrame   Overlap-period Solcast data.
    local_overlap   : pd.DataFrame   Overlap-period local data.
    pv_power_col    : str            Column name for measured PV power (W).
    pdc0_w          : float          PV system nameplate DC capacity (W).
    poly_order      : int            Polynomial degree for kt curve (2 or 3).

    Returns
    -------
    dict with keys:
        "coeffs"     : np.ndarray   polynomial coefficients (ascending power)
        "bins"       : pd.Series    mean correction per kt_solcast decile
        "r2"         : float        goodness of fit on calibration data
        "pdc0_w"     : float        capacity used
    """
    # Resolve GHI and clearsky column names
    ghi_col = next((c for c in ("ghi", "ALLSKY_SFC_SW_DWN_cal") if c in solcast_overlap.columns), None)
    cks_col = next((c for c in ("clearsky_ghi", "CLRSKY_SFC_SW_DWN_cal") if c in solcast_overlap.columns), None)
    if ghi_col is None or cks_col is None:
        raise KeyError("Cannot find GHI / clearsky_GHI columns in Solcast data")

    ghi_s   = solcast_overlap[ghi_col].clip(lower=0)
    clrsky  = solcast_overlap[cks_col].replace(0, np.nan).clip(lower=1)
    kt_s    = (ghi_s / clrsky).clip(0, 1.2).fillna(0)

    # Physics simulation
    logger.info("  Computing physics PV simulation for kt calibration …")
    p_phys  = _physics_pv_sim(solcast_overlap, pdc0_w)

    # Measured PV (W)
    p_act   = local_overlap[pv_power_col].clip(lower=0)

    # Performance ratio (proxy for kt correction)
    pr = (p_act / p_phys.replace(0, np.nan)).clip(0.05, 3.0).fillna(1.0)
    kt_target = (kt_s * pr).clip(0, 1.3)

    # Daytime-only (GHI > 30, kt_solcast > 0.02)
    day = (ghi_s > 30) & (kt_s > 0.02)
    x   = kt_s[day].values
    y   = kt_target[day].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs_hp = np.polyfit(x, y, poly_order)  # highest power first

    p = np.poly1d(coeffs_hp)
    y_pred = p(x)
    r2  = float(1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

    # Bin statistics (10 decile bins)
    bins_x = pd.cut(x, bins=10)
    bins_df = pd.DataFrame({"kt_s": x, "kt_t": y, "bin": bins_x})
    bin_means = bins_df.groupby("bin", observed=True)["kt_t"].mean()

    logger.info(
        f"  kt normalisation  order-{poly_order} poly  R²={r2:.4f}  n={len(x):,}"
    )
    logger.info(f"  Coefficients (ascending power): {coeffs_hp[::-1]}")

    return {
        "coeffs" : coeffs_hp[::-1],   # ascending power [const, x, x², ...]
        "bins"   : bin_means,
        "r2"     : r2,
        "pdc0_w" : pdc0_w,
    }


def apply_kt_normalisation(
    solcast_full: pd.DataFrame,
    kt_params:    dict,
) -> pd.DataFrame:
    """
    Apply kt-normalisation to GHI in the full 6-year Solcast record.

    Computes:
        kt_solcast  = GHI / GHI_clearsky
        kt_cal      = polynomial(kt_solcast)   [clipped 0–1.3]
        GHI_cal     = kt_cal × GHI_clearsky

    Parameters
    ----------
    solcast_full : pd.DataFrame   Full 6-year Solcast dataset.
    kt_params    : dict           Output of ``fit_kt_normalisation``.

    Returns
    -------
    pd.DataFrame  with new columns: kt_solcast, kt_calibrated, ghi_kt_cal
    """
    out = solcast_full.copy()

    ghi_col = next((c for c in ("ghi", "ALLSKY_SFC_SW_DWN_cal") if c in out.columns), None)
    cks_col = next((c for c in ("clearsky_ghi", "CLRSKY_SFC_SW_DWN_cal") if c in out.columns), None)

    ghi    = out[ghi_col].clip(lower=0)
    clrsky = out[cks_col].replace(0, np.nan).clip(lower=1)
    kt_s   = (ghi / clrsky).clip(0, 1.2).fillna(0)

    coeffs = kt_params["coeffs"]
    kt_cal = sum(coeffs[k] * kt_s**k for k in range(len(coeffs))).clip(0, 1.3)

    out["kt_solcast"]    = kt_s.astype(np.float32)
    out["kt_calibrated"] = kt_cal.astype(np.float32)
    out["ghi_kt_cal"]    = (kt_cal * clrsky.fillna(0)).clip(lower=0).astype(np.float32)

    ghi_delta = float((out["ghi_kt_cal"] - ghi).where(ghi > 10).mean())
    logger.info(
        f"  kt GHI calibration applied  "
        f"mean GHI correction = {ghi_delta:+.1f} W/m²  "
        f"(daytime intervals only)"
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sky-Regime Noise Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Regime codes used throughout the pipeline
REGIMES = {
    0: "Clear",
    1: "PartlyCloudy",
    2: "Overcast",
    3: "HighlyVolatile",
}


def classify_regime(
    kt: pd.Series,
    cloud_opacity: pd.Series | None = None,
) -> pd.Series:
    """
    Classify each 5-min interval into one of four sky regimes.

    Regimes
    -------
    0  Clear          kt > 0.75  AND  cloud_opacity < 20 (if available)
    1  PartlyCloudy   0.35 < kt ≤ 0.75
    2  Overcast       kt ≤ 0.35
    3  HighlyVolatile rolling-30min std(kt) > 0.12  (overrides 1/2 regime)

    Returns
    -------
    pd.Series[int8]  regime code (0–3)
    """
    regime = pd.Series(1, index=kt.index, dtype=np.int8)  # default PartlyCloudy

    overcast_mask = kt <= 0.35
    clear_mask    = kt > 0.75
    if cloud_opacity is not None:
        clear_mask = clear_mask & (cloud_opacity < 20)

    regime[overcast_mask] = 2
    regime[clear_mask]    = 0

    # Highly Volatile: 30-min rolling std of kt > threshold
    kt_std = kt.rolling(6, min_periods=3).std().fillna(0)
    volatile_mask = (kt_std > 0.12) & (regime != 0)   # Clear sky is never "volatile"
    regime[volatile_mask] = 3

    counts = regime.value_counts().sort_index()
    for code, name in REGIMES.items():
        n = int(counts.get(code, 0))
        logger.info(f"    {name:>15}: {n:>8,}  ({100*n/len(regime):.1f}%)")

    return regime


def compute_regime_noise_params(
    regime:   pd.Series,
    residual: pd.Series,
) -> dict[int, dict]:
    """
    Compute AR(1) noise parameters for each sky regime.

    The residual time series ε(t) = P_actual(t) − P_physics(t) is modelled
    as an AR(1) process within each regime:

        ε(t) = φ × ε(t−1) + σ_ε × η(t),   η ~ N(0,1)

    where φ (persistence) and σ_ε (innovation std) are estimated from the
    calibration-window residuals.

    Parameters
    ----------
    regime   : pd.Series[int8]   Sky regime code per 5-min interval.
    residual : pd.Series[float]  Actual − physics residual (kW).

    Returns
    -------
    dict[regime_code, {"phi": float, "sigma": float, "mean": float}]
    """
    params: dict[int, dict] = {}

    for code, name in REGIMES.items():
        mask = regime == code
        eps  = residual[mask].dropna()

        if len(eps) < 30:
            params[code] = {"phi": 0.0, "sigma": 0.0, "mean": 0.0}
            logger.warning(f"  [{name}] insufficient residuals ({len(eps)}) — zero noise")
            continue

        # AR(1) estimate via autocorrelation at lag 1
        eps_c  = eps - eps.mean()
        lag1   = float(eps_c.autocorr(lag=1))
        phi    = max(-0.95, min(0.95, lag1))
        sigma  = float(eps_c.std() * np.sqrt(1 - phi**2))
        mean   = float(eps.mean())

        params[code] = {"phi": phi, "sigma": sigma, "mean": mean}
        logger.info(
            f"  [{name:>15}]  φ={phi:+.3f}  σ={sigma:.2f} kW  "
            f"μ={mean:+.2f} kW  n={len(eps):,}"
        )

    return params


def generate_ar1_noise(
    regime:      pd.Series,
    noise_params: dict[int, dict],
    seed:        int = 42,
) -> pd.Series:
    """
    Generate regime-aware AR(1) synthetic noise for the full 6-year period.

    Parameters
    ----------
    regime       : pd.Series[int8]   Sky regime for every 5-min interval.
    noise_params : dict              Output of ``compute_regime_noise_params``.
    seed         : int               Random seed for reproducibility.

    Returns
    -------
    pd.Series[float]  Synthetic noise (kW) per 5-min interval.
    """
    rng   = np.random.default_rng(seed)
    noise = np.zeros(len(regime), dtype=np.float32)
    prev  = 0.0

    for i, r in enumerate(regime.values):
        p = noise_params.get(int(r), {"phi": 0.0, "sigma": 0.0, "mean": 0.0})
        innov  = float(rng.normal(0, max(p["sigma"], 1e-9)))
        noise[i] = p["mean"] + p["phi"] * prev + innov
        prev = noise[i] - p["mean"]   # keep zero-mean AR component

    return pd.Series(noise, index=regime.index, name="noise_kW")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def eval_metrics(
    obs:   pd.Series,
    pred:  pd.Series,
    label: str = "",
    min_obs: float = 1.0,
) -> dict:
    """Return RMSE, MBE, R², nRMSE for daytime (obs > min_obs kW)."""
    mask = obs > min_obs
    o, p = obs[mask].values, pred[mask].values

    if len(o) < 10:
        return {"RMSE": np.nan, "MBE": np.nan, "R2": np.nan, "nRMSE": np.nan, "n": 0}

    rmse  = float(np.sqrt(np.mean((o - p) ** 2)))
    mbe   = float(np.mean(p - o))
    ss    = float(np.sum((o - o.mean()) ** 2))
    r2    = float(1 - np.sum((o - p) ** 2) / ss) if ss > 0 else np.nan
    nrmse = rmse / float(o.mean()) * 100

    logger.info(
        f"  {label:<30}  R²={r2:.4f}  RMSE={rmse:.2f} kW  "
        f"nRMSE={nrmse:.1f}%  MBE={mbe:+.2f} kW  n={mask.sum():,}"
    )
    return {"RMSE": rmse, "MBE": mbe, "R2": r2, "nRMSE": nrmse, "n": int(mask.sum())}
