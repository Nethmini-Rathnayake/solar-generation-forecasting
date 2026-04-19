"""
src/physics/pv_model.py
------------------------
Physics-based PV power simulation using pvlib PVWatts model.

The simulation is driven by calibrated NASA POWER irradiance and temperature
(output of src/calibration/). It produces an hourly AC power time series for
the full 6-year dataset (2020–2026) that becomes the target variable for the
24-hour-ahead forecasting model.

Physics model (PVWatts, NREL)
------------------------------
    G_eff   = plane-of-array (POA) irradiance  [W/m²]
              computed from GHI/DNI/DHI + solar position + panel tilt/azimuth

    T_cell  = cell temperature [°C]
              Faiman model: T_cell = T_air + G_eff × (u0 / (u0 + u1 × WS))

    P_dc    = pdc0 × (G_eff / 1000) × [1 + γ_pdc × (T_cell − 25)]
              PVWatts DC equation; γ_pdc = temperature coefficient (negative)

    P_ac    = PVWatts AC: P_dc × η_inv  (clipped to pac0)

Capacity calibration
---------------------
site.yaml lists capacity_kw = 10.0, which is incorrect.
The empirical system capacity is derived from the 1-year overlap window:

    pdc0_empirical = OLS slope × 1000
    where   P_observed = slope × P_sim_1kw + intercept
    and     P_sim_1kw  = pvwatts output with pdc0 = 1000 W (normalised)

This gives pdc0 ≈ 237 kW, matching the observed peak of ~259 kW at GHI peak.
The intercept (~7 kW) reflects a constant auxiliary load not captured by
the physics model; it is not added to the simulation output.

Output
------
    data/synthetic/pv_synthetic_6yr.csv
        Columns: pv_ac_W, pv_dc_W, poa_global, temp_cell, solar_elevation
        Index  : timestamp_utc (UTC)

Usage
-----
    from src.physics.pv_model import simulate_pv, calibrate_pdc0, save_synthetic

    cal   = load calibrated NASA df
    local = load local hourly df (overlap year)

    pdc0  = calibrate_pdc0(cal_overlap, local)
    df    = simulate_pv(cal_full, pdc0, cfg)
    save_synthetic(df, cfg)
"""

import pandas as pd
import numpy as np
from scipy import stats
import pvlib

from src.utils.config import resolve_path
from src.utils.logger import get_logger
from src.features.weather_patterns import classify_sky_condition

logger = get_logger(__name__)

# ── Site constants (also in site.yaml — kept here for module independence)
_LAT        =  6.7912
_LON        = 79.9005
_ELEV_M     = 20
_TILT       = 10        # panel tilt degrees
_AZIMUTH    = 180       # south-facing (180°)
_GAMMA_PDC  = -0.0037   # temperature coefficient per °C
_ETA_INV    = 0.96      # inverter nominal efficiency

# PV column used for calibration
_PV_POWER_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"

# Faiman thermal model defaults (standard)
_U0 = 25.0   # W/(m² K) — constant heat loss coefficient
_U1 =  6.84  # W/(m² K) / (m/s) — wind-driven heat loss


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_poa_and_temp(nasa_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute plane-of-array irradiance, cell temperature, and solar elevation.

    Parameters
    ----------
    nasa_df : pd.DataFrame
        Must contain calibrated columns:
        ALLSKY_SFC_SW_DWN_cal, ALLSKY_SFC_SW_DNI_cal, ALLSKY_SFC_SW_DIFF_cal,
        T2M_cal, WS10M_cal

    Returns
    -------
    poa_global : pd.Series   [W/m²]
    temp_cell  : pd.Series   [°C]
    solar_elev : pd.Series   [degrees]
    """
    location = pvlib.location.Location(
        latitude=_LAT, longitude=_LON,
        tz="UTC", altitude=_ELEV_M,
    )

    solar_pos = location.get_solarposition(nasa_df.index)

    ghi = nasa_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)

    # ── Erbs decomposition: derive self-consistent DHI / DNI from GHI ────────
    # NASA and ERA5 provide GHI, DNI, DHI as independent satellite retrievals.
    # At the hourly level they often violate  GHI = DNI·cos(z) + DHI, causing
    # systematic errors in the plane-of-array (POA) calculation.
    # The Erbs model derives DHI from kt = GHI/GHI_clearsky, then
    # DNI = (GHI − DHI) / cos(z), guaranteeing energy-balance consistency.
    # GHI is kept from the dataset (most reliable component in both NASA & ERA5).
    erbs_out = pvlib.irradiance.erbs(
        ghi=ghi,
        zenith=solar_pos["apparent_zenith"],
        datetime_or_doy=nasa_df.index,
    )
    dni = erbs_out["dni"].clip(lower=0)
    dhi = erbs_out["dhi"].clip(lower=0)

    # ── POA irradiance — adaptive sky model ──────────────────────────────────
    # Dry season (Jan–Mar, Jun–Sep): direct-beam dominant → isotropic sky
    # Wet season (Apr–May, Oct–Dec): diffuse-dominant (cloud cover) → Perez sky
    #
    # The Perez (1990) model explicitly accounts for circumsolar and horizon
    # brightening, giving 5–15% lower POA error under overcast tropical skies
    # compared to the isotropic model which assumes uniform sky radiance.

    poa_iso = pvlib.irradiance.get_total_irradiance(
        surface_tilt=_TILT,
        surface_azimuth=_AZIMUTH,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model="isotropic",
    )

    # Perez model requires extra-terrestrial DNI and relative airmass
    dni_extra = pvlib.irradiance.get_extra_radiation(nasa_df.index)
    airmass   = location.get_airmass(
        solar_position=solar_pos,
        model="kastenyoung1989",
    )
    poa_perez = pvlib.irradiance.get_total_irradiance(
        surface_tilt=_TILT,
        surface_azimuth=_AZIMUTH,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model="perez",
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
    )

    # Merge: wet months → Perez, dry months → isotropic
    poa_iso_g   = poa_iso["poa_global"].fillna(0.0).clip(lower=0.0)
    poa_perez_g = poa_perez["poa_global"].fillna(0.0).clip(lower=0.0)
    wet_mask    = nasa_df.index.month.isin(_WET_MONTHS)

    poa_global             = poa_iso_g.copy()
    poa_global[wet_mask]   = poa_perez_g[wet_mask]

    # ── Cell temperature (Faiman model) ───────────────────────────────────────
    wind_speed = (
        nasa_df["WS10M_cal"].clip(lower=0)
        if "WS10M_cal" in nasa_df.columns
        else pd.Series(1.5, index=nasa_df.index)   # default 1.5 m/s
    )
    temp_cell = pvlib.temperature.faiman(
        poa_global=poa_global,
        temp_air=nasa_df["T2M_cal"],
        wind_speed=wind_speed,
        u0=_U0,
        u1=_U1,
    )

    solar_elev = solar_pos["elevation"]
    return poa_global, temp_cell, solar_elev


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Season definitions (Sri Lanka climate)
# Dry:              Jan–Mar, Jun–Sep  (NE monsoon retreat + SW pre-monsoon dry)
# Wet/transition:   Apr–May, Oct–Dec  (inter-monsoon + NE monsoon onset)
_DRY_MONTHS: frozenset[int] = frozenset([1, 2, 3, 6, 7, 8, 9])
_WET_MONTHS: frozenset[int] = frozenset([4, 5, 10, 11, 12])


def _fit_polynomial(x: np.ndarray, y: np.ndarray, label: str) -> tuple[float, float]:
    """Internal: fit quadratic zero-intercept OLS and log diagnostics."""
    A = np.column_stack([x, x ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    pred  = a * x + b * x ** 2
    resid = pred - y
    rmse  = float(np.sqrt((resid ** 2).mean()))
    r2    = float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    mbe   = float(resid.mean())
    logger.info(
        f"  [{label}]  P_obs = {a:.2f}·sim + {b:.5f}·sim²"
        f"  R²={r2:.4f}  RMSE={rmse/1000:.1f} kW  MBE={mbe/1000:+.1f} kW  n={len(x):,}"
    )
    return a, b


def calibrate_seasonal(
    nasa_overlap: pd.DataFrame,
    local_hourly: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Fit separate quadratic calibrations for dry and wet/transition seasons.

    Sri Lanka has two distinct radiation regimes:
      Dry (Jan–Mar, Jun–Sep):   R² 0.88–0.93  — stable maritime air, clear skies
      Wet (Apr–May, Oct–Dec):   R² 0.67–0.82  — convective cloud, monsoon

    A single global polynomial is pulled toward the dominant dry-season
    pattern and systematically over-predicts during wet months.  Separate
    fits correct for the different cloud-cover statistics in each season.

    Parameters
    ----------
    nasa_overlap : pd.DataFrame   Calibrated NASA data (overlap window).
    local_hourly : pd.DataFrame   Must contain _PV_POWER_COL.

    Returns
    -------
    dict with keys "dry" and "wet", each mapping to (a, b) coefficients for:
        P_ac_W = a × sim_1kw + b × sim_1kw²
    """
    logger.info("Seasonal calibration: fitting dry and wet season polynomials …")

    poa, temp_cell, _ = _compute_poa_and_temp(nasa_overlap)

    sim_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa,
        temp_cell=temp_cell,
        pdc0=1000.0,
        gamma_pdc=_GAMMA_PDC,
    )

    obs = local_hourly[_PV_POWER_COL]
    ghi = nasa_overlap["ALLSKY_SFC_SW_DWN_cal"]
    df  = pd.concat([sim_1kw.rename("sim"), obs.rename("obs"), ghi.rename("ghi")], axis=1).dropna()
    df  = df[(df["ghi"] >= 50) & (df["obs"] > 100)]

    dry_mask = df.index.month.isin(_DRY_MONTHS)
    wet_mask = ~dry_mask

    logger.info(f"  Dry season months (1,2,3,6,7,8,9): {dry_mask.sum():,} daytime hours")
    logger.info(f"  Wet season months (4,5,10,11,12):  {wet_mask.sum():,} daytime hours")

    a_dry, b_dry = _fit_polynomial(df.loc[dry_mask, "sim"].values,
                                   df.loc[dry_mask, "obs"].values, "DRY")
    a_wet, b_wet = _fit_polynomial(df.loc[wet_mask, "sim"].values,
                                   df.loc[wet_mask, "obs"].values, "WET")

    return {"dry": (a_dry, b_dry), "wet": (a_wet, b_wet)}


_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
_MIN_DAYTIME_POINTS = 50   # minimum daytime samples to fit a monthly polynomial
_MIN_SKY_POINTS     = 30   # minimum samples for sky-stratified polynomial fit

_SKY_LABELS = {0: "Clear", 1: "PartlyCloudy", 2: "MostlyCloudy", 3: "Overcast"}


def calibrate_monthly(
    nasa_overlap: pd.DataFrame,
    local_hourly: pd.DataFrame,
    seasonal_coeffs: dict[str, tuple[float, float]] | None = None,
) -> dict[int, tuple[float, float]]:
    """
    Fit a separate quadratic zero-intercept polynomial for each calendar month.

    Addresses systematic per-month bias that 2-season calibration cannot fix
    (e.g. Apr over-predicts by +11 kW while Dec under-predicts by −12 kW).

    Parameters
    ----------
    nasa_overlap : pd.DataFrame
        Calibrated irradiance data clipped to the calibration window.
    local_hourly : pd.DataFrame
        Local PV measurements (must contain _PV_POWER_COL).
    seasonal_coeffs : dict, optional
        {"dry": (a,b), "wet": (a,b)} fallback for months with < 50 daytime
        points.  If None, falls back to the global polynomial.

    Returns
    -------
    dict[int, tuple[float, float]]
        {1: (a_jan, b_jan), 2: (a_feb, b_feb), ..., 12: (a_dec, b_dec)}
    """
    logger.info("Monthly calibration: fitting per-month quadratic polynomials …")

    poa, temp_cell, _ = _compute_poa_and_temp(nasa_overlap)
    sim_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa,
        temp_cell=temp_cell,
        pdc0=1000.0,
        gamma_pdc=_GAMMA_PDC,
    )

    obs = local_hourly[_PV_POWER_COL]
    ghi = nasa_overlap["ALLSKY_SFC_SW_DWN_cal"]
    df  = pd.concat([sim_1kw.rename("sim"), obs.rename("obs"), ghi.rename("ghi")], axis=1).dropna()
    df  = df[(df["ghi"] >= 50) & (df["obs"] > 100)]

    monthly_coeffs: dict[int, tuple[float, float]] = {}

    for m in range(1, 13):
        mname  = _MONTH_NAMES[m - 1]
        m_data = df[df.index.month == m]

        if len(m_data) < _MIN_DAYTIME_POINTS:
            # Fall back to seasonal coefficients
            if seasonal_coeffs is not None:
                key = "dry" if m in _DRY_MONTHS else "wet"
                monthly_coeffs[m] = seasonal_coeffs[key]
                logger.info(
                    f"  [{mname}]  only {len(m_data)} points — "
                    f"using {key.upper()} seasonal fallback"
                )
            else:
                logger.warning(
                    f"  [{mname}]  only {len(m_data)} points and no seasonal "
                    "fallback provided — month will use global polynomial"
                )
            continue

        a, b = _fit_polynomial(m_data["sim"].values, m_data["obs"].values, mname)
        monthly_coeffs[m] = (a, b)

    logger.info(f"  Monthly calibration complete: {len(monthly_coeffs)}/12 months fitted")
    return monthly_coeffs


def calibrate_sky_stratified(
    solcast_overlap: pd.DataFrame,
    local_5min: pd.DataFrame,
    monthly_coeffs: dict[int, tuple[float, float]] | None = None,
    seasonal_coeffs: dict[str, tuple[float, float]] | None = None,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Fit per-(month, sky_condition) quadratic zero-intercept polynomials.

    Sky conditions (from classify_sky_condition):
        0 = Clear
        1 = PartlyCloudy
        2 = MostlyCloudy
        3 = Overcast

    Fallback hierarchy (highest → lowest priority):
        sky_stratified[(month, sky)] → monthly[month] → seasonal[dry/wet] → global

    Parameters
    ----------
    solcast_overlap : pd.DataFrame
        Solcast 5-min data for the calibration window. Must contain
        ALLSKY_SFC_SW_DWN_cal, CLRSKY_SFC_SW_DWN_cal, cloud_opacity.
    local_5min : pd.DataFrame
        Aligned 5-min local PV measurements (must contain _PV_POWER_COL).
    monthly_coeffs : dict, optional
        Per-month (a, b) coefficients — used as fallback when n < _MIN_SKY_POINTS.
    seasonal_coeffs : dict, optional
        {"dry": (a,b), "wet": (a,b)} — second-level fallback.

    Returns
    -------
    dict[tuple[int, int], tuple[float, float]]
        {(month, sky_condition): (a, b), …}
    """
    logger.info("Sky-stratified calibration: fitting per-(month, sky_condition) polynomials …")

    poa, temp_cell, _ = _compute_poa_and_temp(solcast_overlap)
    sim_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa,
        temp_cell=temp_cell,
        pdc0=1000.0,
        gamma_pdc=_GAMMA_PDC,
    )

    obs = local_5min[_PV_POWER_COL]
    ghi = solcast_overlap["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)

    # Compute clearness index
    if "CLRSKY_SFC_SW_DWN_cal" in solcast_overlap.columns:
        clrsky = solcast_overlap["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan).clip(lower=1)
    else:
        clrsky = solcast_overlap["clearsky_ghi"].replace(0, np.nan).clip(lower=1)
    kt = (ghi / clrsky).clip(0, 1.2).fillna(0)

    # Classify sky condition
    if "cloud_opacity" in solcast_overlap.columns:
        cloud_op = solcast_overlap["cloud_opacity"]
    else:
        cloud_op = ((1 - kt) * 100).clip(0, 100)
    sky_cond = classify_sky_condition(cloud_op, kt)

    df = pd.concat(
        [sim_1kw.rename("sim"), obs.rename("obs"), ghi.rename("ghi"),
         sky_cond.rename("sky")],
        axis=1,
    ).dropna()
    df = df[(df["ghi"] >= 50) & (df["obs"] > 100)]

    sky_stratified: dict[tuple[int, int], tuple[float, float]] = {}
    total_cells = 0
    fitted_cells = 0

    for m in range(1, 13):
        mname = _MONTH_NAMES[m - 1]
        m_data = df[df.index.month == m]

        for s in range(4):
            sky_label = _SKY_LABELS[s]
            cell_data = m_data[m_data["sky"] == s]
            n = len(cell_data)
            total_cells += 1

            if n >= _MIN_SKY_POINTS:
                a, b = _fit_polynomial(
                    cell_data["sim"].values,
                    cell_data["obs"].values,
                    f"{mname}/{sky_label}",
                )
                sky_stratified[(m, s)] = (a, b)
                fitted_cells += 1
            else:
                # Fallback: monthly → seasonal → global
                if monthly_coeffs and m in monthly_coeffs:
                    sky_stratified[(m, s)] = monthly_coeffs[m]
                    logger.info(
                        f"  [{mname}/{sky_label}]  n={n} < {_MIN_SKY_POINTS} "
                        f"— using monthly fallback"
                    )
                elif seasonal_coeffs:
                    key = "dry" if m in _DRY_MONTHS else "wet"
                    sky_stratified[(m, s)] = seasonal_coeffs[key]
                    logger.info(
                        f"  [{mname}/{sky_label}]  n={n} < {_MIN_SKY_POINTS} "
                        f"— using {key.upper()} seasonal fallback"
                    )
                else:
                    logger.warning(
                        f"  [{mname}/{sky_label}]  n={n} < {_MIN_SKY_POINTS} "
                        f"and no fallback — cell will use global polynomial at simulate time"
                    )

    logger.info(
        f"  Sky-stratified calibration complete: "
        f"{fitted_cells}/{total_cells} cells fitted directly "
        f"({total_cells - fitted_cells} used fallback)"
    )
    return sky_stratified


def calibrate_polynomial(
    nasa_overlap: pd.DataFrame,
    local_hourly: pd.DataFrame,
) -> tuple[float, float]:
    """
    Fit a quadratic zero-intercept calibration from the 1-year overlap.

    Calibration equation
    --------------------
        P_ac_obs = a × P_sim_1kw + b × P_sim_1kw²

    where P_sim_1kw = pvwatts_dc(pdc0=1000 W) — normalised physics output.

    Why quadratic zero-intercept?
    ------------------------------
    • Zero-intercept: physically correct — zero irradiance → zero power.
      The linear OLS intercept of ~+10 kW produces artificial output at
      dawn/dusk and corrupts the validation plot.
    • Quadratic term: captures slight saturation at high irradiance
      (inverter clipping, cell heating, reflection losses at oblique angles).
    • R² improvement: 0.702 vs 0.690 for linear OLS.
    • The R² ceiling of ~0.70 is set by NASA satellite cloud-cover noise
      (hourly obs/sim ratio std ≈ 121 W per kW); no regression form can
      exceed this without a ground-based pyranometer.

    Parameters
    ----------
    nasa_overlap : pd.DataFrame
        Calibrated NASA data clipped to the overlap window.
    local_hourly : pd.DataFrame
        Aligned local measurements (must contain _PV_POWER_COL).

    Returns
    -------
    tuple[float, float]
        (a, b) — polynomial coefficients for:
        P_ac_W = a × sim_1kw + b × sim_1kw²
    """
    logger.info("Calibrating system (global quadratic zero-intercept) from 1-year overlap …")

    poa, temp_cell, _ = _compute_poa_and_temp(nasa_overlap)
    sim_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa, temp_cell=temp_cell,
        pdc0=1000.0, gamma_pdc=_GAMMA_PDC,
    )

    obs = local_hourly[_PV_POWER_COL]
    ghi = nasa_overlap["ALLSKY_SFC_SW_DWN_cal"]
    df  = pd.concat([sim_1kw.rename("sim"), obs.rename("obs"), ghi.rename("ghi")], axis=1).dropna()
    df  = df[(df["ghi"] >= 50) & (df["obs"] > 100)]

    # Also log linear OLS for comparison
    sl_lin, ic_lin, r_lin, _, _ = stats.linregress(df["sim"].values, df["obs"].values)
    pred_lin = sl_lin * df["sim"].values + ic_lin
    logger.info("  ── Linear OLS (baseline) ─────────────────────────────────")
    logger.info(
        f"  P_obs = {sl_lin:.2f}·sim + {ic_lin:.0f} W"
        f"  R²={r_lin**2:.4f}  RMSE={np.sqrt(((pred_lin-df['obs'].values)**2).mean())/1000:.1f} kW"
    )
    logger.info("  ── Quadratic zero-intercept (adopted) ────────────────────")
    a, b = _fit_polynomial(df["sim"].values, df["obs"].values, "GLOBAL")
    logger.info("  R² ceiling ≈ 0.73 (NASA hourly cloud-cover noise floor).")
    return a, b


# Keep old name as an alias so scripts that called calibrate_pdc0 still work.
def calibrate_pdc0(
    nasa_overlap: pd.DataFrame,
    local_hourly: pd.DataFrame,
) -> float:
    """
    Legacy wrapper — calls calibrate_polynomial and returns effective pdc0.

    The returned float is the linear-equivalent pdc0 (coefficient a × 1000),
    useful only for logging and the annual summary. The simulation itself
    now uses the full polynomial.
    """
    a, _ = calibrate_polynomial(nasa_overlap, local_hourly)
    return a * 1000.0


def simulate_pv(
    nasa_df:               pd.DataFrame,
    poly_coeffs:           tuple[float, float],
    seasonal_coeffs:       dict[str, tuple[float, float]] | None = None,
    monthly_coeffs:        dict[int, tuple[float, float]] | None = None,
    sky_stratified_coeffs: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Simulate AC PV power using the calibrated polynomial.

    Calibration priority (highest to lowest):
      1. sky_stratified_coeffs — per-(month, sky_condition) polynomial (best)
      2. monthly_coeffs        — per-month polynomial
      3. seasonal_coeffs       — dry / wet polynomial
      4. poly_coeffs           — single global polynomial (fallback)

    Sky model used internally:
      Dry months (Jan–Mar, Jun–Sep) → isotropic
      Wet months (Apr–May, Oct–Dec) → Perez

    Parameters
    ----------
    nasa_df : pd.DataFrame
        Full irradiance data (must contain ALLSKY_SFC_SW_DWN_cal, T2M_cal).
        For sky_stratified_coeffs, also needs CLRSKY_SFC_SW_DWN_cal and
        cloud_opacity (or clearsky_ghi).
    poly_coeffs : tuple[float, float]
        Global (a, b) — fallback when monthly or seasonal not available.
    seasonal_coeffs : dict, optional
        {"dry": (a, b), "wet": (a, b)} from calibrate_seasonal().
    monthly_coeffs : dict, optional
        {1: (a, b), …, 12: (a, b)} from calibrate_monthly().
        Takes priority over seasonal_coeffs.
    sky_stratified_coeffs : dict, optional
        {(month, sky_condition): (a, b), …} from calibrate_sky_stratified().
        Takes priority over monthly_coeffs.

    Returns
    -------
    pd.DataFrame
        Columns: pv_ac_W, sim_1kw, poa_global, temp_cell, solar_elevation
    """
    if sky_stratified_coeffs:
        logger.info(
            f"Simulating PV power (sky-stratified calibration + adaptive sky model) …"
            f"  [{len(sky_stratified_coeffs)} (month, sky_condition) cells]"
        )
    elif monthly_coeffs:
        lines = "\n".join(
            f"  {_MONTH_NAMES[m-1]:>3}: P_ac = {a:.2f}·sim + {b:.5f}·sim²"
            for m, (a, b) in sorted(monthly_coeffs.items())
        )
        logger.info(f"Simulating PV power (monthly calibration + adaptive sky model) …\n{lines}")
    elif seasonal_coeffs:
        logger.info(
            f"Simulating PV power (seasonal calibration + adaptive sky model) …\n"
            f"  dry (Jan-Mar,Jun-Sep): P_ac = {seasonal_coeffs['dry'][0]:.2f}·sim"
            f" + {seasonal_coeffs['dry'][1]:.5f}·sim²\n"
            f"  wet (Apr-May,Oct-Dec): P_ac = {seasonal_coeffs['wet'][0]:.2f}·sim"
            f" + {seasonal_coeffs['wet'][1]:.5f}·sim²"
        )
    else:
        a, b = poly_coeffs
        logger.info(f"Simulating PV power  (P_ac = {a:.2f}·sim + {b:.5f}·sim²) …")

    poa, temp_cell, solar_elev = _compute_poa_and_temp(nasa_df)

    sim_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa,
        temp_cell=temp_cell,
        pdc0=1000.0,
        gamma_pdc=_GAMMA_PDC,
    ).clip(lower=0.0)

    # ── Apply calibration ─────────────────────────────────────────────────────
    pv_ac = pd.Series(0.0, index=nasa_df.index)

    if sky_stratified_coeffs:
        # Compute kt and sky condition for every row
        ghi = nasa_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
        if "CLRSKY_SFC_SW_DWN_cal" in nasa_df.columns:
            clrsky = nasa_df["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan).clip(lower=1)
        elif "clearsky_ghi" in nasa_df.columns:
            clrsky = nasa_df["clearsky_ghi"].replace(0, np.nan).clip(lower=1)
        else:
            clrsky = None

        if clrsky is not None:
            kt = (ghi / clrsky).clip(0, 1.2).fillna(0)
        else:
            kt = pd.Series(0.5, index=nasa_df.index)

        if "cloud_opacity" in nasa_df.columns:
            cloud_op = nasa_df["cloud_opacity"]
        else:
            cloud_op = ((1 - kt) * 100).clip(0, 100)

        sky_cond = classify_sky_condition(cloud_op, kt)

        for m in range(1, 13):
            for s in range(4):
                mask = (nasa_df.index.month == m) & (sky_cond == s)
                if not mask.any():
                    continue
                # Fallback hierarchy: sky_stratified → monthly → seasonal → global
                if (m, s) in sky_stratified_coeffs:
                    am, bm = sky_stratified_coeffs[(m, s)]
                elif monthly_coeffs and m in monthly_coeffs:
                    am, bm = monthly_coeffs[m]
                elif seasonal_coeffs:
                    key = "dry" if m in _DRY_MONTHS else "wet"
                    am, bm = seasonal_coeffs[key]
                else:
                    am, bm = poly_coeffs
                pv_ac[mask] = (am * sim_1kw[mask] + bm * sim_1kw[mask] ** 2).clip(lower=0.0)

    elif monthly_coeffs:
        # Per-month polynomial; missing months fall back to seasonal → global
        for m in range(1, 13):
            m_mask = nasa_df.index.month == m
            if m in monthly_coeffs:
                am, bm = monthly_coeffs[m]
            elif seasonal_coeffs:
                key = "dry" if m in _DRY_MONTHS else "wet"
                am, bm = seasonal_coeffs[key]
            else:
                am, bm = poly_coeffs
            pv_ac[m_mask] = (am * sim_1kw[m_mask] + bm * sim_1kw[m_mask] ** 2).clip(lower=0.0)
    elif seasonal_coeffs:
        a_dry, b_dry = seasonal_coeffs["dry"]
        a_wet, b_wet = seasonal_coeffs["wet"]
        dry_mask = nasa_df.index.month.isin(_DRY_MONTHS)
        pv_ac[dry_mask]  = (a_dry * sim_1kw[dry_mask]  + b_dry * sim_1kw[dry_mask]  ** 2).clip(lower=0.0)
        pv_ac[~dry_mask] = (a_wet * sim_1kw[~dry_mask] + b_wet * sim_1kw[~dry_mask] ** 2).clip(lower=0.0)
    else:
        a, b = poly_coeffs
        pv_ac = (a * sim_1kw + b * sim_1kw ** 2).clip(lower=0.0)

    # Force night to zero
    pv_ac[solar_elev <= 0] = 0.0

    result = pd.DataFrame({
        "pv_ac_W":         pv_ac,
        "sim_1kw":         sim_1kw,
        "poa_global":      poa,
        "temp_cell":       temp_cell,
        "solar_elevation": solar_elev,
    }, index=nasa_df.index)

    day = result[result["solar_elevation"] > 0]
    logger.info(f"  Total rows     : {len(result):,}")
    logger.info(f"  Daytime hours  : {len(day):,}")
    logger.info(f"  AC power max   : {result['pv_ac_W'].max()/1000:.1f} kW")
    logger.info(f"  AC power mean  : {day['pv_ac_W'].mean()/1000:.1f} kW  (daytime)")
    logger.info(f"  Range          : {result.index.min()}  →  {result.index.max()}")
    return result


def save_synthetic(df: pd.DataFrame, cfg: dict) -> None:
    """
    Save the simulated PV time series to data/synthetic/pv_synthetic_6yr.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulate_pv().
    cfg : dict
        Config dict from load_config().
    """
    out_dir  = resolve_path(cfg["paths"]["synthetic"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pv_synthetic_6yr.csv"
    df.to_csv(out_path)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved → {out_path}  ({size_kb:.1f} KB)")


def load_synthetic(cfg: dict) -> pd.DataFrame:
    """
    Load the saved synthetic PV CSV from data/synthetic/.

    Returns
    -------
    pd.DataFrame  with UTC-aware DatetimeIndex.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    """
    path = resolve_path(cfg["paths"]["synthetic"]) / "pv_synthetic_6yr.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic PV file not found: {path}\n"
            "Run  python scripts/run_pv_model.py  to generate it."
        )
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(
        f"Loaded synthetic PV: {len(df):,} rows  "
        f"({df.index.min()} → {df.index.max()})"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Empirical noise model  (NREL/TP-5K00-86459 approach, adapted for 5-min site)
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_PERCENTILES = np.array([1, 10, 20, 40, 50, 60, 80, 90, 99], dtype=float)
_CLEAR_KT_THRESH   = 0.65   # kt threshold separating clear from cloudy sky
_MIN_BIN_SAMPLES   = 5      # minimum pts to populate a 2-D CDF bin


def _find_nearest_noise_cdf(
    cloudy_cdf:   dict,
    irr_b:        float,
    var_b:        float,
    irr_bin_size: float,
    var_bin_size: float,
    search_steps: int = 3,
) -> "np.ndarray | None":
    """Return the nearest populated CDF in the 2-D (irr, var) lookup table."""
    for di in range(search_steps):
        for dv in (0.0, var_bin_size, -var_bin_size, var_bin_size * 2, -var_bin_size * 2):
            key = (irr_b + di * irr_bin_size, round(var_b + dv, 1))
            if key in cloudy_cdf:
                return cloudy_cdf[key]
    return None


def build_noise_model(
    solcast_overlap: pd.DataFrame,
    actual_pv:       pd.DataFrame,
    physics_pv:      pd.DataFrame,
    irr_bin_size:    float = 100.0,
    var_bin_size:    float = 0.5,
) -> dict:
    """
    Build an empirical noise model from the 1-year overlap period.

    Adapts NREL/TP-5K00-86459 to 5-min tropical site data.

    N5min = (actual_pv_W – physics_pv_W) / P_max_W

    Partitioning
    ─────────────
    Clear sky (kt ≥ 0.65):
        Noise is stable within a day → one daily median N5min per day.
        Empirical CDF over all clear-sky day medians.

    Cloudy sky (kt < 0.65):
        5-min noise depends on irradiance level and short-term variability.
        2-D CDF lookup:  {(irr_bin, var_bin): percentile_array}
        irr_bin  = ⌊GHI / irr_bin_size⌋ × irr_bin_size  [W/m²]
        var_bin  = round(log₁₀(rolling_6step_std_GHI) / var_bin_size) × var_bin_size

    Parameters
    ----------
    solcast_overlap : 5-min Solcast data with ALLSKY_SFC_SW_DWN_cal and
                      CLRSKY_SFC_SW_DWN_cal (or clearsky_ghi).
    actual_pv       : 5-min actual PV measurements (must contain _PV_POWER_COL).
    physics_pv      : output of simulate_pv() on the same overlap period.
    irr_bin_size    : W/m² bin width for the cloudy CDF (default 100).
    var_bin_size    : log₁₀ bin width for the variability axis (default 0.5).

    Returns
    -------
    dict
        clear_cdf    – np.ndarray  percentile values for clear-sky daily bias
        cloudy_cdf   – dict {(irr_bin, var_bin): np.ndarray}
        percentiles  – np.ndarray  (1,10,20,40,50,60,80,90,99)
        p_max_w      – float  system capacity (99th pct actual daytime PV)
        irr_bin_size – float
        var_bin_size – float
    """
    logger.info("Building empirical noise model from overlap period …")

    ghi = solcast_overlap["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    if "CLRSKY_SFC_SW_DWN_cal" in solcast_overlap.columns:
        clrsky = solcast_overlap["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan).clip(lower=1)
    else:
        clrsky = solcast_overlap["clearsky_ghi"].replace(0, np.nan).clip(lower=1)

    kt       = (ghi / clrsky).clip(0, 1.2).fillna(0)
    actual_w = actual_pv[_PV_POWER_COL].clip(lower=0)
    phys_w   = physics_pv["pv_ac_W"].clip(lower=0)

    daytime_actual = actual_w[ghi >= 50].dropna()
    p_max_w = float(np.percentile(daytime_actual, 99)) if len(daytime_actual) > 0 else 100_000.0
    logger.info(f"  P_max (99th pct actual daytime) = {p_max_w / 1000:.1f} kW")

    df = pd.DataFrame({"ghi": ghi, "kt": kt, "actual": actual_w, "physics": phys_w}).dropna()
    df = df[df["ghi"] >= 50]
    df["N5min"] = (df["actual"] - df["physics"]) / p_max_w

    rolling_std = ghi.rolling(window=6, min_periods=3).std().clip(lower=0.1)
    log10_var   = np.log10(rolling_std).clip(-2.0, 3.0)

    # ── Clear-sky CDF: daily median bias ─────────────────────────────────────
    clear_df = df[df["kt"] >= _CLEAR_KT_THRESH]
    if len(clear_df) >= 10:
        daily_med = clear_df["N5min"].resample("D").median().dropna()
        clear_cdf = np.percentile(daily_med.values, _NOISE_PERCENTILES)
    else:
        logger.warning("  Clear-sky: insufficient data — CDF zeroed.")
        clear_cdf = np.zeros(len(_NOISE_PERCENTILES))
    logger.info(
        f"  Clear-sky: {len(clear_df):,} pts ({int(clear_df['kt'].ge(_CLEAR_KT_THRESH).mean()*100)}%)"
        f"  CDF range [{clear_cdf[0]:.3f}, {clear_cdf[-1]:.3f}]"
    )

    # ── Cloudy-sky 2-D CDF: irradiance × log₁₀ variability ──────────────────
    cloudy_df = df[df["kt"] < _CLEAR_KT_THRESH].copy()
    cloudy_df["log10_var"] = log10_var.loc[cloudy_df.index]
    cloudy_df["irr_bin"]   = (
        (cloudy_df["ghi"] / irr_bin_size).apply(np.floor) * irr_bin_size
    ).astype(int)
    cloudy_df["var_bin"]   = (
        (cloudy_df["log10_var"] / var_bin_size).round() * var_bin_size
    ).round(1)

    cloudy_cdf: dict = {}
    for (irr_b, var_b), group in cloudy_df.groupby(["irr_bin", "var_bin"], observed=True):
        if len(group) >= _MIN_BIN_SAMPLES:
            cloudy_cdf[(float(irr_b), float(var_b))] = np.percentile(
                group["N5min"].values, _NOISE_PERCENTILES
            )
    logger.info(
        f"  Cloudy-sky: {len(cloudy_df):,} pts → {len(cloudy_cdf)} (irr, var) bins populated"
    )

    return {
        "clear_cdf":    clear_cdf,
        "cloudy_cdf":   cloudy_cdf,
        "percentiles":  _NOISE_PERCENTILES,
        "p_max_w":      p_max_w,
        "irr_bin_size": irr_bin_size,
        "var_bin_size": var_bin_size,
    }


def apply_synthetic_noise(
    synthetic_pv: pd.DataFrame,
    solcast_df:   pd.DataFrame,
    noise_model:  dict,
    actual_index: "pd.DatetimeIndex | None" = None,
    random_seed:  int = 42,
) -> pd.DataFrame:
    """
    Apply site-matched empirical noise to synthetic PV labels.

    Uses distributions from build_noise_model().  Noise is injected only
    into non-overlap rows (the overlap year uses actual PV in the ML pipeline).

    Clear-sky days receive one sampled bias broadcast across all 5-min slots.
    Cloudy-sky timestamps each receive independently sampled noise from the
    (irr_bin, var_bin) CDF — matching the NREL 2-D cloudy-sky approach.

    Parameters
    ----------
    synthetic_pv : pd.DataFrame   output of simulate_pv()
    solcast_df   : pd.DataFrame   full Solcast data (for irradiance context)
    noise_model  : dict           from build_noise_model()
    actual_index : DatetimeIndex  overlap rows excluded from noise injection
    random_seed  : int

    Returns
    -------
    pd.DataFrame  copy of synthetic_pv with noisy pv_ac_W.
    """
    rng    = np.random.default_rng(random_seed)
    out    = synthetic_pv.copy()
    pv_arr = out["pv_ac_W"].values.astype(float)

    # ── Masks ─────────────────────────────────────────────────────────────────
    if actual_index is not None:
        non_overlap = ~synthetic_pv.index.isin(actual_index)
    else:
        non_overlap = np.ones(len(synthetic_pv), dtype=bool)

    is_day    = (synthetic_pv["solar_elevation"] > 0).values
    non_ov_np = non_overlap.values if hasattr(non_overlap, "values") else np.asarray(non_overlap)
    apply_arr = non_ov_np & is_day

    # ── Align Solcast irradiance to synthetic index ───────────────────────────
    ghi_full = (
        solcast_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
        .reindex(synthetic_pv.index).ffill().fillna(0.0)
    )
    clrsky_col = (
        "CLRSKY_SFC_SW_DWN_cal" if "CLRSKY_SFC_SW_DWN_cal" in solcast_df.columns
        else "clearsky_ghi"
    )
    clrsky_full = (
        solcast_df[clrsky_col].replace(0, np.nan).clip(lower=1)
        .reindex(synthetic_pv.index).ffill().fillna(1.0)
    )

    kt_full        = (ghi_full / clrsky_full).clip(0, 1.2).fillna(0)
    rolling_std    = ghi_full.rolling(window=6, min_periods=3).std().clip(lower=0.1).fillna(0.1)
    log10_var_full = np.log10(rolling_std).clip(-2.0, 3.0)

    clear_cdf    = noise_model["clear_cdf"]
    cloudy_cdf   = noise_model["cloudy_cdf"]
    percs        = noise_model["percentiles"]
    p_max_w      = noise_model["p_max_w"]
    irr_bin_size = noise_model["irr_bin_size"]
    var_bin_size = noise_model["var_bin_size"]

    noise_arr = np.zeros(len(synthetic_pv))

    # ── 1. Clear-sky: one daily bias per day, broadcast to 5-min slots ───────
    clear_arr = apply_arr & (kt_full.values >= _CLEAR_KT_THRESH)
    if clear_arr.any():
        clear_dates  = synthetic_pv.index[clear_arr].normalize().unique()
        u_vals       = rng.uniform(0, 100, size=len(clear_dates))
        daily_biases = np.interp(u_vals, percs, clear_cdf)
        date_map     = pd.Series(daily_biases, index=clear_dates)

        all_norm = synthetic_pv.index.normalize()
        mapped   = date_map.reindex(all_norm[clear_arr]).values
        noise_arr[clear_arr] = np.where(np.isnan(mapped), 0.0, mapped)

    # ── 2. Cloudy-sky: per-5-min noise from 2-D CDF (vectorised by bin) ──────
    cloudy_arr = apply_arr & (kt_full.values < _CLEAR_KT_THRESH)
    if cloudy_arr.any():
        ghi_c    = ghi_full.values[cloudy_arr]
        var_c    = log10_var_full.values[cloudy_arr]
        irr_bins = (np.floor(ghi_c / irr_bin_size) * irr_bin_size).astype(int)
        var_bins = np.round(np.round(var_c / var_bin_size) * var_bin_size, 1)

        cloudy_noise = np.zeros(cloudy_arr.sum())
        tmp = pd.DataFrame({"irr_bin": irr_bins, "var_bin": var_bins})
        for (irr_b, var_b), grp in tmp.groupby(["irr_bin", "var_bin"], observed=True):
            cdf_vals = _find_nearest_noise_cdf(
                cloudy_cdf, float(irr_b), float(var_b), irr_bin_size, var_bin_size
            )
            if cdf_vals is not None:
                u = rng.uniform(0, 100, size=len(grp))
                cloudy_noise[grp.index.values] = np.interp(u, percs, cdf_vals)
        noise_arr[cloudy_arr] = cloudy_noise

    # ── Apply noise: pv_noisy = clip(pv_clean × (1 + noise), 0, 1.1 × P_max) ─
    apply_pos         = np.where(apply_arr)[0]
    pv_arr[apply_pos] = np.clip(
        pv_arr[apply_pos] * (1.0 + noise_arr[apply_pos]),
        0.0,
        p_max_w * 1.1,
    )
    pv_arr[~is_day] = 0.0
    out["pv_ac_W"]  = pv_arr

    noisy_vals = noise_arr[apply_pos]
    logger.info(
        f"  Noise injected: {len(apply_pos):,} non-overlap daytime rows  "
        f"(mean|N|={np.abs(noisy_vals).mean():.3f}  std={noisy_vals.std():.3f})"
    )
    return out
