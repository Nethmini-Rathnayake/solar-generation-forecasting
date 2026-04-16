"""
scripts/run_hifi_pipeline.py
──────────────────────────────────────────────────────────────────────────────
High-Fidelity 6-Year Pseudo-Historical Solar PV Dataset Generator
University of Moratuwa  |  6.7912° N, 79.9005° E  |  5-min resolution

Goal
────
Produce a calibrated, physics-driven synthetic PV time series for
2020-01-01 → 2024-02-28 that maximises agreement with the 1-year
ground-truth PV measurements (Apr 2022 – Mar 2023).

Pipeline stages
───────────────
  1.  Load & synchronise all datasets to UTC 5-min grid
  2.  Polynomial Bias Correction (2nd/3rd order) for T, RH, wind
  3.  Clear-Sky Index (kt) GHI Normalisation
  4.  Sky-Regime Classification (Clear / PartlyCloudy / Overcast / HighlyVolatile)
  5.  Physics PV model  (Hay-Davies POA → Faiman T_cell → PVWatts AC)
  6.  Sky-Stratified Polynomial Calibration (per month × sky regime)
  7.  Regime-Aware AR(1) Noise Injection
  8.  Validation & Metrics against 1-year ground truth
  9.  Visualisation Suite
  10. Save 6-year synthetic dataset

Data availability note
──────────────────────
• "Ground-truth meteorology" = tempC, humidity, windspeedKmph from a
  commercial weather API co-located with the site (not an on-site
  pyranometer). No measured GHI/DNI/DHI is available.
• GHI calibration uses the inverse-PV-model approach: measured AC power
  is used to back-derive the effective site irradiance, from which the
  kt correction curve is learnt.
• The achievable R² ceiling with 5-min Solcast satellite data is ~0.87
  (satellite GHI resolution ≈ 5 km; sub-min cloud transients unresolvable).
  >96% accuracy is achievable at daily/weekly energy totals.

Run
───
    python scripts/run_hifi_pipeline.py
    python scripts/run_hifi_pipeline.py --poly-order 3 --noise-seed 7
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.data.solcast_loader import load_solcast_local_files, solcast_to_nasa_schema
from src.calibration.multistage import (
    fit_polynomial_corrections,
    apply_polynomial_corrections,
    fit_kt_normalisation,
    apply_kt_normalisation,
    classify_regime,
    compute_regime_noise_params,
    generate_ar1_noise,
    eval_metrics,
    REGIMES,
)

logger = get_logger("run_hifi_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
_OUT_DIR  = Path("results/figures/hifi")
_MET_DIR  = Path("results/metrics/hifi")
_SYN_PATH = Path("data/synthetic/hifi_solcast_pv_6yr_5min.csv")

# ── PV system constants ────────────────────────────────────────────────────────
_LAT      =  6.7912
_LON      = 79.9005
_ELEV_M   = 20
_TILT     = 10
_AZIMUTH  = 180
_GAMMA    = -0.0037
_ETA_INV  = 0.96
_PDC0_W   = 259_000.0    # empirical nameplate (W) — from calibrate_polynomial()

_PV_COL   = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"

# Regime palette
_REGIME_COLORS = {
    "Clear":          "#f4a261",
    "PartlyCloudy":   "#90be6d",
    "Overcast":       "#577590",
    "HighlyVolatile": "#e63946",
}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Load & synchronise
# ═════════════════════════════════════════════════════════════════════════════

def load_and_align(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data sources and return them on a common 5-min UTC index.

    Returns
    -------
    solcast_full  : 4-year Solcast (2020–2024)
    local_5min    : 1-year local PV + weather (2022-04 – 2023-03)
    solcast_ovlp  : Solcast clipped to overlap window
    """
    # ── Solcast 5-min (2020–2024) ─────────────────────────────────────────────
    logger.info("Stage 1 — Loading Solcast 4-year data …")
    raw_sc       = load_solcast_local_files(cfg)
    solcast_full = solcast_to_nasa_schema(raw_sc)

    # Also keep original Solcast columns (cloud_opacity etc.)
    for col in ("cloud_opacity", "clearsky_ghi", "ghi", "air_temp",
                "relative_humidity", "wind_speed_10m", "dhi", "dni"):
        if col in raw_sc.columns and col not in solcast_full.columns:
            solcast_full[col] = raw_sc[col]

    logger.info(
        f"  Solcast: {len(solcast_full):,} rows  "
        f"({solcast_full.index.min().date()} → {solcast_full.index.max().date()})"
    )

    # ── Local 5-min (2022-04 – 2023-03) ───────────────────────────────────────
    logger.info("Stage 1 — Loading local 5-min data …")
    interim    = resolve_path(cfg["paths"]["interim"])
    local_5min = pd.read_csv(
        interim / "local_5min_utc.csv",
        index_col="timestamp_utc", parse_dates=True
    )
    local_5min.index = pd.to_datetime(local_5min.index, utc=True)
    logger.info(
        f"  Local: {len(local_5min):,} rows  "
        f"({local_5min.index.min().date()} → {local_5min.index.max().date()})"
    )

    # ── Overlap window ────────────────────────────────────────────────────────
    t0 = local_5min.index.min()
    t1 = local_5min.index.max()
    solcast_ovlp = solcast_full.loc[t0:t1]
    logger.info(
        f"  Overlap: {len(solcast_ovlp):,} rows  ({t0.date()} → {t1.date()})"
    )

    return solcast_full, local_5min, solcast_ovlp


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2–3: Polynomial + kt calibration
# ═════════════════════════════════════════════════════════════════════════════

def run_calibration(
    solcast_full: pd.DataFrame,
    solcast_ovlp: pd.DataFrame,
    local_5min:   pd.DataFrame,
    poly_order:   int,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run multi-stage calibration and return the corrected 6-year dataframe.

    Returns
    -------
    sc_cal        : calibrated full Solcast dataframe
    poly_corr     : polynomial correction coefficients (for logging)
    kt_params     : kt normalisation parameters (for logging)
    """
    # ── Stage 2: Polynomial bias correction ───────────────────────────────────
    logger.info(f"\nStage 2 — Polynomial Bias Correction (order {poly_order}) …")
    poly_corr = fit_polynomial_corrections(solcast_ovlp, local_5min, poly_order)
    sc_poly   = apply_polynomial_corrections(solcast_full, poly_corr)

    # Use corrected temperature / wind for downstream steps
    if "air_temp_pcal" in sc_poly.columns:
        sc_poly["air_temp"] = sc_poly["air_temp_pcal"]
    if "wind_speed_10m_pcal" in sc_poly.columns:
        sc_poly["wind_speed_10m"] = sc_poly["wind_speed_10m_pcal"]

    # ── Stage 3: kt GHI normalisation ────────────────────────────────────────
    logger.info(f"\nStage 3 — Clear-Sky Index (kt) GHI Normalisation …")

    # Use overlap-window slice of the poly-corrected dataframe
    sc_ovlp_poly = sc_poly.loc[local_5min.index.min():local_5min.index.max()]
    kt_params    = fit_kt_normalisation(
        sc_ovlp_poly, local_5min, _PV_COL, pdc0_w=_PDC0_W, poly_order=poly_order
    )
    sc_cal = apply_kt_normalisation(sc_poly, kt_params)

    # Swap calibrated GHI into the primary column so downstream uses it
    if "ghi_kt_cal" in sc_cal.columns:
        sc_cal["ghi"] = sc_cal["ghi_kt_cal"]
    if "ALLSKY_SFC_SW_DWN_cal" in sc_cal.columns:
        sc_cal["ALLSKY_SFC_SW_DWN_cal"] = sc_cal["ghi_kt_cal"]

    return sc_cal, poly_corr, kt_params


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Sky-Regime Classification
# ═════════════════════════════════════════════════════════════════════════════

def classify_sky_regimes(sc_cal: pd.DataFrame) -> pd.Series:
    """
    Classify each 5-min interval into one of four sky regimes.

    Returns pd.Series[int8] aligned to sc_cal.index.
    """
    logger.info("\nStage 4 — Sky-Regime Classification …")
    ghi_col = next((c for c in ("ghi", "ALLSKY_SFC_SW_DWN_cal") if c in sc_cal.columns), None)
    cks_col = next((c for c in ("clearsky_ghi", "CLRSKY_SFC_SW_DWN_cal") if c in sc_cal.columns), None)

    ghi    = sc_cal[ghi_col].clip(lower=0)
    clrsky = sc_cal[cks_col].replace(0, np.nan).clip(lower=1)
    kt     = (ghi / clrsky).clip(0, 1.2).fillna(0)

    cloud_op = sc_cal.get("cloud_opacity", None)
    regime   = classify_regime(kt, cloud_op)
    sc_cal["kt"]     = kt.astype(np.float32)
    sc_cal["regime"] = regime
    return regime


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5: Physics PV Model (Hay-Davies POA + Faiman + PVWatts)
# ═════════════════════════════════════════════════════════════════════════════

def run_physics_model(sc_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physics-based AC power for the full 6-year calibrated Solcast.

    Model chain
    ───────────
    GHI → Erbs decomposition → DNI, DHI
         → Perez POA irradiance (Hay-Davies for clear, Perez for cloudy)
         → Faiman cell temperature
         → PVWatts DC
         → AC (× η_inv)

    Returns
    -------
    pd.DataFrame with columns:
        poa_global, temp_cell, p_dc_W, p_ac_W, solar_elevation
    """
    logger.info("\nStage 5 — Physics PV Model (Perez POA + Faiman + PVWatts) …")

    location = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=_ELEV_M)
    sol_pos  = location.get_solarposition(sc_cal.index)

    ghi_col  = next((c for c in ("ghi", "ALLSKY_SFC_SW_DWN_cal") if c in sc_cal.columns))
    cks_col  = next((c for c in ("clearsky_ghi", "CLRSKY_SFC_SW_DWN_cal") if c in sc_cal.columns))
    ghi      = sc_cal[ghi_col].clip(lower=0)

    # Erbs decomposition → self-consistent DNI, DHI
    erbs = pvlib.irradiance.erbs(ghi, sol_pos["apparent_zenith"], sc_cal.index)
    dni  = erbs["dni"].clip(lower=0)
    dhi  = erbs["dhi"].clip(lower=0)

    # Extra-terrestrial DNI and airmass for Perez model
    dni_extra = pvlib.irradiance.get_extra_radiation(sc_cal.index)
    airmass   = location.get_airmass(solar_position=sol_pos)

    wet_months = {4, 5, 10, 11, 12}
    wet_mask   = sc_cal.index.month.isin(wet_months)

    # Hay-Davies (isotropic+circumsolar) for dry months
    poa_hd = pvlib.irradiance.get_total_irradiance(
        _TILT, _AZIMUTH,
        sol_pos["apparent_zenith"], sol_pos["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi, model="haydavies",
        dni_extra=dni_extra,
    )["poa_global"].fillna(0).clip(lower=0)

    # Perez model for wet months (better under diffuse/overcast)
    poa_perez = pvlib.irradiance.get_total_irradiance(
        _TILT, _AZIMUTH,
        sol_pos["apparent_zenith"], sol_pos["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi, model="perez",
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
    )["poa_global"].fillna(0).clip(lower=0)

    poa_global = poa_hd.copy()
    poa_global[wet_mask] = poa_perez[wet_mask]

    # Faiman cell temperature
    t_air = sc_cal.get("air_temp", pd.Series(28.0, index=sc_cal.index)).clip(lower=-5)
    wind  = sc_cal.get("wind_speed_10m", pd.Series(1.5, index=sc_cal.index)).clip(lower=0)
    t_cell = pvlib.temperature.faiman(poa_global, t_air, wind, u0=25.0, u1=6.84)

    # PVWatts DC → AC
    p_dc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance = poa_global,
        temp_cell            = t_cell,
        pdc0                 = _PDC0_W,
        gamma_pdc            = _GAMMA,
    ).clip(lower=0)
    p_ac = (p_dc * _ETA_INV).clip(lower=0)

    # Zero at night
    night = sol_pos["elevation"] <= 0
    p_ac[night]  = 0.0
    p_dc[night]  = 0.0
    poa_global[night] = 0.0

    result = pd.DataFrame({
        "poa_global"     : poa_global.astype(np.float32),
        "temp_cell"      : t_cell.astype(np.float32),
        "p_dc_W"         : p_dc.astype(np.float32),
        "p_ac_W"         : p_ac.astype(np.float32),
        "solar_elevation": sol_pos["elevation"].astype(np.float32),
    }, index=sc_cal.index)

    day = result["solar_elevation"] > 0
    logger.info(f"  Daytime rows : {day.sum():,}")
    logger.info(f"  AC power max : {result['p_ac_W'].max()/1000:.1f} kW")
    logger.info(f"  AC power mean: {result.loc[day,'p_ac_W'].mean()/1000:.1f} kW  (daytime)")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Stage 6: Sky-Stratified Polynomial Calibration
# ═════════════════════════════════════════════════════════════════════════════

def sky_stratified_calibration(
    pv_sim:       pd.DataFrame,
    local_5min:   pd.DataFrame,
    regime:       pd.Series,
    poly_order:   int = 2,
) -> tuple[pd.Series, dict]:
    """
    Fit per-(month, regime) polynomials  P_actual = f(P_physics).

    Returns
    -------
    p_cal    : pd.Series  calibrated AC power (W) for full 6-year period
    coeffs   : dict  {(month, regime): (a, b [, c])}  fitted coefficients
    """
    logger.info("\nStage 6 — Sky-Stratified Polynomial Calibration …")

    # Build calibration dataframe (overlap period)
    ovlp_idx = local_5min.index
    cal_df   = pd.DataFrame({
        "p_phys"  : pv_sim.loc[ovlp_idx, "p_ac_W"].values,
        "p_actual": local_5min[_PV_COL].clip(lower=0).values,
        "month"   : ovlp_idx.month,
        "regime"  : regime.loc[ovlp_idx].values,
        "ghi"     : pv_sim.loc[ovlp_idx, "poa_global"].values,
    }, index=ovlp_idx)

    # Daytime + positive actual
    cal_df = cal_df[(cal_df["ghi"] > 20) & (cal_df["p_actual"] > 100)]

    # Fit polynomials per (month, regime)
    MIN_PTS  = 30
    coeffs: dict = {}

    # Fallback: global polynomial
    x_all, y_all = cal_df["p_phys"].values, cal_df["p_actual"].values
    valid_all     = np.isfinite(x_all) & np.isfinite(y_all)
    global_coeffs = np.polyfit(x_all[valid_all], y_all[valid_all], poly_order)

    for m in range(1, 13):
        for r in range(4):
            sub = cal_df[(cal_df["month"] == m) & (cal_df["regime"] == r)]
            if len(sub) < MIN_PTS:
                coeffs[(m, r)] = global_coeffs
                continue
            x, y = sub["p_phys"].values, sub["p_actual"].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < MIN_PTS:
                coeffs[(m, r)] = global_coeffs
                continue
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                coeffs[(m, r)] = np.polyfit(x[valid], y[valid], poly_order)

    logger.info(f"  Fitted {len(coeffs)} (month × regime) cells")

    # Apply to full simulation (use float64 to avoid dtype mismatch warnings)
    p_cal = pd.Series(np.nan, index=pv_sim.index, dtype=np.float64)
    for m in range(1, 13):
        for r in range(4):
            mask  = (pv_sim.index.month == m) & (regime == r)
            c     = np.poly1d(coeffs.get((m, r), global_coeffs))
            p_cal[mask] = c(pv_sim.loc[mask, "p_ac_W"].values).clip(min=0)

    # Fill any remaining NaN with global polynomial
    nan_mask = p_cal.isna()
    if nan_mask.any():
        g = np.poly1d(global_coeffs)
        p_cal[nan_mask] = g(pv_sim.loc[nan_mask, "p_ac_W"].values).clip(min=0)

    p_cal = p_cal.astype(np.float32)

    # Night → zero
    night = pv_sim["solar_elevation"] <= 0
    p_cal[night] = 0.0

    return p_cal, coeffs


# ═════════════════════════════════════════════════════════════════════════════
# Stage 7: Regime-Aware AR(1) Noise Injection
# ═════════════════════════════════════════════════════════════════════════════

def inject_noise(
    p_cal:      pd.Series,
    local_5min: pd.DataFrame,
    regime:     pd.Series,
    seed:       int = 42,
) -> tuple[pd.Series, dict]:
    """
    Inject regime-aware AR(1) noise onto the calibrated physics output.

    Noise parameters are learnt from the calibration-window residuals,
    then applied to the full 6-year period.

    Returns
    -------
    p_final      : pd.Series  physics + noise (W)
    noise_params : dict       AR(1) parameters per regime
    """
    logger.info("\nStage 7 — Regime-Aware AR(1) Noise Injection …")

    ovlp_idx = local_5min.index
    p_actual = local_5min[_PV_COL].clip(lower=0)

    # Residual in kW
    residual_kw = (p_actual / 1000 - p_cal.loc[ovlp_idx] / 1000)
    regime_ovlp = regime.loc[ovlp_idx]

    # Restrict to daytime
    day_mask = p_actual > 100
    residual_kw = residual_kw[day_mask]
    regime_ovlp = regime_ovlp[day_mask]

    noise_params = compute_regime_noise_params(regime_ovlp, residual_kw)
    noise_kw     = generate_ar1_noise(regime, noise_params, seed=seed)

    # Add noise (in W) — clip to non-negative
    p_final = (p_cal + noise_kw * 1000).clip(lower=0).astype(np.float32)

    # Night always zero
    night = p_cal == 0
    p_final[night] = 0.0

    return p_final, noise_params


# ═════════════════════════════════════════════════════════════════════════════
# Stage 8: Validation
# ═════════════════════════════════════════════════════════════════════════════

def validate(
    p_physics:  pd.Series,
    p_cal:      pd.Series,
    p_final:    pd.Series,
    local_5min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute RMSE / MBE / R² at three pipeline stages vs 1-year ground truth.
    """
    logger.info("\nStage 8 — Validation against 1-year ground truth …")
    ovlp = local_5min.index
    obs  = local_5min[_PV_COL].clip(lower=0) / 1000   # → kW

    rows = []
    for label, pred_w in [
        ("Raw Physics (pre-cal)",   p_physics.loc[ovlp]),
        ("Sky-Strat. Calibrated",   p_cal.loc[ovlp]),
        ("+ AR(1) Noise",           p_final.loc[ovlp]),
    ]:
        pred_kw = pred_w / 1000
        m = eval_metrics(obs, pred_kw, label)
        m["stage"] = label
        rows.append(m)

    df = pd.DataFrame(rows).set_index("stage")
    print("\n" + "═" * 72)
    print("  VALIDATION RESULTS — 1-YEAR OVERLAP (Apr 2022 – Mar 2023)")
    print("  5-min resolution  |  daytime only (obs > 1 kW)")
    print("═" * 72)
    print(df[["R2", "RMSE", "MBE", "nRMSE", "n"]].to_string())
    print("═" * 72)

    # Monthly breakdown for final output
    df_monthly = _monthly_breakdown(obs, p_final.loc[ovlp] / 1000)
    print("\n  Monthly accuracy (final output):")
    print(df_monthly.to_string())
    return df


def _monthly_breakdown(obs_kw: pd.Series, pred_kw: pd.Series) -> pd.DataFrame:
    rows = []
    mnames = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        mask = obs_kw.index.month == m
        o, p = obs_kw[mask], pred_kw[mask]
        day  = o > 1.0
        if day.sum() < 10:
            continue
        rmse = float(np.sqrt(np.mean((o[day] - p[day])**2)))
        r2   = float(1 - np.sum((o[day] - p[day])**2) /
                     np.sum((o[day] - o[day].mean())**2))
        rows.append({"month": mnames[m-1], "R2": round(r2, 4),
                     "RMSE_kW": round(rmse, 2), "n": int(day.sum())})
    return pd.DataFrame(rows).set_index("month")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 9: Visualisation Suite
# ═════════════════════════════════════════════════════════════════════════════

def plot_before_after_scatter(
    p_physics:  pd.Series,
    p_cal:      pd.Series,
    p_final:    pd.Series,
    local_5min: pd.DataFrame,
) -> None:
    """
    3-panel scatter: Raw Physics | Sky-Strat. Calibrated | + Noise
    vs 1-year actual PV.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    ovlp = local_5min.index
    obs  = local_5min[_PV_COL].clip(lower=0) / 1000
    day  = obs > 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Before vs After Calibration — 1-Year Overlap (Apr 2022 – Mar 2023)\n"
        "University of Moratuwa  |  5-min resolution",
        fontsize=12, fontweight="bold",
    )

    panels = [
        (p_physics.loc[ovlp] / 1000, "Raw Physics (stage 5)",    "#aaaaaa"),
        (p_cal.loc[ovlp]     / 1000, "Sky-Strat. Calibrated",    "#2a9d8f"),
        (p_final.loc[ovlp]   / 1000, "Calibrated + AR(1) Noise", "#e63946"),
    ]

    for ax, (pred_kw, title, color) in zip(axes, panels):
        o, p = obs[day].values, pred_kw[day].values
        r2   = float(1 - np.sum((o - p)**2) / np.sum((o - o.mean())**2))
        rmse = float(np.sqrt(np.mean((o - p)**2)))
        mbe  = float(np.mean(p - o))

        ax.hexbin(o, p, gridsize=55, cmap="YlOrRd", mincnt=1)
        lim = max(o.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1 line")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Actual PV (kW)", fontsize=9)
        ax.set_ylabel(f"{title} (kW)", fontsize=9)
        ax.set_title(
            f"{title}\nR²={r2:.3f}  RMSE={rmse:.1f} kW  MBE={mbe:+.1f} kW",
            fontsize=9, color=color, fontweight="bold",
        )

    fig.tight_layout()
    out = _OUT_DIR / "01_before_after_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_mean_diurnal_profile(
    p_final:    pd.Series,
    local_5min: pd.DataFrame,
) -> None:
    """
    Mean Diurnal Profile ±1σ — actual vs synthetic, per calendar season.

    Seasons (Sri Lanka):
        Dry:  Jan-Mar, Jun-Sep
        Wet:  Apr-May, Oct-Dec
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    ovlp = local_5min.index
    obs  = local_5min[_PV_COL].clip(lower=0) / 1000
    pred = p_final.loc[ovlp] / 1000

    df = pd.DataFrame({"obs": obs, "pred": pred})
    df["hour_utc"] = df.index.hour + df.index.minute / 60.0
    df["month"]    = df.index.month

    season_map = {m: "Wet" if m in {4, 5, 10, 11, 12} else "Dry"
                  for m in range(1, 13)}
    df["season"] = df["month"].map(season_map)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Mean Diurnal PV Profile ±1σ  |  Actual vs Synthetic\n"
        "University of Moratuwa  |  Calibration Window Apr 2022 – Mar 2023",
        fontsize=12, fontweight="bold",
    )

    for ax, season in zip(axes, ["Dry", "Wet"]):
        sub = df[df["season"] == season]
        for col, label, color in [
            ("obs",  "Actual",    "steelblue"),
            ("pred", "Synthetic", "#e63946"),
        ]:
            grp   = sub.groupby("hour_utc")[col]
            mean_ = grp.mean()
            std_  = grp.std()
            ax.plot(mean_.index, mean_.values, lw=2, color=color, label=label)
            ax.fill_between(mean_.index,
                            (mean_ - std_).clip(lower=0),
                            mean_ + std_,
                            alpha=0.18, color=color)

        ax.set_title(
            f"{season} Season  "
            f"({'Jan–Mar, Jun–Sep' if season=='Dry' else 'Apr–May, Oct–Dec'})",
            fontsize=10,
        )
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=9); ax.set_xlim(0, 24); ax.set_ylim(0)
        ax.axvspan(0, 2, alpha=0.05, color="navy")  # pre-dawn
        ax.axvspan(12, 14, alpha=0.05, color="navy")  # solar noon UTC

    fig.tight_layout()
    out = _OUT_DIR / "02_diurnal_profile_by_season.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_residuals_by_irradiance_band(
    p_final:    pd.Series,
    p_physics:  pd.Series,
    local_5min: pd.DataFrame,
    sc_cal:     pd.DataFrame,
    regime:     pd.Series,
) -> None:
    """
    Residual (kW) by GHI irradiance band and sky regime (violin plot).

    Bands (W/m²): 0-100, 100-300, 300-500, 500-700, 700-900, 900+
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    ovlp = local_5min.index
    obs  = local_5min[_PV_COL].clip(lower=0) / 1000
    day  = obs > 1.0

    ghi_col = next((c for c in ("ghi", "ALLSKY_SFC_SW_DWN_cal") if c in sc_cal.columns))
    ghi     = sc_cal.loc[ovlp, ghi_col].clip(lower=0)

    df = pd.DataFrame({
        "obs":     obs[day],
        "pred_phy": p_physics.loc[ovlp][day] / 1000,
        "pred_fin": p_final.loc[ovlp][day]   / 1000,
        "ghi":     ghi[day],
        "regime":  regime.loc[ovlp][day].map(REGIMES),
    }).dropna()

    df["resid_phy"] = df["pred_phy"] - df["obs"]
    df["resid_fin"] = df["pred_fin"] - df["obs"]
    df["ghi_band"]  = pd.cut(
        df["ghi"],
        bins=[0, 100, 300, 500, 700, 900, 1400],
        labels=["0-100", "100-300", "300-500", "500-700", "700-900", "900+"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(
        "Residuals by Irradiance Band — Raw Physics vs Calibrated Output\n"
        f"Residual = Predicted − Actual (kW)  |  n = {len(df):,} daytime intervals",
        fontsize=12, fontweight="bold",
    )

    palette = list(_REGIME_COLORS.values())

    for ax, (col, title) in zip(axes, [
        ("resid_phy", "Raw Physics"),
        ("resid_fin", "After Full Calibration + Noise"),
    ]):
        sns.violinplot(
            data=df, x="ghi_band", y=col, hue="regime",
            palette=_REGIME_COLORS, ax=ax, inner="quartile",
            split=False, scale="width", linewidth=0.8,
        )
        ax.axhline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("GHI Band (W/m²)", fontsize=9)
        ax.set_ylabel("Residual (kW)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(-200, 200)
        ax.legend(title="Regime", fontsize=8, loc="upper left")

    fig.tight_layout()
    out = _OUT_DIR / "03_residuals_by_irradiance_band.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_6yr_overview(
    p_final: pd.Series,
    regime:  pd.Series,
) -> None:
    """
    Overview of the full 6-year synthetic dataset:
    daily energy production coloured by dominant sky regime.
    """
    sns.set_theme(style="whitegrid", font_scale=0.92)

    # Daily energy (kWh = kW × 5min/60)
    daily_kwh = (p_final / 1000 * (5 / 60)).resample("1D").sum()

    # Dominant regime per day
    dominant = regime.resample("1D").apply(lambda x: x.mode()[0] if len(x) > 0 else 1)
    dominant_name = dominant.map(REGIMES)

    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle(
        "6-Year Synthetic PV Dataset Overview  |  University of Moratuwa\n"
        "Physics-calibrated  |  Hay-Davies POA + Faiman + PVWatts + AR(1) noise",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: daily energy coloured by regime
    ax = axes[0]
    for regime_name, color in _REGIME_COLORS.items():
        mask = dominant_name == regime_name
        ax.bar(daily_kwh.index[mask], daily_kwh.values[mask],
               width=1, color=color, alpha=0.8, label=regime_name)
    ax.set_ylabel("Daily Energy (kWh)")
    ax.set_title("Daily PV Energy — coloured by dominant sky regime", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # Panel 2: monthly energy bar
    ax = axes[1]
    monthly_kwh = (p_final / 1000 * (5 / 60)).resample("ME").sum()
    ax.bar(monthly_kwh.index, monthly_kwh.values, width=20, color="steelblue", alpha=0.85)
    ax.set_ylabel("Monthly Energy (kWh)")
    ax.set_title("Monthly PV Energy", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # Panel 3: regime distribution pie
    ax = axes[2]
    regime_counts = regime.value_counts().sort_index()
    labels = [REGIMES.get(c, str(c)) for c in regime_counts.index]
    colors = [_REGIME_COLORS.get(l, "grey") for l in labels]
    ax.pie(regime_counts.values, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Sky Regime Distribution (all intervals)", fontsize=10)

    fig.tight_layout()
    out = _OUT_DIR / "04_6yr_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_kt_calibration_curve(kt_params: dict, sc_cal: pd.DataFrame) -> None:
    """Scatter of kt_solcast vs kt_calibrated with the fitted polynomial."""
    sns.set_theme(style="whitegrid", font_scale=0.95)

    if "kt_solcast" not in sc_cal.columns or "kt_calibrated" not in sc_cal.columns:
        return

    day = sc_cal.index.hour.isin(range(2, 13))  # UTC 2–13 = local 7:30–18:30
    x   = sc_cal.loc[day, "kt_solcast"].values
    y   = sc_cal.loc[day, "kt_calibrated"].values
    valid = (x > 0.02) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    coeffs = kt_params["coeffs"]
    x_line = np.linspace(0, 1.2, 200)
    y_line = sum(coeffs[k] * x_line**k for k in range(len(coeffs)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(x, y, gridsize=50, cmap="Blues", mincnt=1, alpha=0.8)
    ax.plot(x_line, y_line, "r-", lw=2, label=f"Poly fit  R²={kt_params['r2']:.3f}")
    ax.plot([0, 1.2], [0, 1.2], "k--", lw=1, label="1:1 line")
    ax.set_xlabel("kt Solcast (satellite)")
    ax.set_ylabel("kt Calibrated (target)")
    ax.set_title(
        "Clear-Sky Index Calibration Curve\n"
        "kt_calibrated = f(kt_solcast)  |  Derived via inverse PV model",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.3)

    out = _OUT_DIR / "05_kt_calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 10: Save output
# ═════════════════════════════════════════════════════════════════════════════

def save_output(
    p_final:  pd.Series,
    pv_sim:   pd.DataFrame,
    sc_cal:   pd.DataFrame,
    regime:   pd.Series,
) -> None:
    """
    Save the 6-year synthetic dataset to CSV.

    Columns: pv_ac_kW, pv_ac_W, poa_global, temp_cell, solar_elevation,
             kt_solcast, kt_calibrated, ghi_calibrated, sky_regime
    """
    out = pd.DataFrame({
        "pv_ac_kW"       : (p_final / 1000).clip(lower=0).round(4),
        "pv_ac_W"        : p_final.clip(lower=0).round(1),
        "poa_global"     : pv_sim["poa_global"].round(2),
        "temp_cell"      : pv_sim["temp_cell"].round(2),
        "solar_elevation": pv_sim["solar_elevation"].round(3),
        "kt_solcast"     : sc_cal.get("kt_solcast",     pd.Series(np.nan, index=sc_cal.index)).round(4),
        "kt_calibrated"  : sc_cal.get("kt_calibrated",  pd.Series(np.nan, index=sc_cal.index)).round(4),
        "ghi_calibrated" : sc_cal.get("ghi_kt_cal",     pd.Series(np.nan, index=sc_cal.index)).round(1),
        "sky_regime"     : regime.map(REGIMES),
    }, index=p_final.index)

    out.index.name = "timestamp_utc"
    _SYN_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_SYN_PATH)

    size_mb = _SYN_PATH.stat().st_size / 1e6
    logger.info(f"\n  Saved → {_SYN_PATH}  ({size_mb:.1f} MB  |  {len(out):,} rows)")

    # Annual energy summary
    annual = out.groupby(out.index.year)["pv_ac_kW"].sum() * (5 / 60)
    print("\n  Annual Energy Summary:")
    for yr, kwh in annual.items():
        print(f"    {yr}: {kwh/1000:.2f} MWh")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hi-Fi Solar PV Pipeline")
    parser.add_argument("--config",      default="configs/site.yaml")
    parser.add_argument("--poly-order",  type=int, default=3,
                        help="Polynomial order for bias correction (2 or 3)")
    parser.add_argument("--noise-seed",  type=int, default=42,
                        help="Random seed for AR(1) noise generation")
    parser.add_argument("--no-noise",    action="store_true",
                        help="Skip noise injection (pure physics output)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MET_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 72)
    print("  HIGH-FIDELITY SOLAR PV DATASET GENERATOR")
    print("  University of Moratuwa  |  6.7912°N 79.9005°E")
    print(f"  Polynomial order: {args.poly_order}  |  Noise seed: {args.noise_seed}")
    print("═" * 72)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    solcast_full, local_5min, solcast_ovlp = load_and_align(cfg)

    # ── Stages 2–3 ───────────────────────────────────────────────────────────
    sc_cal, poly_corr, kt_params = run_calibration(
        solcast_full, solcast_ovlp, local_5min, args.poly_order
    )

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    regime = classify_sky_regimes(sc_cal)

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    pv_sim = run_physics_model(sc_cal)

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    p_cal, strat_coeffs = sky_stratified_calibration(
        pv_sim, local_5min, regime, poly_order=args.poly_order
    )

    # ── Stage 7 ──────────────────────────────────────────────────────────────
    if args.no_noise:
        logger.info("\nStage 7 — Noise injection skipped (--no-noise)")
        p_final      = p_cal.copy()
        noise_params = {}
    else:
        p_final, noise_params = inject_noise(
            p_cal, local_5min, regime, seed=args.noise_seed
        )

    # ── Stage 8 ──────────────────────────────────────────────────────────────
    val_df = validate(pv_sim["p_ac_W"], p_cal, p_final, local_5min)
    val_df.to_csv(_MET_DIR / "hifi_validation.csv")

    # ── Stage 9 — Visualisations ──────────────────────────────────────────────
    logger.info("\nStage 9 — Generating visualisation suite …")
    plot_before_after_scatter(pv_sim["p_ac_W"], p_cal, p_final, local_5min)
    plot_mean_diurnal_profile(p_final, local_5min)
    plot_residuals_by_irradiance_band(p_final, pv_sim["p_ac_W"], local_5min, sc_cal, regime)
    plot_6yr_overview(p_final, regime)
    plot_kt_calibration_curve(kt_params, sc_cal)

    # ── Stage 10 ─────────────────────────────────────────────────────────────
    save_output(p_final, pv_sim, sc_cal, regime)

    print("\n" + "═" * 72)
    print("  PIPELINE COMPLETE")
    print(f"  Synthetic dataset : {_SYN_PATH}")
    print(f"  Figures           : {_OUT_DIR}/")
    print(f"  Metrics           : {_MET_DIR}/")
    print("═" * 72)


if __name__ == "__main__":
    main()
