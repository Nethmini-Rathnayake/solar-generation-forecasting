"""
scripts/run_pv_model.py
------------------------
Runs the physics-based PV simulation for the full 6-year dataset and
generates diagnostic plots.

Supports multiple irradiance sources via --source.  All output files are
prefixed with the source name so runs with different sources do not
overwrite each other.

Sources
-------
  nasa      NASA POWER 0.5° satellite (default)
            Pre-requisite: python scripts/run_calibration.py
  era5      ERA5 reanalysis 0.25° (ECMWF)
            Pre-requisite: python scripts/fetch_era5.py
  nsrdb     NSRDB PSM v3 ~4 km (NREL, free API key)
            Pre-requisite: python scripts/fetch_nsrdb.py
  solcast   Solcast ~1-2 km (free researcher tier)
            Pre-requisite: python scripts/fetch_solcast.py
  solargis  SolarGIS HelioSat-4 ~90 m (commercial/research)
            Pre-requisite: python scripts/fetch_solargis.py --local

Outputs (prefix = source name, e.g. "nsrdb_")
----------------------------------------------
  data/synthetic/<prefix>pv_synthetic_6yr.csv
  results/figures/<prefix>pv_6yr_annual_profiles.png
  results/figures/<prefix>pv_6yr_monthly_heatmap.png
  results/figures/<prefix>pv_6yr_timeseries.png
  results/figures/<prefix>syn_vs_act_validation_overlap.png
  results/figures/<prefix>syn_vs_act_validation_timeseries.png
  results/figures/<prefix>syn_vs_act_real_vs_synthetic.png
  results/figures/<prefix>syn_vs_act_monthly_comparison.png
  results/figures/<prefix>syn_vs_act_weekly_comparison.png
  results/figures/<prefix>syn_vs_act_daily_profiles.png
  results/metrics/<prefix>synthetic_accuracy_by_season.csv

Run from project root:
    python scripts/run_pv_model.py                    # NASA POWER (default)
    python scripts/run_pv_model.py --source era5      # ERA5 reanalysis
    python scripts/run_pv_model.py --source nsrdb     # NSRDB PSM v3
    python scripts/run_pv_model.py --source solcast   # Solcast
    python scripts/run_pv_model.py --source solargis  # SolarGIS
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.preproccesing.align import load_aligned, align_solcast_5min, save_aligned_5min
from src.physics.pv_model import (
    calibrate_polynomial, calibrate_seasonal, calibrate_monthly,
    calibrate_sky_stratified, simulate_pv, save_synthetic,
)

logger = get_logger("run_pv_model")

_PV_OBS_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — 6-year time series (monthly mean daily profile)
# ─────────────────────────────────────────────────────────────────────────────

def plot_6yr_timeseries(pv_sim: pd.DataFrame, cfg: dict) -> None:
    """
    Hourly AC power trace for the full 6-year period.

    Plotted as a daily mean per calendar day (reduces visual clutter)
    with ±1 std shading to show intra-day variance.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    # Daily mean of daytime AC power
    pv_day = pv_sim["pv_ac_W"].copy()
    pv_day[pv_day == 0] = np.nan    # suppress night zeros from mean/std

    daily_mean = pv_day.resample("1D").mean().dropna() / 1000   # kW
    daily_std  = pv_day.resample("1D").std().dropna() / 1000

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily_mean.index, daily_mean.values,
            lw=0.7, color="darkorange", alpha=0.9, label="Daily mean AC power")
    ax.fill_between(
        daily_mean.index,
        (daily_mean - daily_std).values.clip(min=0),
        (daily_mean + daily_std).values,
        color="darkorange", alpha=0.18, label="±1 std (intra-day)"
    )

    ax.set_ylabel("AC Power  (kW)", fontsize=11)
    ax.set_title(
        "Simulated PV AC Power — 6-Year Hourly Record  (2020–2026)\n"
        "Daily mean ± 1 std of daytime hours",
        fontsize=10, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "pv_6yr_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Annual energy bar chart (per year)
# ─────────────────────────────────────────────────────────────────────────────

def plot_annual_energy(pv_sim: pd.DataFrame, cfg: dict) -> None:
    """
    Annual AC energy in MWh per year (2020–2026).
    Bars coloured by year; 2026 marked as partial.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    # groupby year avoids timezone-aware year-end binning edge cases
    annual_mwh = (
        pv_sim.groupby(pv_sim.index.year)["pv_ac_W"]
        .sum()
        .div(1e6)           # W·h → MWh
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("Blues_d", len(annual_mwh))
    bars = ax.bar(annual_mwh.index.astype(str),
                  annual_mwh.values,
                  color=colors, edgecolor="white", width=0.6)

    for bar, (yr, val) in zip(bars, annual_mwh.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.set_ylabel("Annual AC Energy  (MWh)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title(
        "Simulated Annual PV Energy Production  (2020–2025)\n"
        "6 full years — NASA satellite irradiance available through 2025",
        fontsize=10, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "pv_6yr_annual_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Month × Hour heatmap (average AC power kW)
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_hour_heatmap(pv_sim: pd.DataFrame, cfg: dict) -> None:
    """
    Heatmap: rows = calendar month (Jan–Dec), cols = hour of day (0–23).
    Cell value = mean AC power in kW across all years.
    Shows the diurnal and seasonal generation pattern at a glance.
    """
    sns.set_theme(style="white", font_scale=0.95)

    df = pv_sim[["pv_ac_W"]].copy()
    df["month"] = df.index.month
    df["hour"]  = df.index.hour

    pivot = (
        df.groupby(["month", "hour"])["pv_ac_W"]
        .mean()
        .unstack("hour")
        .div(1000)          # W → kW
    )
    pivot.index = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd",
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Mean AC Power (kW)", "shrink": 0.8},
        fmt=".0f", annot=True, annot_kws={"size": 7},
    )
    ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.set_title(
        "Average PV AC Power by Month and Hour  (2020–2026, UTC)\n"
        "Sri Lanka local noon ≈ UTC 06:30–07:00",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "pv_6yr_monthly_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Validation: simulated vs observed (1-year overlap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_validation(pv_sim: pd.DataFrame, local_hourly: pd.DataFrame,
                    nasa_overlap: pd.DataFrame, poly_coeffs: tuple,
                    cfg: dict) -> None:
    """
    Four-panel validation plot showing the quadratic calibration improvement.

    Panel 1 — Scatter: sim vs obs with polynomial fit + 1:1 line
              (shows R², RMSE, equation clearly labelled)
    Panel 2 — Before vs After: linear (old) vs quadratic (new) fit scatter
    Panel 3 — Residuals vs GHI band: systematic error remaining per irradiance level
    Panel 4 — Mean diurnal profile: hourly mean sim vs obs across the full year
    """
    from scipy import stats as sp_stats
    import pvlib as _pvlib

    sns.set_theme(style="whitegrid", font_scale=0.95)
    a, b = poly_coeffs

    # ── Rebuild sim_1kw on the overlap window for scatter comparison ──────────
    loc_pvlib = _pvlib.location.Location(6.7912, 79.9005, tz="UTC", altitude=20)
    solar_pos = loc_pvlib.get_solarposition(nasa_overlap.index)
    poa = _pvlib.irradiance.get_total_irradiance(
        surface_tilt=10, surface_azimuth=180,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=nasa_overlap["ALLSKY_SFC_SW_DNI_cal"].clip(lower=0),
        ghi=nasa_overlap["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0),
        dhi=nasa_overlap["ALLSKY_SFC_SW_DIFF_cal"].clip(lower=0),
        model="isotropic",
    )
    poa_g     = poa["poa_global"].fillna(0).clip(lower=0)
    t_cell    = _pvlib.temperature.faiman(poa_g, nasa_overlap["T2M_cal"], nasa_overlap["WS10M_cal"])
    sim_1kw_s = _pvlib.pvsystem.pvwatts_dc(effective_irradiance=poa_g, temp_cell=t_cell,
                                            pdc0=1000.0, gamma_pdc=-0.0037)
    sim_1kw_s[solar_pos["elevation"] <= 0] = 0.0

    obs_raw = local_hourly[_PV_OBS_COL]
    ghi     = nasa_overlap["ALLSKY_SFC_SW_DWN_cal"]

    df_raw = pd.concat([
        sim_1kw_s.rename("sim_1kw"),
        obs_raw.rename("obs"),
        ghi.rename("ghi"),
    ], axis=1).dropna()
    df_day = df_raw[(df_raw["ghi"] >= 50) & (df_raw["obs"] > 100)].copy()

    # Predictions in W
    df_day["pred_poly"] = (a * df_day["sim_1kw"] + b * df_day["sim_1kw"] ** 2).clip(lower=0)
    sl_lin, ic_lin, r_lin, _, _ = sp_stats.linregress(df_day["sim_1kw"], df_day["obs"])
    df_day["pred_lin"]  = sl_lin * df_day["sim_1kw"] + ic_lin

    def _metrics(pred, obs):
        err  = pred - obs
        rmse = np.sqrt((err**2).mean())
        mae  = np.abs(err).mean()
        mbe  = err.mean()
        r2   = 1 - (err**2).sum() / ((obs - obs.mean())**2).sum()
        return rmse, mae, mbe, r2

    rmse_p, mae_p, mbe_p, r2_p = _metrics(df_day["pred_poly"].values, df_day["obs"].values)
    rmse_l, mae_l, mbe_l, r2_l = _metrics(df_day["pred_lin"].values,  df_day["obs"].values)

    # Convert to kW for display
    df_kw = df_day.copy()
    for col in ["obs","pred_poly","pred_lin"]:
        df_kw[col] /= 1000

    sub = df_kw.sample(min(300, len(df_kw)), random_state=42)
    x_line = np.linspace(df_kw["sim_1kw"].min(), df_kw["sim_1kw"].max(), 300)
    y_poly = (a * x_line + b * x_line**2) / 1000
    y_lin  = (sl_lin * x_line + ic_lin) / 1000

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "PV Model Validation — Quadratic Calibration  (1-Year Overlap: 2022–2023)\n"
        "Calibration: P_ac = a·sim + b·sim²  (zero-intercept, physically motivated)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Quadratic scatter ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(sub["sim_1kw"], sub["pred_poly"], alpha=0.30, s=15,
               color="darkorange", edgecolors="none", label=f"Simulated ({len(sub)} pts)")
    ax.scatter(sub["sim_1kw"], sub["obs"],        alpha=0.30, s=15,
               color="steelblue",  edgecolors="none", label="Observed")
    ax.plot(x_line, y_poly, color="crimson", lw=2.2,
            label="Quadratic fit")

    sign_b = "+" if b >= 0 else "−"
    eq_box = (
        f"Calibration equation:\n"
        f"  P_ac = {a:.2f}·sim {sign_b} {abs(b):.5f}·sim²\n"
        f"  (sim = pvwatts_dc at pdc0=1 kW)\n\n"
        f"Quadratic:  R² = {r2_p:.4f}\n"
        f"            RMSE = {rmse_p/1000:.1f} kW\n"
        f"            MAE  = {mae_p/1000:.1f} kW\n"
        f"            MBE  = {mbe_p/1000:+.1f} kW\n\n"
        f"⚠ R² ceiling ≈ 0.70\n"
        f"  (NASA hourly cloud noise\n"
        f"   σ_ratio = 121 W/kW)"
    )
    ax.text(0.02, 0.98, eq_box, transform=ax.transAxes, va="top", fontsize=7.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                      edgecolor="0.75", alpha=0.95))
    ax.set_xlabel("pvwatts sim_1kw  (W per 1 kW installed)", fontsize=9)
    ax.set_ylabel("AC Power (kW)", fontsize=9)
    ax.set_title("Quadratic Calibration  (adopted)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel 2: Before (linear) vs After (quadratic) ─────────────────────────
    ax = axes[0, 1]
    ax.scatter(sub["sim_1kw"], sub["obs"], alpha=0.25, s=14,
               color="steelblue", edgecolors="none", label="Observed", zorder=2)
    ax.plot(x_line, y_lin,  color="grey",   lw=1.8, linestyle="--",
            label=f"Linear OLS  R²={r2_l:.4f}  (old)")
    ax.plot(x_line, y_poly, color="crimson", lw=2.2,
            label=f"Quadratic   R²={r2_p:.4f}  (new)")

    lo = min(df_kw["sim_1kw"].min(), df_kw["obs"].min())
    hi = max(df_kw["sim_1kw"].max(), df_kw["obs"].max())
    comparison = (
        f"Before (linear OLS):\n"
        f"  y = {sl_lin:.2f}·x + {ic_lin:.0f}\n"
        f"  RMSE = {rmse_l/1000:.1f} kW  R²={r2_l:.4f}\n\n"
        f"After (quadratic, zero-int.):\n"
        f"  y = {a:.2f}·x {sign_b} {abs(b):.5f}·x²\n"
        f"  RMSE = {rmse_p/1000:.1f} kW  R²={r2_p:.4f}\n\n"
        f"ΔRMSE = {(rmse_l-rmse_p)/1000:+.1f} kW\n"
        f"ΔR²   = {r2_p-r2_l:+.4f}"
    )
    ax.text(0.02, 0.98, comparison, transform=ax.transAxes, va="top", fontsize=7.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f4f8",
                      edgecolor="0.75", alpha=0.95))
    ax.set_xlabel("pvwatts sim_1kw  (W per 1 kW installed)", fontsize=9)
    ax.set_ylabel("Observed AC Power (kW)", fontsize=9)
    ax.set_title("Before vs After Calibration", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel 3: Residuals by GHI band ────────────────────────────────────────
    ax = axes[1, 0]
    df_day["resid_poly"] = df_day["pred_poly"] - df_day["obs"]
    df_day["resid_lin"]  = df_day["pred_lin"]  - df_day["obs"]
    df_day["ghi_band"] = pd.cut(
        df_day["ghi"],
        bins=[0,100,200,300,400,500,600,700,800,900,1100],
        labels=["0-100","100-200","200-300","300-400","400-500",
                "500-600","600-700","700-800","800-900","900+"],
    )
    band_poly = df_day.groupby("ghi_band", observed=True)["resid_poly"].mean() / 1000
    band_lin  = df_day.groupby("ghi_band", observed=True)["resid_lin"].mean()  / 1000
    x_pos     = np.arange(len(band_poly))
    w = 0.38
    ax.bar(x_pos - w/2, band_lin.values,  width=w, color="grey",    alpha=0.7,
           label="Linear OLS (old)", edgecolor="white")
    ax.bar(x_pos + w/2, band_poly.values, width=w, color="crimson",  alpha=0.7,
           label="Quadratic (new)", edgecolor="white")
    ax.axhline(0, color="black", lw=1.0, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_poly.index, rotation=35, fontsize=8)
    ax.set_xlabel("GHI band (W/m²)", fontsize=9)
    ax.set_ylabel("Mean Residual  (kW)  [pred − obs]", fontsize=9)
    ax.set_title(
        "Systematic Bias by GHI Band\n"
        "Ideal: all bars near zero",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)

    # ── Panel 4: Mean diurnal profile ─────────────────────────────────────────
    ax = axes[1, 1]
    obs_kw  = local_hourly[_PV_OBS_COL] / 1000
    sim_kw  = pv_sim["pv_ac_W"] / 1000
    df_diur = pd.concat([obs_kw.rename("obs"), sim_kw.rename("sim")], axis=1).dropna()
    df_diur = df_diur[df_diur.index.isin(local_hourly.index)]

    mean_obs_h = df_diur.groupby(df_diur.index.hour)["obs"].mean()
    mean_sim_h = df_diur.groupby(df_diur.index.hour)["sim"].mean()
    std_obs_h  = df_diur.groupby(df_diur.index.hour)["obs"].std()
    std_sim_h  = df_diur.groupby(df_diur.index.hour)["sim"].std()

    hours = mean_obs_h.index
    ax.plot(hours, mean_obs_h, color="steelblue", lw=2, marker="s", ms=4,
            label="Observed (mean ±1σ)")
    ax.fill_between(hours,
                    (mean_obs_h - std_obs_h).clip(lower=0),
                    mean_obs_h + std_obs_h,
                    color="steelblue", alpha=0.15)
    ax.plot(hours, mean_sim_h, color="darkorange", lw=2, marker="o", ms=4,
            label="Simulated (mean ±1σ)")
    ax.fill_between(hours,
                    (mean_sim_h - std_sim_h).clip(lower=0),
                    mean_sim_h + std_sim_h,
                    color="darkorange", alpha=0.15)
    ax.set_xlabel("Hour of Day (UTC)  [local noon ≈ UTC 06:30]", fontsize=9)
    ax.set_ylabel("Mean AC Power (kW)", fontsize=9)
    ax.set_title("Mean Diurnal Profile ±1σ  (full year)", fontsize=10, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_validation_overlap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Validation time series (full overlap year)
# ─────────────────────────────────────────────────────────────────────────────

def plot_validation_timeseries(pv_sim: pd.DataFrame, local_hourly: pd.DataFrame,
                                cfg: dict) -> None:
    """
    Three-panel time series showing simulated vs observed over the full 1-year overlap.

    Panel 1 — Full year daily mean (simulated vs observed)
               Shows overall seasonal agreement and where the model diverges.
    Panel 2 — 14-day zoom: a clear period (January) — best-case match
    Panel 3 — 14-day zoom: a cloudy period (November) — worst-case mismatch

    Each panel shows the error band (sim − obs) shaded below, so the viewer
    can immediately see when and how much the model over/under-predicts.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    obs_kw = local_hourly[_PV_OBS_COL] / 1000
    sim_kw = pv_sim["pv_ac_W"] / 1000

    df = pd.concat([obs_kw.rename("obs"), sim_kw.rename("sim")], axis=1)
    df = df[df.index.isin(local_hourly.index)].dropna()

    # ── Daily means (suppress night zeros — average daytime only) ────────────
    obs_day = df["obs"].where(df["obs"] > 0.1)
    sim_day = df["sim"].where(df["sim"] > 0.1)
    daily_obs = obs_day.resample("1D").mean().dropna()
    daily_sim = sim_day.resample("1D").mean()
    daily_sim = daily_sim.reindex(daily_obs.index)
    daily_err = daily_sim - daily_obs

    # Pick clear window (Jan — R²≈0.89) and cloudy window (Nov — R²≈-2.35)
    clear_start  = pd.Timestamp("2023-01-10", tz="UTC")
    clear_end    = pd.Timestamp("2023-01-24", tz="UTC")
    cloudy_start = pd.Timestamp("2022-11-01", tz="UTC")
    cloudy_end   = pd.Timestamp("2022-11-15", tz="UTC")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(
        "PV Simulation vs Observed — Full 1-Year Validation Period  (2022-03-31 → 2023-03-31)\n"
        "Calibration: P_ac = 313.97·sim − 0.09782·sim²  (quadratic zero-intercept)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Full year daily mean ────────────────────────────────────────
    ax = axes[0]
    ax.plot(daily_obs.index, daily_obs.values, lw=1.0, color="steelblue",
            alpha=0.9, label="Observed (daily mean daytime)")
    ax.plot(daily_sim.index, daily_sim.values, lw=1.0, color="darkorange",
            alpha=0.9, label="Simulated (daily mean daytime)")

    # Error band: shade between obs and sim
    ax.fill_between(daily_err.index,
                    daily_obs.values,
                    daily_sim.reindex(daily_obs.index).values,
                    where=(daily_sim.reindex(daily_obs.index) > daily_obs),
                    alpha=0.18, color="red",   label="Over-prediction")
    ax.fill_between(daily_err.index,
                    daily_obs.values,
                    daily_sim.reindex(daily_obs.index).values,
                    where=(daily_sim.reindex(daily_obs.index) <= daily_obs),
                    alpha=0.18, color="blue",  label="Under-prediction")

    # Annotate November collapse
    ax.annotate(
        "Nov: northeast monsoon\nNASA GHI >> actual\nR²=−2.35",
        xy=(pd.Timestamp("2022-11-10", tz="UTC"), 10),
        xytext=(pd.Timestamp("2022-09-15", tz="UTC"), 60),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="red", alpha=0.85),
    )
    ax.annotate(
        "Jan–Feb: clear season\nR²≈0.90",
        xy=(pd.Timestamp("2023-01-20", tz="UTC"), 80),
        xytext=(pd.Timestamp("2022-12-01", tz="UTC"), 120),
        fontsize=8, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="steelblue", alpha=0.85),
    )

    # Overall metrics box
    err_all = df["sim"] - df["obs"]
    day_mask = (df["obs"] > 0.1) | (df["sim"] > 0.1)
    err_day  = err_all[day_mask]
    rmse_d   = np.sqrt((err_day**2).mean())
    mbe_d    = err_day.mean()
    r2_d     = 1 - (err_day**2).sum() / ((df.loc[day_mask,"obs"] - df.loc[day_mask,"obs"].mean())**2).sum()
    metrics_str = (
        f"Full-year metrics (daytime):\n"
        f"  RMSE = {rmse_d:.1f} kW  ({100*rmse_d/df.loc[day_mask,'obs'].mean():.0f}% of mean)\n"
        f"  MBE  = {mbe_d:+.1f} kW  ({100*mbe_d/df.loc[day_mask,'obs'].mean():.0f}% bias)\n"
        f"  R²   = {r2_d:.4f}"
    )
    ax.text(0.01, 0.97, metrics_str, transform=ax.transAxes, va="top", fontsize=8.2,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.6", alpha=0.92))
    ax.set_ylabel("Daily Mean Daytime Power (kW)", fontsize=10)
    ax.set_title("Full Year — Daily Mean Daytime PV Power", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper left",
              bbox_to_anchor=(0, 0.85))
    ax.tick_params(axis="x", rotation=15)

    # ── Panel 2: Clear-sky 14-day zoom (January) ─────────────────────────────
    ax = axes[1]
    win_c = df.loc[clear_start:clear_end]
    ax.plot(win_c.index, win_c["obs"], lw=1.2, color="steelblue",
            alpha=0.9, label="Observed")
    ax.plot(win_c.index, win_c["sim"], lw=1.2, color="darkorange",
            alpha=0.9, label="Simulated")
    ax.fill_between(win_c.index,
                    win_c["obs"], win_c["sim"],
                    where=(win_c["sim"] > win_c["obs"]),
                    alpha=0.15, color="red")
    ax.fill_between(win_c.index,
                    win_c["obs"], win_c["sim"],
                    where=(win_c["sim"] <= win_c["obs"]),
                    alpha=0.15, color="blue")

    err_c = win_c["sim"] - win_c["obs"]
    rmse_c = np.sqrt((err_c**2).mean())
    r2_c   = 1 - (err_c**2).sum() / ((win_c["obs"] - win_c["obs"].mean())**2).sum()
    ax.text(0.01, 0.97,
            f"Clear period (Jan 10–24)\n  RMSE={rmse_c:.1f} kW   R²={r2_c:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="0.7", alpha=0.92))
    ax.set_ylabel("AC Power (kW)", fontsize=10)
    ax.set_title("14-Day Zoom — Clear Period (January)  ← best-case match",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    # ── Panel 3: Cloudy/monsoon 14-day zoom (November) ───────────────────────
    ax = axes[2]
    win_n = df.loc[cloudy_start:cloudy_end]
    ax.plot(win_n.index, win_n["obs"], lw=1.2, color="steelblue",
            alpha=0.9, label="Observed")
    ax.plot(win_n.index, win_n["sim"], lw=1.2, color="darkorange",
            alpha=0.9, label="Simulated")
    ax.fill_between(win_n.index,
                    win_n["obs"], win_n["sim"],
                    where=(win_n["sim"] > win_n["obs"]),
                    alpha=0.22, color="red",   label="Over-prediction (cloud miss)")
    ax.fill_between(win_n.index,
                    win_n["obs"], win_n["sim"],
                    where=(win_n["sim"] <= win_n["obs"]),
                    alpha=0.18, color="blue")

    err_n = win_n["sim"] - win_n["obs"]
    rmse_n = np.sqrt((err_n**2).mean())
    r2_n   = 1 - (err_n**2).sum() / ((win_n["obs"] - win_n["obs"].mean())**2).sum()
    ax.text(0.01, 0.97,
            f"Monsoon period (Nov 1–15)\n  RMSE={rmse_n:.1f} kW   R²={r2_n:.3f}\n"
            f"  NASA GHI >> actual (cloud miss at 0.5° grid)",
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                      edgecolor="red", alpha=0.92))
    ax.set_ylabel("AC Power (kW)", fontsize=10)
    ax.set_title("14-Day Zoom — Monsoon Period (November)  ← worst-case mismatch",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_validation_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Real vs Synthetic: yearly / weekly / daily comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_real_vs_synthetic(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    cfg:          dict,
) -> None:
    """
    Three-panel comparison of real observed PV vs synthetic PV data.

    Panel 1 — Full overlap year: daily mean daytime power (kW)
        Shows overall agreement and where the seasonal calibration helps.

    Panel 2 — Two representative weeks side by side
        Left:  one dry-season week (Jan) — expected tight match
        Right: one wet-season week (Nov) — shows cloud-miss errors

    Panel 3 — Mean diurnal profile by season (dry vs wet)
        Averaged hour-of-day profile for real and synthetic, split into
        dry (Jan–Mar, Jun–Sep) and wet/transition (Apr–May, Oct–Dec).
        Shows whether the seasonal calibration corrects the shape.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    _PV_OBS_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
    if _PV_OBS_COL not in local_hourly.columns:
        logger.warning("Real PV column not found — skipping real vs synthetic plot.")
        return

    obs_kw = local_hourly[_PV_OBS_COL].dropna() / 1000
    sim_kw = pv_sim["pv_ac_W"].reindex(obs_kw.index) / 1000

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    fig.suptitle(
        "Real vs Synthetic PV Data — University of Moratuwa  (2022–2023)\n"
        "Seasonal calibration: dry (Jan–Mar, Jun–Sep) / wet (Apr–May, Oct–Dec)",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Full overlap year daily means ───────────────────────────────
    ax = axes[0]
    obs_day = obs_kw.where(obs_kw > 0.5)
    sim_day = sim_kw.where(sim_kw > 0.5)
    daily_obs = obs_day.resample("1D").mean().dropna()
    daily_sim = sim_day.resample("1D").mean().reindex(daily_obs.index)

    ax.plot(daily_obs.index, daily_obs.values, lw=1.2, color="#1565C0",
            alpha=0.9, label="Real observed (daily mean daytime)")
    ax.plot(daily_sim.index, daily_sim.values, lw=1.2, color="#E65100",
            alpha=0.85, linestyle="--", label="Synthetic pvlib (daily mean daytime)")
    ax.fill_between(daily_obs.index,
                    daily_obs.values, daily_sim.reindex(daily_obs.index).values,
                    where=(daily_sim.reindex(daily_obs.index) > daily_obs),
                    alpha=0.15, color="#E65100", label="Over-prediction")
    ax.fill_between(daily_obs.index,
                    daily_obs.values, daily_sim.reindex(daily_obs.index).values,
                    where=(daily_sim.reindex(daily_obs.index) <= daily_obs),
                    alpha=0.15, color="#1565C0", label="Under-prediction")

    # Season band annotations
    dry_months  = [1, 2, 3, 6, 7, 8, 9]
    for month_group, label, color in [
        ([1, 2, 3], "Dry", "#4CAF50"),
        ([4, 5],    "Wet", "#FF9800"),
        ([6, 7, 8, 9], "Dry", "#4CAF50"),
        ([10, 11, 12], "Wet", "#FF9800"),
    ]:
        for m in month_group:
            mask = (daily_obs.index.month == m)
            if mask.any():
                ax.axvspan(daily_obs.index[mask][0], daily_obs.index[mask][-1],
                           alpha=0.04, color=color, lw=0)

    overall_rmse = np.sqrt(((daily_sim.reindex(daily_obs.index) - daily_obs)**2).mean())
    ax.text(0.01, 0.97, f"Daily RMSE = {overall_rmse:.1f} kW",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88))
    ax.set_ylabel("Mean Daytime PV Power  [kW]", fontsize=10)
    ax.set_title("Full Overlap Year — Daily Mean  (real vs synthetic)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=15)

    # ── Panel 2: Representative weeks ────────────────────────────────────────
    ax = axes[1]
    # Find a clear dry week and a monsoon wet week within overlap
    dry_start  = pd.Timestamp("2023-01-09", tz="UTC")
    wet_start  = pd.Timestamp("2022-11-07", tz="UTC")
    dry_end    = dry_start  + pd.Timedelta(days=7)
    wet_end    = wet_start  + pd.Timedelta(days=7)

    # Clip to available data
    dry_obs = obs_kw.loc[dry_start:dry_end]
    dry_sim = sim_kw.loc[dry_start:dry_end]
    wet_obs = obs_kw.loc[wet_start:wet_end]
    wet_sim = sim_kw.loc[wet_start:wet_end]

    # Plot dry week on left portion of x-axis, wet on right with gap
    # Use integer x-axis: hours 0-167 for dry, 200-367 for wet
    hrs_dry = np.arange(len(dry_obs))
    hrs_wet = np.arange(len(wet_obs)) + len(dry_obs) + 30   # 30h gap

    ax.plot(hrs_dry, dry_obs.values, color="#1565C0", lw=1.2, alpha=0.9)
    ax.plot(hrs_dry, dry_sim.values, color="#E65100", lw=1.2, alpha=0.85, linestyle="--")
    ax.plot(hrs_wet, wet_obs.values, color="#1565C0", lw=1.2, alpha=0.9,
            label="Real observed")
    ax.plot(hrs_wet, wet_sim.values, color="#E65100", lw=1.2, alpha=0.85, linestyle="--",
            label="Synthetic pvlib")

    ax.fill_between(hrs_dry, dry_obs.values, dry_sim.values,
                    where=(dry_sim.values > dry_obs.values), alpha=0.15, color="#E65100")
    ax.fill_between(hrs_dry, dry_obs.values, dry_sim.values,
                    where=(dry_sim.values <= dry_obs.values), alpha=0.15, color="#1565C0")
    ax.fill_between(hrs_wet, wet_obs.values, wet_sim.values,
                    where=(wet_sim.values > wet_obs.values), alpha=0.15, color="#E65100")
    ax.fill_between(hrs_wet, wet_obs.values, wet_sim.values,
                    where=(wet_sim.values <= wet_obs.values), alpha=0.15, color="#1565C0")

    mid_dry = (hrs_dry[0] + hrs_dry[-1]) / 2
    mid_wet = (hrs_wet[0] + hrs_wet[-1]) / 2
    ax.axvline(hrs_dry[-1] + 15, color="gray", lw=1, ls=":", alpha=0.5)
    ax.text(mid_dry, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200,
            "Dry season\n(Jan 9–16 2023)", ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#E8F5E9", ec="#4CAF50", alpha=0.85))
    ax.text(mid_wet, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200,
            "Wet season\n(Nov 7–14 2022)", ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#FF9800", alpha=0.85))

    dry_rmse = np.sqrt(((dry_sim.values - dry_obs.values)**2).mean())
    wet_rmse = np.sqrt(((wet_sim.values - wet_obs.values)**2).mean())
    ax.set_xticks([])
    ax.set_ylabel("PV Power  [kW]", fontsize=10)
    ax.set_title(
        f"Representative Weeks — Dry (RMSE={dry_rmse:.1f} kW) vs Wet (RMSE={wet_rmse:.1f} kW)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # ── Panel 3: Mean diurnal profile by season ───────────────────────────────
    ax = axes[2]
    df_all = pd.concat([obs_kw.rename("obs"), sim_kw.rename("sim")], axis=1).dropna()
    df_all = df_all[df_all["obs"] > 0.5]   # daytime only

    dry_mask_all = df_all.index.month.isin([1, 2, 3, 6, 7, 8, 9])
    wet_mask_all = ~dry_mask_all

    colors = {
        ("dry", "obs"): ("#1565C0", "-",  "Real — Dry season"),
        ("dry", "sim"): ("#1565C0", "--", "Synthetic — Dry season"),
        ("wet", "obs"): ("#E65100", "-",  "Real — Wet season"),
        ("wet", "sim"): ("#E65100", "--", "Synthetic — Wet season"),
    }
    for (season, col), (color, ls, label) in colors.items():
        mask = dry_mask_all if season == "dry" else wet_mask_all
        data = df_all.loc[mask, col]
        mean_h = data.groupby(data.index.hour).mean()
        std_h  = data.groupby(data.index.hour).std()
        hrs    = mean_h.index
        ax.plot(hrs, mean_h.values, color=color, lw=2.0, ls=ls, label=label, marker="o", ms=3)
        if ls == "-":   # shade ±1σ for real data only
            ax.fill_between(hrs,
                            (mean_h - std_h).clip(lower=0).values,
                            (mean_h + std_h).values,
                            color=color, alpha=0.08)

    ax.set_xlabel("Hour of Day (UTC)  [local noon ≈ UTC 06:30]", fontsize=10)
    ax.set_ylabel("Mean PV Power  [kW]", fontsize=10)
    ax.set_title(
        "Mean Diurnal Profile — Real vs Synthetic by Season\n"
        "Shading = ±1σ of real observations  |  Dashed = synthetic",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_real_vs_synthetic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Yearly monthly breakdown: per-month real vs synthetic
# ─────────────────────────────────────────────────────────────────────────────

def plot_yearly_monthly_breakdown(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    cfg:          dict,
) -> None:
    """
    4×3 grid: one subplot per calendar month, comparing daily mean daytime
    power (real vs synthetic) across the overlap year.

    Shows systematic over/under-prediction patterns by month.
    Dry months (Jan–Mar, Jun–Sep) have green background; wet months amber.
    Each subplot annotated with RMSE, MBE, and R².
    """
    _PV_OBS_COL_L = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
    if _PV_OBS_COL_L not in local_hourly.columns:
        logger.warning("Real PV column not found — skipping monthly breakdown plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=0.9)

    obs_kw = local_hourly[_PV_OBS_COL_L].dropna() / 1000
    sim_kw = pv_sim["pv_ac_W"].reindex(obs_kw.index) / 1000

    obs_day   = obs_kw.where(obs_kw > 0.5)
    sim_day   = sim_kw.where(sim_kw > 0.5)
    daily_obs = obs_day.resample("1D").mean().dropna()
    daily_sim = sim_day.resample("1D").mean().reindex(daily_obs.index)

    dry_months_set = {1, 2, 3, 6, 7, 8, 9}
    month_names    = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]

    err_all   = (daily_sim - daily_obs).dropna()
    rmse_all  = np.sqrt((err_all**2).mean())
    mbe_all   = err_all.mean()
    obs_valid = daily_obs.reindex(err_all.index)
    r2_all    = 1 - (err_all**2).sum() / ((obs_valid - obs_valid.mean())**2).sum()

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        f"Synthetic vs Real PV — Monthly Daily Mean  (Overlap Year 2022–2023)\n"
        f"Overall: RMSE={rmse_all:.1f} kW  MBE={mbe_all:+.1f} kW  R²={r2_all:.3f}  |  "
        f"Green = Dry season  |  Amber = Wet season",
        fontsize=11, fontweight="bold",
    )

    for idx, month in enumerate(range(1, 13)):
        ax     = axes[idx // 3][idx % 3]
        season = "Dry" if month in dry_months_set else "Wet"
        ax.set_facecolor("#F1F8E9" if season == "Dry" else "#FFF8E1")

        mask  = daily_obs.index.month == month
        m_obs = daily_obs[mask]
        m_sim = daily_sim[mask]

        if m_obs.empty:
            ax.set_title(f"{month_names[month-1]}  ({season})\n[no data]", fontsize=9)
            continue

        ax.plot(m_obs.index, m_obs.values, color="#1565C0", lw=1.4, label="Real",
                marker="o", ms=3)
        ax.plot(m_sim.index, m_sim.values, color="#E65100", lw=1.4, linestyle="--",
                label="Synthetic", marker="s", ms=3)
        ax.fill_between(m_obs.index,
                        m_obs.values, m_sim.reindex(m_obs.index).values,
                        where=(m_sim.reindex(m_obs.index) > m_obs),
                        alpha=0.20, color="#E65100")
        ax.fill_between(m_obs.index,
                        m_obs.values, m_sim.reindex(m_obs.index).values,
                        where=(m_sim.reindex(m_obs.index) <= m_obs),
                        alpha=0.20, color="#1565C0")

        err_m = (m_sim.reindex(m_obs.index) - m_obs).dropna()
        if len(err_m) > 1:
            rmse_m = np.sqrt((err_m**2).mean())
            mbe_m  = err_m.mean()
            obs_m2 = m_obs.reindex(err_m.index)
            r2_m   = 1 - (err_m**2).sum() / ((obs_m2 - obs_m2.mean())**2).sum()
            metrics_txt = f"RMSE={rmse_m:.1f} kW\nMBE={mbe_m:+.1f} kW\nR²={r2_m:.3f}"
        else:
            metrics_txt = "n/a"

        ax.text(0.03, 0.97, metrics_txt,
                transform=ax.transAxes, va="top", fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.90))
        ax.set_title(f"{month_names[month-1]}  ({season})", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Mean daytime power (kW)", fontsize=7.5)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_monthly_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Weekly comparison: 4 representative seasonal windows
# ─────────────────────────────────────────────────────────────────────────────

def plot_weekly_seasons(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    cfg:          dict,
) -> None:
    """
    Four 7-day windows covering distinct atmospheric regimes.

    Row 1: Jan  9–16, 2023 — clear dry season (best-case match)
    Row 2: Apr 11–18, 2022 — inter-monsoon (patchy convective cloud)
    Row 3: Aug  8–15, 2022 — SW monsoon dry spell
    Row 4: Nov  7–14, 2022 — NE monsoon onset (worst-case mismatch)

    Left panel: hourly AC power (real=blue, synthetic=orange dashed) + error shading.
    Right panel: hourly error (sim − obs) bar chart.
    """
    _PV_OBS_COL_L = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
    if _PV_OBS_COL_L not in local_hourly.columns:
        logger.warning("Real PV column not found — skipping weekly seasons plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=0.9)

    obs_kw = local_hourly[_PV_OBS_COL_L] / 1000
    sim_kw = pv_sim["pv_ac_W"].reindex(obs_kw.index) / 1000

    weeks = [
        (pd.Timestamp("2023-01-09", tz="UTC"), pd.Timestamp("2023-01-16", tz="UTC"),
         "Jan 9–16, 2023", "Clear Dry Season (best-case match)",
         "#E8F5E9", "#2E7D32"),
        (pd.Timestamp("2022-04-11", tz="UTC"), pd.Timestamp("2022-04-18", tz="UTC"),
         "Apr 11–18, 2022", "Inter-monsoon — Patchy Convective Cloud",
         "#FFF8E1", "#E65100"),
        (pd.Timestamp("2022-08-08", tz="UTC"), pd.Timestamp("2022-08-15", tz="UTC"),
         "Aug 8–15, 2022", "SW Monsoon Dry Spell",
         "#E8F5E9", "#2E7D32"),
        (pd.Timestamp("2022-11-07", tz="UTC"), pd.Timestamp("2022-11-14", tz="UTC"),
         "Nov 7–14, 2022", "NE Monsoon Onset — Cloud Miss (worst-case)",
         "#FFEBEE", "#C62828"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(20, 20),
                             gridspec_kw={"width_ratios": [5, 3]})
    fig.suptitle(
        "Real vs Synthetic PV — Weekly Comparison Across Seasons\n"
        "Left: hourly AC power  |  Right: hourly error  (synthetic − real)",
        fontsize=12, fontweight="bold",
    )

    for row, (t0, t1, date_label, regime_label, bg_color, title_color) in enumerate(weeks):
        w_obs = obs_kw.loc[t0:t1]
        w_sim = sim_kw.loc[t0:t1]

        ax_ts  = axes[row, 0]
        ax_err = axes[row, 1]
        ax_ts.set_facecolor(bg_color)
        ax_err.set_facecolor(bg_color)

        ax_ts.plot(w_obs.index, w_obs.values, color="#1565C0", lw=1.3,
                   alpha=0.9, label="Real observed")
        ax_ts.plot(w_sim.index, w_sim.values, color="#E65100", lw=1.3,
                   alpha=0.85, linestyle="--", label="Synthetic pvlib")
        ax_ts.fill_between(w_obs.index,
                           w_obs.values, w_sim.reindex(w_obs.index).values,
                           where=(w_sim.reindex(w_obs.index) > w_obs),
                           alpha=0.18, color="#E65100", label="Over-pred.")
        ax_ts.fill_between(w_obs.index,
                           w_obs.values, w_sim.reindex(w_obs.index).values,
                           where=(w_sim.reindex(w_obs.index) <= w_obs),
                           alpha=0.18, color="#1565C0", label="Under-pred.")

        err_w = (w_sim.reindex(w_obs.index) - w_obs).dropna()
        if len(err_w) > 1:
            rmse_w = np.sqrt((err_w**2).mean())
            mbe_w  = err_w.mean()
            obs_v  = w_obs.reindex(err_w.index)
            obs_var = ((obs_v - obs_v.mean())**2).sum()
            r2_w    = (1 - (err_w**2).sum() / obs_var) if obs_var > 0 else float("nan")
            r2_str  = f"{r2_w:.3f}" if not np.isnan(r2_w) else "n/a"
            metrics_txt = f"RMSE = {rmse_w:.1f} kW\nMBE  = {mbe_w:+.1f} kW\nR²   = {r2_str}"
        else:
            metrics_txt = "n/a"

        ax_ts.text(0.01, 0.97, metrics_txt,
                   transform=ax_ts.transAxes, va="top", fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.90))
        ax_ts.set_title(f"{date_label}  —  {regime_label}",
                        fontsize=10, fontweight="bold", color=title_color)
        ax_ts.set_ylabel("AC Power (kW)", fontsize=9)
        ax_ts.set_ylim(bottom=0)
        ax_ts.tick_params(axis="x", rotation=20, labelsize=8)
        if row == 0:
            ax_ts.legend(fontsize=8, ncol=2, loc="upper right")

        if not err_w.empty:
            bar_colors = ["#E65100" if v > 0 else "#1565C0" for v in err_w.values]
            ax_err.bar(range(len(err_w)), err_w.values,
                       color=bar_colors, alpha=0.65, width=0.8)
            ax_err.axhline(0, color="black", lw=0.8)
            ax_err.axhline(err_w.mean(), color="red", lw=1.2, ls="--",
                           label=f"MBE = {err_w.mean():+.1f} kW")
            ax_err.legend(fontsize=8)
        ax_err.set_ylabel("Error  sim−obs  (kW)", fontsize=9)
        ax_err.set_xlabel("Hour index in week", fontsize=8)
        ax_err.set_title("Hourly Error", fontsize=9, fontweight="bold")
        ax_err.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_weekly_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9 — Selected daily profiles (incl. hardware failure day)
# ─────────────────────────────────────────────────────────────────────────────

def plot_selected_daily_profiles(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    nasa_overlap: pd.DataFrame,
    cfg:          dict,
) -> None:
    """
    Six specific days in a 3×2 grid, each chosen to expose a distinct
    failure mode of the synthetic data generation.

    Each panel:
      - Hourly AC power: real (blue) vs synthetic (orange dashed)
      - Secondary y-axis: NASA GHI as a grey fill
      - Clearness index kt and daily RMSE/MBE annotations
      - Hardware failure and cloud-miss events highlighted

    Days:
      1. Jan 12, 2023 — clear dry day (reference)
      2. Jan 13, 2023 — hardware failure (PV System 1 inverter trip)
      3. Apr 18, 2022 — clear inter-monsoon
      4. Nov 10, 2022 — NE monsoon cloud miss
      5. Aug 15, 2022 — SW monsoon dry spell
      6. Jan 15, 2023 — recovery day after hardware failure
    """
    _PV_OBS_COL_L = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
    if _PV_OBS_COL_L not in local_hourly.columns:
        logger.warning("Real PV column not found — skipping daily profiles plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=0.9)

    obs_kw = local_hourly[_PV_OBS_COL_L] / 1000
    sim_kw = pv_sim["pv_ac_W"] / 1000
    ghi    = (nasa_overlap["ALLSKY_SFC_SW_DWN_cal"]
              if "ALLSKY_SFC_SW_DWN_cal" in nasa_overlap.columns else None)

    days = [
        (pd.Timestamp("2023-01-12", tz="UTC"),
         "Jan 12, 2023 — Clear Dry Day",
         "Reference: expected good match", None, "#E8F5E9"),
        (pd.Timestamp("2023-01-13", tz="UTC"),
         "Jan 13, 2023 — Hardware Failure Day",
         "! PV System 1 inverter trip ~09:00 UTC", "hardware", "#FFEBEE"),
        (pd.Timestamp("2022-04-18", tz="UTC"),
         "Apr 18, 2022 — Clear Inter-monsoon Day",
         "Transition season: moderate agreement", None, "#FFF8E1"),
        (pd.Timestamp("2022-11-10", tz="UTC"),
         "Nov 10, 2022 — NE Monsoon (Cloud Miss)",
         "NASA 0.5° pixel: clear sky — actual site: overcast", "cloud_miss", "#FFEBEE"),
        (pd.Timestamp("2022-08-15", tz="UTC"),
         "Aug 15, 2022 — SW Monsoon Dry Spell",
         "Partial cloud: mix of clear and overcast hours", None, "#E3F2FD"),
        (pd.Timestamp("2023-01-15", tz="UTC"),
         "Jan 15, 2023 — Recovery After Hardware Failure",
         "System back online: good agreement", None, "#E8F5E9"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(
        "Synthetic vs Real PV — Selected Daily Profiles\n"
        "Blue = Real  |  Orange dashed = Synthetic  |  Grey fill = NASA GHI (right axis)",
        fontsize=12, fontweight="bold",
    )

    for idx, (day_ts, title, subtitle, event_type, bg_color) in enumerate(days):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor(bg_color)

        day_end = day_ts + pd.Timedelta(hours=23)
        d_obs = obs_kw.loc[day_ts:day_end]
        d_sim = sim_kw.loc[day_ts:day_end]

        if d_obs.empty:
            ax.set_title(f"{title}\n[no data in overlap window]", fontsize=9)
            continue

        # Secondary axis: NASA GHI
        ax2 = ax.twinx()
        d_ghi = pd.Series(dtype=float)
        if ghi is not None:
            d_ghi = ghi.loc[day_ts:day_end]
            if not d_ghi.empty:
                ax2.fill_between(d_ghi.index, d_ghi.values,
                                 color="grey", alpha=0.12, zorder=1)
                ax2.set_ylabel("GHI  (W/m²)", fontsize=8, color="grey")
                ax2.tick_params(axis="y", labelcolor="grey", labelsize=7.5)
                ax2.set_ylim(bottom=0, top=max(float(d_ghi.max()) * 1.3, 200))

        # Clearness index: daily observed energy / daily synthetic energy
        sim_aligned = d_sim.reindex(d_obs.index)
        if sim_aligned.sum() > 0:
            kt_str = f"kt = {d_obs.sum() / sim_aligned.sum():.2f}"
        else:
            kt_str = ""

        # Main traces
        ax.plot(d_obs.index, d_obs.values, color="#1565C0", lw=1.8,
                marker="o", ms=4, alpha=0.9, label="Real observed", zorder=3)
        ax.plot(d_sim.index, d_sim.values, color="#E65100", lw=1.8,
                linestyle="--", alpha=0.85, label="Synthetic pvlib", zorder=2)

        ax.set_ylim(bottom=0)

        err_d = (sim_aligned - d_obs).dropna()
        if len(err_d) > 1:
            rmse_d = np.sqrt((err_d**2).mean())
            mbe_d  = err_d.mean()
            metrics_txt = f"{kt_str}\nRMSE = {rmse_d:.1f} kW\nMBE  = {mbe_d:+.1f} kW"
        else:
            metrics_txt = kt_str if kt_str else "n/a"

        ax.text(0.02, 0.97, metrics_txt,
                transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.92),
                zorder=5)

        if event_type == "hardware":
            trip_time = pd.Timestamp("2023-01-13 09:00", tz="UTC")
            ax.axvline(trip_time, color="red", lw=1.8, ls="--", alpha=0.8, zorder=4)
            y_top = ax.get_ylim()[1]
            ax.text(trip_time, y_top * 0.75,
                    " Inverter trip\n ~09:00 UTC",
                    fontsize=7.5, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85),
                    zorder=6)
        elif event_type == "cloud_miss":
            ax.text(0.98, 0.97,
                    "NASA pixel: clear\nActual site: overcast",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=7.5, color="#B71C1C", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="red", alpha=0.90),
                    zorder=5)

        ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")
        ax.set_ylabel("AC Power (kW)", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
        if idx in (0, 1):
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = resolve_path(cfg["paths"]["figures"]) / "syn_vs_act_daily_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal accuracy metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_seasonal_accuracy(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    cfg:          dict,
) -> pd.DataFrame:
    """
    Compute synthetic-vs-actual accuracy metrics broken down by season and month.

    Metrics (daytime hours only, obs > 0.5 kW):
      RMSE_kW, MAE_kW, MBE_kW, nRMSE_pct (RMSE / mean_obs × 100), R²

    Groupings:
      Overall, Dry season (Jan–Mar Jun–Sep), Wet season (Apr–May Oct–Dec),
      and all 12 individual months.

    Saves  results/metrics/synthetic_accuracy_by_season.csv
    Prints a formatted summary table.
    """
    _PV_OBS_COL_L = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
    if _PV_OBS_COL_L not in local_hourly.columns:
        logger.warning("Real PV column not found — skipping seasonal accuracy.")
        return pd.DataFrame()

    obs_kw = local_hourly[_PV_OBS_COL_L] / 1000
    sim_kw = pv_sim["pv_ac_W"].reindex(obs_kw.index) / 1000

    df = pd.concat([obs_kw.rename("obs"), sim_kw.rename("sim")], axis=1).dropna()
    df = df[df["obs"] > 0.5]   # daytime only

    def _metrics(subset: pd.DataFrame) -> dict:
        err  = subset["sim"] - subset["obs"]
        obs  = subset["obs"]
        rmse = float(np.sqrt((err**2).mean()))
        mae  = float(np.abs(err).mean())
        mbe  = float(err.mean())
        obs_var = float(((obs - obs.mean())**2).sum())
        r2   = float(1 - (err**2).sum() / obs_var) if obs_var > 0 else float("nan")
        nrmse = rmse / obs.mean() * 100
        return {
            "n_hours":   len(subset),
            "RMSE_kW":   round(rmse, 2),
            "MAE_kW":    round(mae, 2),
            "MBE_kW":    round(mbe, 2),
            "nRMSE_pct": round(nrmse, 2),
            "R2":        round(r2, 4),
        }

    dry_months = {1, 2, 3, 6, 7, 8, 9}
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    rows = {}
    rows["Overall"]     = _metrics(df)
    rows["Dry season"]  = _metrics(df[df.index.month.isin(dry_months)])
    rows["Wet season"]  = _metrics(df[~df.index.month.isin(dry_months)])
    for m in range(1, 13):
        mask = df.index.month == m
        if mask.sum() > 10:
            rows[month_names[m]] = _metrics(df[mask])

    results = pd.DataFrame(rows).T
    results.index.name = "Period"

    # ── Save ──────────────────────────────────────────────────────────────────
    from src.utils.config import resolve_path as _rp
    metrics_dir = _rp(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "synthetic_accuracy_by_season.csv"
    results.to_csv(out_path)
    logger.info(f"Saved → {out_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n── Synthetic PV Accuracy vs Actual  (daytime hours, obs > 0.5 kW) ──")
    print(f"  {'Period':<14}  {'n':>6}  {'RMSE':>8}  {'MAE':>8}  {'MBE':>8}  "
          f"{'nRMSE':>8}  {'R²':>7}")
    print(f"  {'':<14}  {'hrs':>6}  {'(kW)':>8}  {'(kW)':>8}  {'(kW)':>8}  "
          f"{'(%)':>8}  {'':>7}")
    print("  " + "─" * 70)
    for period, row in results.iterrows():
        marker = ""
        if period == "Dry season": marker = "  ← Jan-Mar,Jun-Sep"
        if period == "Wet season": marker = "  ← Apr-May,Oct-Dec"
        print(
            f"  {period:<14}  {int(row['n_hours']):>6}  "
            f"{row['RMSE_kW']:>8.2f}  {row['MAE_kW']:>8.2f}  "
            f"{row['MBE_kW']:>+8.2f}  {row['nRMSE_pct']:>7.1f}%  "
            f"{row['R2']:>7.4f}{marker}"
        )
        if period == "Wet season":
            print("  " + "─" * 70)

    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5-min Solcast comparison plot
# ─────────────────────────────────────────────────────────────────────────────

_PV_OBS_COL_5M = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"


def plot_5min_syn_vs_actual(
    pv_sim:      pd.DataFrame,
    local_5min:  pd.DataFrame,
    cfg:         dict,
) -> None:
    """
    Four-panel comparison of 5-min synthetic PV vs actual for 2022-2023.

    Panel 1 — Full year daily-mean envelope (synthetic vs observed)
    Panel 2 — Hourly means by month (heat map style comparison)
    Panel 3 — Clear day zoom: 5-min resolution, Jan 2023
    Panel 4 — Cloudy day zoom: 5-min resolution, Nov 2022
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    if _PV_OBS_COL_5M not in local_5min.columns:
        logger.warning(f"'{_PV_OBS_COL_5M}' not in local_5min — skipping 5-min plot")
        return

    obs_kw = local_5min[_PV_OBS_COL_5M].clip(lower=0) / 1000
    sim_kw = pv_sim["pv_ac_W"].clip(lower=0) / 1000

    # Align on common index
    common = obs_kw.index.intersection(sim_kw.index)
    obs_kw = obs_kw.loc[common]
    sim_kw = sim_kw.loc[common]

    if len(common) == 0:
        logger.warning("No common timestamps between synthetic and observed — skipping plot")
        return

    logger.info(f"  5-min comparison: {len(common):,} common points  "
                f"({common.min().date()} → {common.max().date()})")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Solcast 5-min Synthetic PV vs Actual  (2022–2023)\n"
        "University of Moratuwa — 250 kWp Rooftop System",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Full-year daily mean (daytime only) ──────────────────────────
    ax = axes[0, 0]
    obs_day = obs_kw.where(obs_kw > 0.5).resample("1D").mean()
    sim_day = sim_kw.where(sim_kw > 0.5).resample("1D").mean()

    ax.fill_between(sim_day.index, 0, sim_day.values,
                    alpha=0.35, color="darkorange", label="Synthetic (daily mean)")
    ax.fill_between(obs_day.index, 0, obs_day.values,
                    alpha=0.35, color="steelblue", label="Actual (daily mean)")
    ax.plot(sim_day.index, sim_day.values, lw=0.9, color="darkorange", alpha=0.8)
    ax.plot(obs_day.index, obs_day.values, lw=0.9, color="steelblue", alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_title("Daily Mean Daytime Power — Full Year", fontsize=10)
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    # ── Panel 2: Monthly diurnal profile comparison ───────────────────────────
    ax = axes[0, 1]
    obs_hr = obs_kw.groupby([obs_kw.index.month, obs_kw.index.hour]).mean().unstack(0)
    sim_hr = sim_kw.groupby([sim_kw.index.month, sim_kw.index.hour]).mean().unstack(0)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    cmap = plt.colormaps["tab20"].resampled(12)

    for m in range(1, 13):
        if m in obs_hr.columns:
            lbl = month_names[m - 1]
            ax.plot(obs_hr.index, obs_hr[m].values,
                    lw=1.0, color=cmap(m - 1), alpha=0.85, label=lbl)
            ax.plot(sim_hr.index, sim_hr[m].values,
                    lw=1.0, color=cmap(m - 1), alpha=0.85, ls="--")

    ax.set_title("Monthly Mean Diurnal Profile\n(solid=actual, dashed=synthetic)",
                 fontsize=10)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    # ── Panel 3: Clear day — Jan 12–13 2023 ──────────────────────────────────
    ax = axes[1, 0]
    t0 = pd.Timestamp("2023-01-12", tz="UTC")
    t1 = pd.Timestamp("2023-01-14", tz="UTC")
    o3 = obs_kw.loc[t0:t1]
    s3 = sim_kw.loc[t0:t1]

    ax.plot(o3.index, o3.values, lw=1.2, color="steelblue",
            alpha=0.9, label="Actual (5-min)")
    ax.plot(s3.index, s3.values, lw=1.2, color="darkorange",
            alpha=0.85, label="Synthetic (5-min)", ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_title("Clear-Sky Period — Jan 12–13 2023 (5-min resolution)", fontsize=10)
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)

    # Compute R² for this window
    _o, _s = o3.dropna(), s3.reindex(o3.index).dropna()
    _cm = _o.index.intersection(_s.index)
    if len(_cm) > 2:
        _ov, _sv = _o.loc[_cm].values, _s.loc[_cm].values
        _ss_res  = ((_ov - _sv) ** 2).sum()
        _ss_tot  = ((_ov - _ov.mean()) ** 2).sum()
        _r2      = 1 - _ss_res / _ss_tot if _ss_tot > 0 else float("nan")
        ax.set_title(
            f"Clear-Sky Period — Jan 12–13 2023  (R²={_r2:.3f})", fontsize=10
        )

    # ── Panel 4: Cloudy/monsoon day — Nov 10–11 2022 ─────────────────────────
    ax = axes[1, 1]
    t0 = pd.Timestamp("2022-11-10", tz="UTC")
    t1 = pd.Timestamp("2022-11-12", tz="UTC")
    o4 = obs_kw.loc[t0:t1]
    s4 = sim_kw.loc[t0:t1]

    ax.plot(o4.index, o4.values, lw=1.2, color="steelblue",
            alpha=0.9, label="Actual (5-min)")
    ax.plot(s4.index, s4.values, lw=1.2, color="darkorange",
            alpha=0.85, label="Synthetic (5-min)", ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_title("NE Monsoon Period — Nov 10–11 2022 (5-min resolution)", fontsize=10)
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)

    _o, _s = o4.dropna(), s4.reindex(o4.index).dropna()
    _cm = _o.index.intersection(_s.index)
    if len(_cm) > 2:
        _ov, _sv = _o.loc[_cm].values, _s.loc[_cm].values
        _ss_res  = ((_ov - _sv) ** 2).sum()
        _ss_tot  = ((_ov - _ov.mean()) ** 2).sum()
        _r2      = 1 - _ss_res / _ss_tot if _ss_tot > 0 else float("nan")
        ax.set_title(
            f"NE Monsoon Period — Nov 10–11 2022  (R²={_r2:.3f})", fontsize=10
        )

    fig.tight_layout()
    out = fig_dir / "syn_vs_act_solcast_5min_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_5min_scatter_metrics(
    pv_sim:     pd.DataFrame,
    local_5min: pd.DataFrame,
    cfg:        dict,
) -> None:
    """
    Scatter + accuracy metrics at 5-min resolution vs actual for 2022-2023.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = resolve_path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if _PV_OBS_COL_5M not in local_5min.columns:
        return

    obs_kw = local_5min[_PV_OBS_COL_5M].clip(lower=0) / 1000
    sim_kw = pv_sim["pv_ac_W"].clip(lower=0) / 1000
    common = obs_kw.index.intersection(sim_kw.index)
    obs_kw = obs_kw.loc[common]
    sim_kw = sim_kw.loc[common]

    # Daytime only (obs > 1 kW to exclude night / inverter-off noise)
    mask   = obs_kw > 1.0
    o, s   = obs_kw[mask].values, sim_kw[mask].values

    rmse  = float(np.sqrt(((o - s) ** 2).mean()))
    mae   = float(np.abs(o - s).mean())
    mbe   = float((s - o).mean())
    nrmse = rmse / float(o.mean()) * 100
    ss_res = ((o - s) ** 2).sum()
    ss_tot = ((o - o.mean()) ** 2).sum()
    r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    logger.info("── 5-min accuracy metrics (daytime, Solcast) ──────────────────")
    logger.info(f"  R²    = {r2:.4f}")
    logger.info(f"  RMSE  = {rmse:.2f} kW   ({nrmse:.1f}% nRMSE)")
    logger.info(f"  MAE   = {mae:.2f} kW")
    logger.info(f"  MBE   = {mbe:+.2f} kW")
    logger.info(f"  n     = {len(o):,} 5-min intervals")

    # Save metrics CSV
    import pandas as _pd_inner
    metrics_df = _pd_inner.DataFrame([{
        "source": "solcast_5min", "resolution": "5min",
        "R2": round(r2, 4), "RMSE_kW": round(rmse, 3),
        "nRMSE_pct": round(nrmse, 2), "MAE_kW": round(mae, 3),
        "MBE_kW": round(mbe, 3), "n": len(o),
    }])
    metrics_df.to_csv(metrics_dir / "solcast_5min_accuracy.csv", index=False)
    logger.info(f"Saved → {metrics_dir / 'solcast_5min_accuracy.csv'}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hexbin(o, s, gridsize=60, cmap="YlOrRd", mincnt=1)
    fig.colorbar(h, ax=ax, label="Count")
    lim = max(o.max(), s.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, alpha=0.6, label="1:1 line")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual PV Power (kW)")
    ax.set_ylabel("Synthetic PV Power (kW)")
    ax.set_title(
        f"Solcast 5-min: Synthetic vs Actual\n"
        f"R²={r2:.4f}  RMSE={rmse:.1f} kW  nRMSE={nrmse:.1f}%  "
        f"MBE={mbe:+.1f} kW  n={len(o):,}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "syn_vs_act_solcast_5min_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_5min_yearly_timeseries(
    pv_sim:      pd.DataFrame,
    local_5min:  pd.DataFrame,
    cfg:         dict,
) -> None:
    """Full 2022-2023 yearly comparison — daily means + weekly rolling band."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    obs_kw = local_5min[_PV_OBS_COL_5M].clip(lower=0) / 1000
    sim_kw = pv_sim["pv_ac_W"].clip(lower=0) / 1000
    common = obs_kw.index.intersection(sim_kw.index)
    obs_kw = obs_kw.loc[common]
    sim_kw = sim_kw.loc[common]

    # Daily means (daytime only)
    obs_day = obs_kw.where(obs_kw > 0.5).resample("1D").mean()
    sim_day = sim_kw.where(sim_kw > 0.5).resample("1D").mean()
    err_day = sim_day - obs_day

    # 7-day rolling mean
    obs_roll = obs_day.rolling(7, center=True, min_periods=3).mean()
    sim_roll = sim_day.rolling(7, center=True, min_periods=3).mean()

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(
        "Solcast 5-min Synthetic PV vs Actual — Full Year 2022–2023\n"
        "University of Moratuwa  |  Monthly calibration + Perez sky model",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Daily mean + 7-day rolling ──────────────────────────────────
    ax = axes[0]
    ax.fill_between(obs_day.index, 0, obs_day.values,
                    alpha=0.25, color="steelblue")
    ax.fill_between(sim_day.index, 0, sim_day.values,
                    alpha=0.25, color="darkorange")
    ax.plot(obs_day.index, obs_day.values, lw=0.6, color="steelblue",
            alpha=0.6, label="Actual — daily mean")
    ax.plot(sim_day.index, sim_day.values, lw=0.6, color="darkorange",
            alpha=0.6, label="Synthetic — daily mean")
    ax.plot(obs_roll.index, obs_roll.values, lw=2.0, color="steelblue",
            label="Actual — 7-day rolling mean")
    ax.plot(sim_roll.index, sim_roll.values, lw=2.0, color="darkorange",
            label="Synthetic — 7-day rolling mean")

    # Shade seasons
    for ax_ in axes:
        for yr in [2022, 2023]:
            ax_.axvspan(pd.Timestamp(f"{yr}-04-01", tz="UTC"),
                        pd.Timestamp(f"{yr}-05-31", tz="UTC"),
                        alpha=0.07, color="teal", label="Wet (Apr–May)" if yr==2022 else "")
            ax_.axvspan(pd.Timestamp(f"{yr}-10-01", tz="UTC"),
                        pd.Timestamp(f"{yr}-12-31", tz="UTC"),
                        alpha=0.07, color="teal")

    ax.set_ylabel("Mean Daytime Power (kW)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_ylim(0)

    # ── Panel 2: Error (sim − obs) ────────────────────────────────────────────
    ax = axes[1]
    err_roll = err_day.rolling(7, center=True, min_periods=3).mean()
    ax.bar(err_day.index, err_day.values, width=1.0,
           color=np.where(err_day.values >= 0, "tomato", "steelblue"),
           alpha=0.5, label="Daily error (sim − obs)")
    ax.plot(err_roll.index, err_roll.values, lw=2.0, color="black",
            label="7-day rolling error")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Error (kW)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out = fig_dir / "syn_vs_act_solcast_yearly_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_5min_weekly_timeseries(
    pv_sim:      pd.DataFrame,
    local_5min:  pd.DataFrame,
    cfg:         dict,
) -> None:
    """Three representative weeks at 5-min resolution: clear, mixed, monsoon."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    obs_kw = local_5min[_PV_OBS_COL_5M].clip(lower=0) / 1000
    sim_kw = pv_sim["pv_ac_W"].clip(lower=0) / 1000
    common = obs_kw.index.intersection(sim_kw.index)
    obs_kw = obs_kw.loc[common]
    sim_kw = sim_kw.loc[common]

    weeks = [
        ("Clear / Dry", pd.Timestamp("2023-01-09", tz="UTC"),
                        pd.Timestamp("2023-01-16", tz="UTC")),
        ("Transition",  pd.Timestamp("2022-09-12", tz="UTC"),
                        pd.Timestamp("2022-09-19", tz="UTC")),
        ("NE Monsoon",  pd.Timestamp("2022-11-07", tz="UTC"),
                        pd.Timestamp("2022-11-14", tz="UTC")),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharey=False)
    fig.suptitle(
        "Solcast 5-min Synthetic vs Actual — Representative Weeks\n"
        "University of Moratuwa  |  Monthly calibration + Perez sky model",
        fontsize=13, fontweight="bold",
    )

    for ax, (label, t0, t1) in zip(axes, weeks):
        o = obs_kw.loc[t0:t1]
        s = sim_kw.loc[t0:t1]

        ax.fill_between(o.index, 0, o.values, alpha=0.3, color="steelblue")
        ax.fill_between(s.index, 0, s.values, alpha=0.3, color="darkorange")
        ax.plot(o.index, o.values, lw=0.8, color="steelblue",  alpha=0.9,
                label="Actual (5-min)")
        ax.plot(s.index, s.values, lw=0.8, color="darkorange", alpha=0.9,
                ls="--", label="Synthetic (5-min)")

        # Compute R² for this window
        _cm = o.index.intersection(s.index)
        _o, _s = o.loc[_cm].values, s.loc[_cm].values
        _day = _o > 1.0
        if _day.sum() > 5:
            ss_res = ((_o[_day] - _s[_day]) ** 2).sum()
            ss_tot = ((_o[_day] - _o[_day].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(((_o[_day] - _s[_day]) ** 2).mean()))
            ax.set_title(f"{label}  |  R²={r2:.3f}  RMSE={rmse:.1f} kW", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.set_ylabel("Power (kW)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0)

    axes[-1].set_xlabel("Date / Time (UTC)", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "syn_vs_act_solcast_weekly_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_5min_daily_timeseries(
    pv_sim:      pd.DataFrame,
    local_5min:  pd.DataFrame,
    cfg:         dict,
) -> None:
    """Six representative days at 5-min resolution across all seasons."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    obs_kw = local_5min[_PV_OBS_COL_5M].clip(lower=0) / 1000
    sim_kw = pv_sim["pv_ac_W"].clip(lower=0) / 1000
    common = obs_kw.index.intersection(sim_kw.index)
    obs_kw = obs_kw.loc[common]
    sim_kw = sim_kw.loc[common]

    days = [
        ("Jan 2023 — Clear dry",       pd.Timestamp("2023-01-12", tz="UTC")),
        ("Mar 2023 — Pre-monsoon",     pd.Timestamp("2023-03-15", tz="UTC")),
        ("May 2022 — SW Monsoon",      pd.Timestamp("2022-05-20", tz="UTC")),
        ("Jul 2022 — Dry inter-monsoon", pd.Timestamp("2022-07-14", tz="UTC")),
        ("Oct 2022 — NE Monsoon onset", pd.Timestamp("2022-10-18", tz="UTC")),
        ("Dec 2022 — Cool dry",        pd.Timestamp("2022-12-08", tz="UTC")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Solcast 5-min Synthetic vs Actual — Representative Days\n"
        "University of Moratuwa  |  Monthly calibration + Perez sky model",
        fontsize=13, fontweight="bold",
    )

    for ax, (label, t0) in zip(axes.flat, days):
        t1 = t0 + pd.Timedelta(days=1)
        o  = obs_kw.loc[t0:t1]
        s  = sim_kw.loc[t0:t1]

        ax.fill_between(o.index, 0, o.values, alpha=0.3, color="steelblue")
        ax.fill_between(s.index, 0, s.values, alpha=0.3, color="darkorange")
        ax.plot(o.index, o.values, lw=1.2, color="steelblue",  alpha=0.9,
                label="Actual")
        ax.plot(s.index, s.values, lw=1.2, color="darkorange", alpha=0.9,
                ls="--", label="Synthetic")

        _cm = o.index.intersection(s.index)
        _o, _s = o.loc[_cm].values, s.loc[_cm].values
        _day = _o > 1.0
        if _day.sum() > 5:
            ss_res = ((_o[_day] - _s[_day]) ** 2).sum()
            ss_tot = ((_o[_day] - _o[_day].mean()) ** 2).sum()
            r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(((_o[_day] - _s[_day]) ** 2).mean()))
            ax.set_title(f"{label}\nR²={r2:.3f}  RMSE={rmse:.1f} kW", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.set_ylabel("Power (kW)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0)

    fig.tight_layout()
    out = fig_dir / "syn_vs_act_solcast_daily_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SOURCES = ("nasa", "era5", "nsrdb", "solcast", "solargis")


def _load_irradiance(source: str, cfg: dict) -> pd.DataFrame:
    """Load the irradiance DataFrame for the requested source."""
    if source == "nasa":
        df = pd.read_csv(
            resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv",
            index_col="timestamp_utc", parse_dates=True,
        )
        df.index = pd.to_datetime(df.index, utc=True)
        last_irr = df["ALLSKY_SFC_SW_DWN_cal"].last_valid_index()
        df = df.loc[:last_irr]
        logger.info(f"NASA POWER: {len(df):,} rows  (up to {last_irr})")

    elif source == "era5":
        from src.data.era5_loader import load_era5_processed
        df = load_era5_processed(cfg)

    elif source == "nsrdb":
        from src.data.nsrdb_loader import load_nsrdb_processed
        df = load_nsrdb_processed(cfg)

    elif source == "solcast":
        from src.data.solcast_loader import (
            load_solcast_local_files, solcast_to_nasa_schema,
        )
        raw = load_solcast_local_files(cfg)
        df  = solcast_to_nasa_schema(raw)

    elif source == "solargis":
        from src.data.solargis_loader import load_solargis_processed
        df = load_solargis_processed(cfg)

    else:
        raise ValueError(f"Unknown source '{source}'. Choose from: {_VALID_SOURCES}")

    return df


def _run_solcast_5min(cfg: dict, prefix: str) -> None:
    """
    Full 5-minute resolution pipeline for Solcast irradiance data.

    Steps
    -----
    1. Load local Solcast CSV files → 5-min irradiance in NASA schema
    2. Load raw local PV data → 5-min UTC
    3. Align both at 5-min resolution (no resampling)
    4. Calibrate polynomial on overlap window
    5. Simulate full Solcast period
    6. Save synthetic CSV + plots
    """
    import copy
    from src.data.solcast_loader import load_solcast_local_files, solcast_to_nasa_schema
    from src.preproccesing.align import align_solcast_5min, save_aligned_5min
    from src.utils.config import load_config as _lc

    # ── Load Solcast irradiance ───────────────────────────────────────────────
    logger.info("Loading Solcast local CSV files …")
    raw_solcast = load_solcast_local_files(cfg)
    solcast_cal = solcast_to_nasa_schema(raw_solcast)
    logger.info(
        f"  Solcast range: {solcast_cal.index.min().date()} → "
        f"{solcast_cal.index.max().date()}  ({len(solcast_cal):,} rows)"
    )

    # ── Load raw local PV data (5-min, Asia/Colombo) ─────────────────────────
    logger.info("Loading raw local PV data (5-min) …")
    raw_dir   = resolve_path(cfg["paths"]["raw_local"])
    raw_files = sorted(raw_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    raw_path  = raw_files[0]
    logger.info(f"  Reading {raw_path.name}")
    local_raw = pd.read_csv(raw_path, index_col="datetime", parse_dates=True)
    # Localize to Asia/Colombo (UTC+5:30)
    if local_raw.index.tz is None:
        local_raw.index = local_raw.index.tz_localize("Asia/Colombo", ambiguous="NaT",
                                                        nonexistent="NaT")
    local_raw.index.name = "timestamp_utc"
    logger.info(
        f"  Local raw: {len(local_raw):,} rows  "
        f"({local_raw.index.min()} → {local_raw.index.max()})"
    )

    # ── Align at 5-min resolution ─────────────────────────────────────────────
    logger.info("Aligning local 5-min data with Solcast 5-min data …")
    local_5min, solcast_overlap = align_solcast_5min(local_raw, solcast_cal)
    save_aligned_5min(local_5min, solcast_overlap, cfg)

    # ── Calibration overlap (use the aligned Solcast data) ────────────────────
    t_start = local_5min.index.min()
    t_end   = local_5min.index.max()
    logger.info(f"Calibration overlap: {t_start} → {t_end}  ({len(solcast_overlap):,} rows)")

    # ── Calibrate ─────────────────────────────────────────────────────────────
    poly_coeffs     = calibrate_polynomial(solcast_overlap, local_5min)
    seasonal_coeffs = calibrate_seasonal(solcast_overlap, local_5min)
    monthly_coeffs  = calibrate_monthly(solcast_overlap, local_5min,
                                        seasonal_coeffs=seasonal_coeffs)
    sky_stratified_coeffs = calibrate_sky_stratified(
        solcast_overlap, local_5min,
        monthly_coeffs=monthly_coeffs,
        seasonal_coeffs=seasonal_coeffs,
    )
    a, b = poly_coeffs

    # ── Simulate over full Solcast period ─────────────────────────────────────
    pv_sim = simulate_pv(solcast_cal, poly_coeffs,
                         seasonal_coeffs=seasonal_coeffs,
                         monthly_coeffs=monthly_coeffs,
                         sky_stratified_coeffs=sky_stratified_coeffs)

    # ── Save synthetic CSV ────────────────────────────────────────────────────
    synth_dir  = resolve_path(cfg["paths"]["synthetic"])
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_path = synth_dir / f"{prefix}pv_synthetic_5min.csv"
    pv_sim.to_csv(synth_path)
    logger.info(f"Saved synthetic → {synth_path}  ({synth_path.stat().st_size/1024:.0f} KB)")

    # ── Source-specific output directories ───────────────────────────────────
    source_fig_dir     = resolve_path(cfg["paths"]["figures"]) / "solcast"
    source_metrics_dir = resolve_path(cfg["paths"]["metrics"]) / "solcast"
    source_fig_dir.mkdir(parents=True, exist_ok=True)
    source_metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg_src = copy.deepcopy(cfg)
    cfg_src["paths"]["figures"] = str(source_fig_dir.relative_to(
        Path(__file__).resolve().parents[1]
    ))
    cfg_src["paths"]["metrics"] = str(source_metrics_dir.relative_to(
        Path(__file__).resolve().parents[1]
    ))

    # ── Plots — synthetic overview ────────────────────────────────────────────
    logger.info("Generating plots …")
    plot_6yr_timeseries(pv_sim, cfg_src)
    plot_annual_energy(pv_sim, cfg_src)
    plot_monthly_hour_heatmap(pv_sim, cfg_src)

    # ── Plots — 5-min synthetic vs actual ────────────────────────────────────
    plot_5min_yearly_timeseries(pv_sim, local_5min, cfg_src)
    plot_5min_weekly_timeseries(pv_sim, local_5min, cfg_src)
    plot_5min_daily_timeseries(pv_sim, local_5min, cfg_src)
    plot_5min_syn_vs_actual(pv_sim, local_5min, cfg_src)
    plot_5min_scatter_metrics(pv_sim, local_5min, cfg_src)

    # ── Seasonal accuracy at 5-min ────────────────────────────────────────────
    compute_seasonal_accuracy(pv_sim, local_5min, cfg_src)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── Synthetic PV (first 3 daytime rows) ──────────────────────────────")
    first_day = pv_sim[pv_sim["pv_ac_W"] > 0].head(3)
    print(first_day.to_string())

    print("\n── Annual summary (MWh) ─────────────────────────────────────────────")
    annual = (pv_sim.groupby(pv_sim.index.year)["pv_ac_W"].sum() / 1e6).round(2)
    for yr, mwh in annual.items():
        print(f"  {yr}: {mwh:.2f} MWh")

    print(f"\n  Source     : SOLCAST (5-min)")
    print(f"  Calibration: P_ac = {a:.2f}·sim + {b:.5f}·sim²")
    print(f"  Shape      : {pv_sim.shape}")
    print(f"  Figures    : results/figures/solcast/")
    print(f"  Metrics    : results/metrics/solcast/")
    print(f"  Synthetic  : {synth_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate 6-year PV power using pvlib PVWatts model."
    )
    parser.add_argument("--config", default="configs/site.yaml")
    parser.add_argument(
        "--source", default="nasa",
        choices=_VALID_SOURCES,
        help=(
            "Irradiance source to use for simulation. "
            "nasa (default) | era5 | nsrdb | solcast | solargis. "
            "Each source must have been fetched/processed first."
        ),
    )
    # Keep --era5 as a backwards-compatible alias
    parser.add_argument("--era5", action="store_true",
                        help="Alias for --source era5 (backwards compat).")
    args = parser.parse_args()

    if args.era5:
        args.source = "era5"

    cfg    = load_config(args.config)
    source = args.source
    prefix = f"{source}_"

    logger.info(f"Site  : {cfg['site']['name']}")
    logger.info(f"Source: {source.upper()}")

    # ── Solcast uses a dedicated 5-min pipeline ───────────────────────────────
    if source == "solcast":
        _run_solcast_5min(cfg, prefix)
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading aligned local data …")
    local_hourly, _ = load_aligned(cfg)

    logger.info(f"Loading {source.upper()} irradiance data …")
    nasa_cal = _load_irradiance(source, cfg)
    logger.info(f"  Range: {nasa_cal.index.min()} → {nasa_cal.index.max()}  "
                f"({len(nasa_cal):,} rows)")

    # ── Clip to overlap window for calibration ────────────────────────────────
    t_start = local_hourly.index.min()
    t_end   = local_hourly.index.max()
    nasa_overlap = nasa_cal.loc[t_start:t_end]
    logger.info(f"Calibration overlap: {t_start} → {t_end}  ({len(nasa_overlap):,} rows)")

    # ── Fall back to NASA POWER calibration if no overlap ────────────────────
    nasa_overlap_for_cal = nasa_overlap
    if len(nasa_overlap) == 0:
        logger.warning(
            f"No temporal overlap between {source.upper()} data "
            f"({nasa_cal.index.min().date()} – {nasa_cal.index.max().date()}) "
            f"and local PV observations ({t_start.date()} – {t_end.date()}).\n"
            "  Falling back to NASA POWER calibration coefficients."
        )
        nasa_cal_path = resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv"
        if not nasa_cal_path.exists():
            raise FileNotFoundError(
                f"NASA calibrated data not found at {nasa_cal_path}.\n"
                "Run  python scripts/run_calibration.py  first."
            )
        _nasa_fb = pd.read_csv(nasa_cal_path, index_col="timestamp_utc", parse_dates=True)
        _nasa_fb.index = pd.to_datetime(_nasa_fb.index, utc=True)
        nasa_overlap_for_cal = _nasa_fb.loc[t_start:t_end]
        logger.info(
            f"  NASA fallback overlap: {len(nasa_overlap_for_cal):,} rows  "
            f"({nasa_overlap_for_cal.index.min()} → {nasa_overlap_for_cal.index.max()})"
        )

    # ── Step 1: Calibrate ─────────────────────────────────────────────────────
    poly_coeffs     = calibrate_polynomial(nasa_overlap_for_cal, local_hourly)
    seasonal_coeffs = calibrate_seasonal(nasa_overlap_for_cal, local_hourly)
    monthly_coeffs  = calibrate_monthly(nasa_overlap_for_cal, local_hourly,
                                        seasonal_coeffs=seasonal_coeffs)
    a, b = poly_coeffs

    # ── Step 2: Simulate ──────────────────────────────────────────────────────
    pv_sim = simulate_pv(nasa_cal, poly_coeffs,
                         seasonal_coeffs=seasonal_coeffs,
                         monthly_coeffs=monthly_coeffs)

    # ── Step 3: Save synthetic CSV (source-prefixed) ──────────────────────────
    synth_dir  = resolve_path(cfg["paths"]["synthetic"])
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_path = synth_dir / f"{prefix}pv_synthetic_6yr.csv"
    pv_sim.to_csv(synth_path)
    logger.info(f"Saved synthetic → {synth_path}  ({synth_path.stat().st_size/1024:.0f} KB)")

    # ── Step 4: Plots — each source gets its own subdirectory ────────────────
    # cfg["paths"]["figures"] → results/figures/<source>/
    # cfg["paths"]["metrics"] → results/metrics/<source>/
    # No changes to any plot function needed.
    import copy
    source_fig_dir     = resolve_path(cfg["paths"]["figures"]) / source
    source_metrics_dir = resolve_path(cfg["paths"]["metrics"]) / source
    source_fig_dir.mkdir(parents=True, exist_ok=True)
    source_metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg_src = copy.deepcopy(cfg)
    cfg_src["paths"]["figures"] = str(source_fig_dir.relative_to(
        Path(__file__).resolve().parents[1]
    ))
    cfg_src["paths"]["metrics"] = str(source_metrics_dir.relative_to(
        Path(__file__).resolve().parents[1]
    ))

    logger.info("Generating plots …")

    # Synthetic-only plots (always generated regardless of obs overlap)
    plot_6yr_timeseries(pv_sim, cfg_src)
    plot_annual_energy(pv_sim, cfg_src)
    plot_monthly_hour_heatmap(pv_sim, cfg_src)

    # Obs-vs-synthetic comparison plots require temporal overlap
    _sim_obs_overlap = pv_sim.index.intersection(local_hourly.index)
    _has_obs_overlap = len(_sim_obs_overlap) > 0

    if _has_obs_overlap:
        # For plot_validation we need the calibration overlap window data.
        # If the source had no overlap (e.g. NSRDB 2017-2019 vs obs 2022-2023),
        # nasa_overlap_for_cal contains the NASA fallback data for the same window.
        plot_validation(pv_sim, local_hourly, nasa_overlap_for_cal, poly_coeffs, cfg_src)
        plot_validation_timeseries(pv_sim, local_hourly, cfg_src)
        plot_real_vs_synthetic(pv_sim, local_hourly, cfg_src)
        plot_yearly_monthly_breakdown(pv_sim, local_hourly, cfg_src)
        plot_weekly_seasons(pv_sim, local_hourly, cfg_src)
        # For daily profiles, pass source GHI if available in the obs window,
        # otherwise fall back to the calibration data used above.
        _daily_ghi_src = nasa_overlap if len(nasa_overlap) > 0 else nasa_overlap_for_cal
        plot_selected_daily_profiles(pv_sim, local_hourly, _daily_ghi_src, cfg_src)
    else:
        logger.warning(
            f"Synthetic data ({pv_sim.index.min().date()} – {pv_sim.index.max().date()}) "
            f"has no overlap with local observations ({t_start.date()} – {t_end.date()}).\n"
            "  Skipping obs-vs-synthetic comparison plots."
        )

    # ── Step 5: Seasonal accuracy ─────────────────────────────────────────────
    if _has_obs_overlap:
        compute_seasonal_accuracy(pv_sim, local_hourly, cfg_src)
    else:
        logger.warning("Skipping seasonal accuracy metrics (no obs overlap).")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── Synthetic PV (first 3 daytime rows) ──────────────────────────────")
    first_day = pv_sim[pv_sim["pv_ac_W"] > 0].head(3)
    print(first_day.to_string())

    print("\n── Annual summary (MWh) ─────────────────────────────────────────────")
    annual = (pv_sim.groupby(pv_sim.index.year)["pv_ac_W"].sum() / 1e6).round(2)
    for yr, mwh in annual.items():
        print(f"  {yr}: {mwh:.2f} MWh")

    print(f"\n  Source     : {source.upper()}")
    print(f"  Calibration: P_ac = {a:.2f}·sim + {b:.5f}·sim²")
    print(f"  Shape      : {pv_sim.shape}")
    print(f"  Figures    : results/figures/{source}/")
    print(f"  Metrics    : results/metrics/{source}/")
    print(f"  Synthetic  : {synth_path}")


if __name__ == "__main__":
    main()
