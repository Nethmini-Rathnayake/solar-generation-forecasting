"""
scripts/run_calibration.py
--------------------------
Calibrates the full 6-year NASA POWER dataset using locally measured
PV / weather data as ground-truth during the 1-year overlap window.

Calibrations applied
---------------------
  Met. variables (T2M, RH2M, WS10M):
      Linear bias correction fitted on the overlap period.
      T2M and RH2M use bias-only (R² too low for slope); WS10M uses
      full linear regression.

  Irradiance (GHI, DNI, DHI, clear-sky GHI):
      Monthly correction factors derived from PV power as a proxy.
      Factors correct for seasonal cloud-cover bias in NASA POWER.

Output files
------------
  data/processed/nasa_calibrated.csv
  results/figures/data_pre_calibration_met_scatter.png
  results/figures/data_pre_calibration_ghi_pv.png
  results/figures/data_pre_calibration_monthly_factors.png
  results/figures/data_pre_calibration_before_after.png

Run from project root:
    python scripts/run_calibration.py

Optional flags:
    --config   configs/site.yaml
    --local    data/interim/local_hourly_utc.csv
    --nasa     data/interim/nasa_aligned.csv
    --nasa-full  <filename within data/external/>
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
from src.calibration.bias_correction import fit_met_corrections, _MET_PAIRS
from src.calibration.regression      import (
    fit_ghi_calibration,
    GHI_MIN_DAYTIME, PV_MIN_DAYTIME, PV_POWER_COL,
)
from src.calibration.apply import calibrate_nasa

logger = get_logger("run_calibration")

_N_SCATTER  = 200   # points shown in scatter plots (regression fitted on ALL data)
_MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(df: pd.DataFrame, n: int = _N_SCATTER, seed: int = 42) -> pd.DataFrame:
    """Return up to n rows sampled uniformly at random (reproducible)."""
    return df.sample(min(n, len(df)), random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Meteorological scatter (3 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_met_scatter(local_hourly, nasa_aligned, corrections, cfg):
    """
    3-panel scatter: NASA vs local for T2M, RH2M, WS10M.

    Each panel shows:
      • N_SCATTER random points (scatter)
      • Regression line over the full observed range (red)
      • 1:1 ideal line (black dashed)
      • Correction equation in both fitted and applied forms
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Meteorological Bias Correction — NASA POWER vs Local Measurements\n"
        f"Scatter: {_N_SCATTER} random samples  |  Regression fitted on all overlap data",
        fontsize=11, fontweight="bold",
    )

    panel_specs = [
        ("tempC",         "T2M",   1.0, "Temperature",      "°C",   "steelblue"),
        ("humidity",      "RH2M",  1.0, "Relative Humidity", "%",   "seagreen"),
        ("windspeedKmph", "WS10M", 3.6, "Wind Speed",       "km/h", "darkorange"),
    ]

    for ax, (local_col, nasa_col, scale, title, unit, color) in zip(axes, panel_specs):
        if nasa_col not in corrections:
            ax.text(0.5, 0.5, f"No correction fitted\nfor {nasa_col}",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(title)
            continue

        corr = corrections[nasa_col]

        # ── Build aligned pair in local units ──────────────────────────────
        x_all = (nasa_aligned[nasa_col] * scale).rename("x")
        y_all = local_hourly[local_col].rename("y")
        df_pair = pd.concat([x_all, y_all], axis=1).dropna()

        sub = _subsample(df_pair)
        ax.scatter(sub["x"], sub["y"],
                   alpha=0.45, s=20, color=color, edgecolors="none",
                   label=f"Observed ({_N_SCATTER} pts)")

        # ── Regression line (over full data range) ─────────────────────────
        x_line = np.linspace(df_pair["x"].min(), df_pair["x"].max(), 300)
        y_line = corr.slope * x_line + corr.intercept
        ax.plot(x_line, y_line, color="crimson", linewidth=2.2, label="Correction fit")

        # ── 1:1 reference line ──────────────────────────────────────────────
        lo = min(df_pair["x"].min(), df_pair["y"].min())
        hi = max(df_pair["x"].max(), df_pair["y"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.55, label="1:1 line")

        # ── Equation box ───────────────────────────────────────────────────
        # Line 1: fitted regression  (local units)
        s, b     = corr.slope, corr.intercept
        sign_fit = "+" if b >= 0 else "−"
        # Line 2: correction applied to NASA (NASA units)
        eff_s, eff_b  = corr.correction_in_nasa_units()
        sign_app      = "+" if eff_b >= 0 else "−"
        bias_tag      = "\n[bias-only: slope forced to 1]" if corr.bias_only else ""

        eq_text = (
            f"Fitted ({unit} vs {unit}):\n"
            f"  y = {s:.3f}x {sign_fit} {abs(b):.2f}\n"
            f"  R² = {corr.r2:.3f}   MAE = {corr.mae:.2f} {unit}\n\n"
            f"Applied to full dataset:\n"
            f"  {nasa_col}_cal = {eff_s:.3f}·{nasa_col} {sign_app} {abs(eff_b):.3f}"
            f"{bias_tag}"
        )
        ax.text(0.03, 0.97, eq_text, transform=ax.transAxes,
                va="top", ha="left", fontsize=7.8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                          edgecolor="0.75", alpha=0.92))

        ax.set_xlabel(f"NASA {nasa_col} ({unit})", fontsize=10)
        ax.set_ylabel(f"Local {local_col} ({unit})", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = resolve_path(cfg["paths"]["figures"]) / "data_pre_calibration_met_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — GHI proxy regression scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_ghi_pv_scatter(local_hourly, nasa_aligned, ghi_cal, cfg):
    """
    Scatter of measured PV power vs NASA GHI (daytime only, N_SCATTER pts).

    Shows the empirical PV response function used to derive GHI calibration.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(7, 6))

    ghi_col = "ALLSKY_SFC_SW_DWN"
    df = pd.concat(
        [local_hourly[PV_POWER_COL].rename("pv"),
         nasa_aligned[ghi_col].rename("ghi")],
        axis=1,
    ).dropna()
    df_day = df.loc[(df["ghi"] >= GHI_MIN_DAYTIME) & (df["pv"] >= PV_MIN_DAYTIME)]

    # Scatter — subsample for clarity
    sub = _subsample(df_day)
    ax.scatter(sub["ghi"], sub["pv"] / 1e3,
               alpha=0.50, s=22, color="darkorange", edgecolors="none",
               label=f"Daytime hours ({_N_SCATTER} of {len(df_day):,} pts)")

    # Regression line over full range
    x_line = np.linspace(df_day["ghi"].min(), df_day["ghi"].max(), 400)
    y_line = (ghi_cal.slope * x_line + ghi_cal.intercept) / 1e3   # W → kW
    ax.plot(x_line, y_line, color="navy", linewidth=2.2,
            label="Regression fit")

    # Equation box
    slope_kw     = ghi_cal.slope / 1e3
    intercept_kw = ghi_cal.intercept / 1e3
    sign         = "+" if intercept_kw >= 0 else "−"
    implied_kw   = (ghi_cal.slope * 1000 + ghi_cal.intercept) / 1e3

    eq_text = (
        f"PV = {slope_kw:.4f} × GHI {sign} {abs(intercept_kw):.3f}\n"
        f"[kW]   [kW/(W/m²)]   [W/m²]   [kW]\n\n"
        f"R² = {ghi_cal.r2:.3f}   n = {_N_SCATTER} / {ghi_cal.n_daytime:,}\n\n"
        f"Implied system capacity\n"
        f"  at GHI = 1000 W/m²: {implied_kw:.1f} kW\n"
        f"  (site.yaml lists 10 kW — incorrect)"
    )
    ax.text(0.03, 0.97, eq_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="0.75", alpha=0.92))

    ax.set_xlabel("NASA GHI  (W/m²)", fontsize=11)
    ax.set_ylabel("Measured PV Power  (kW)", fontsize=11)
    ax.set_title(
        "GHI Proxy Calibration — PV Power vs NASA GHI (Daytime)\n"
        "Regression slope gives empirical system response function",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "data_pre_calibration_ghi_pv.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Monthly GHI correction factors
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_factors(ghi_cal, cfg):
    """
    Bar chart: monthly GHI correction factors derived from PV proxy.

    Blue (> 1): NASA underestimates effective irradiance that month.
    Red  (< 1): NASA overestimates.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(10, 5))

    months  = sorted(ghi_cal.monthly_factors.keys())
    factors = [ghi_cal.monthly_factors[m] for m in months]
    labels  = [_MONTH_ABBR[m - 1] for m in months]
    colors  = ["steelblue" if f >= 1.0 else "tomato" for f in factors]

    bars = ax.bar(labels, factors, color=colors, edgecolor="white", width=0.65)

    # Value labels above each bar
    for bar, f in zip(bars, factors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{f:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(1.0, color="black", linewidth=1.3, linestyle="--", alpha=0.65,
               label="No correction  (factor = 1.000)")

    ax.set_ylim(0, max(factors) * 1.18)
    ax.set_ylabel("GHI Correction Factor", fontsize=11)
    ax.set_title(
        "Monthly GHI Correction Factors  (derived from PV power proxy)\n"
        "GHI_cal = GHI_nasa × factor_M   —   applied to all 6 years",
        fontsize=10, fontweight="bold",
    )

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            bars[0],                          # placeholder for the first bar
            Patch(facecolor="steelblue", label="factor > 1 : NASA underestimates GHI"),
            Patch(facecolor="tomato",    label="factor < 1 : NASA overestimates GHI"),
            plt.Line2D([0], [0], color="black", linestyle="--",
                       label="No correction (1.000)"),
        ],
        fontsize=8.5, loc="upper right",
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    out = resolve_path(cfg["paths"]["figures"]) / "data_pre_calibration_monthly_factors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Before vs After calibration (time series)
# ─────────────────────────────────────────────────────────────────────────────

def plot_before_after(nasa_calibrated, overlap_start, cfg):
    """
    2-panel time series: raw vs calibrated for T2M and GHI.

    Shows the first 30 days starting from the overlap window so the
    visual is anchored in the period used for calibration fitting.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    window_end = overlap_start + pd.Timedelta(days=30)
    win = nasa_calibrated.loc[overlap_start:window_end]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Before vs After Calibration — First 30 Days of Overlap Window",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: Temperature
    ax = axes[0]
    if "T2M" in win.columns and "T2M_cal" in win.columns:
        ax.plot(win.index, win["T2M"],
                lw=0.8, color="navy", alpha=0.80, label="T2M  (raw NASA)")
        ax.plot(win.index, win["T2M_cal"],
                lw=0.8, color="tomato", alpha=0.90, label="T2M_cal  (corrected)")
        offset = win["T2M_cal"].mean() - win["T2M"].mean()
        ax.text(0.01, 0.95,
                f"Mean offset applied: {offset:+.2f} °C",
                transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="0.7", alpha=0.9))
        ax.set_ylabel("Temperature (°C)", fontsize=10)
        ax.set_title("Air Temperature at 2 m  (T2M)", fontsize=10)
        ax.legend(fontsize=9)

    # Panel 2: GHI
    ax = axes[1]
    ghi_col = "ALLSKY_SFC_SW_DWN"
    if ghi_col in win.columns and f"{ghi_col}_cal" in win.columns:
        ax.plot(win.index, win[ghi_col],
                lw=0.8, color="goldenrod", alpha=0.80, label="GHI  (raw NASA)")
        ax.plot(win.index, win[f"{ghi_col}_cal"],
                lw=0.8, color="darkorange", alpha=0.90, label="GHI_cal  (corrected)")
        delta = (win[f"{ghi_col}_cal"] - win[ghi_col]).mean()
        ax.text(0.01, 0.95,
                f"Mean Δ (this window): {delta:+.2f} W/m²",
                transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="0.7", alpha=0.9))
        ax.set_ylabel("GHI  (W/m²)", fontsize=10)
        ax.set_title("Global Horizontal Irradiance  (ALLSKY_SFC_SW_DWN)", fontsize=10)
        ax.legend(fontsize=9)

    axes[-1].tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "data_pre_calibration_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate full NASA POWER dataset using local PV measurements."
    )
    parser.add_argument("--config",    default="configs/site.yaml")
    parser.add_argument("--local",     default="data/interim/local_hourly_utc.csv",
                        help="Aligned local hourly CSV (output of run_align.py)")
    parser.add_argument("--nasa",      default="data/interim/nasa_aligned.csv",
                        help="Aligned NASA CSV clipped to overlap window")
    parser.add_argument("--nasa-full", default=None, dest="nasa_full",
                        help="Filename within data/external/ for the full 6-year NASA CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Site : {cfg['site']['name']}")

    # ── Load aligned data ─────────────────────────────────────────────────────
    logger.info("Loading aligned data …")
    local_hourly = pd.read_csv(args.local,  index_col="timestamp_utc", parse_dates=True)
    nasa_aligned = pd.read_csv(args.nasa,   index_col="timestamp_utc", parse_dates=True)
    local_hourly.index = pd.to_datetime(local_hourly.index, utc=True)
    nasa_aligned.index = pd.to_datetime(nasa_aligned.index, utc=True)

    logger.info(f"  Local  : {len(local_hourly):,} rows  "
                f"({local_hourly.index.min()} → {local_hourly.index.max()})")
    logger.info(f"  NASA   : {len(nasa_aligned):,} rows  "
                f"({nasa_aligned.index.min()} → {nasa_aligned.index.max()})")

    overlap_start = local_hourly.index.min()

    # ── Fit corrections ───────────────────────────────────────────────────────
    corrections = fit_met_corrections(local_hourly, nasa_aligned)
    ghi_cal     = fit_ghi_calibration(local_hourly, nasa_aligned)

    # ── Apply to full 6-year NASA dataset ─────────────────────────────────────
    nasa_calibrated = calibrate_nasa(cfg, corrections, ghi_cal,
                                     nasa_filename=args.nasa_full)

    # ── Generate plots ────────────────────────────────────────────────────────
    logger.info("Generating calibration plots …")
    resolve_path(cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)

    plot_met_scatter(local_hourly, nasa_aligned, corrections, cfg)
    plot_ghi_pv_scatter(local_hourly, nasa_aligned, ghi_cal, cfg)
    plot_monthly_factors(ghi_cal, cfg)
    plot_before_after(nasa_calibrated, overlap_start, cfg)

    # ── Preview ───────────────────────────────────────────────────────────────
    cal_cols = [c for c in nasa_calibrated.columns if c.endswith("_cal")]
    print("\n── Calibrated NASA (first 3 rows, _cal columns) ─────────────────────")
    print(nasa_calibrated[cal_cols].head(3).to_string())
    print(f"\nShape  : {nasa_calibrated.shape}")
    print(f"Range  : {nasa_calibrated.index.min()}  →  {nasa_calibrated.index.max()}")

    print("\n── Monthly GHI correction factors ───────────────────────────────────")
    for m in sorted(ghi_cal.monthly_factors):
        f = ghi_cal.monthly_factors[m]
        direction = "↑ scale up" if f > 1.0 else ("↓ scale down" if f < 1.0 else "→ no change")
        print(f"  {_MONTH_ABBR[m-1]:>3}: {f:.4f}  {direction}")


if __name__ == "__main__":
    main()
