"""
scripts/fetch_era5.py
----------------------
Download, process, and save ERA5 reanalysis data for the University of
Moratuwa site (2020–2025).  ERA5 replaces NASA POWER as the irradiance
driver for the PV simulation, offering 0.25° resolution vs NASA's 0.5°.

Prerequisites (one-time setup)
-------------------------------
1. Register at https://cds.climate.copernicus.eu
2. Accept the ERA5 single-levels dataset licence on the CDS website
3. Create  ~/.cdsapirc  containing:
       url: https://cds.climate.copernicus.eu/api
       key: <your-uid>:<your-api-key>
   (copy key from your CDS profile page — top-right → API key)
4. pip install cdsapi xarray netcdf4

Run from project root:
    python scripts/fetch_era5.py

Optional flags:
    --start 2020     first year to download (inclusive)
    --end   2025     last year to download (inclusive)
    --config configs/site.yaml
    --skip-download  use already-downloaded NetCDF files (load + process only)
    --skip-plots     skip comparison plots (faster)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.data.era5_loader import (
    download_era5,
    load_era5,
    era5_to_nasa_schema,
    save_era5_processed,
)

logger = get_logger("fetch_era5")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots (ERA5 vs NASA POWER)
# ─────────────────────────────────────────────────────────────────────────────

def plot_era5_vs_nasa(era5_df: pd.DataFrame, nasa_df: pd.DataFrame, cfg: dict) -> None:
    """
    Side-by-side comparison of ERA5 and NASA POWER over the full overlap.

    Figure 1 — Monthly mean GHI comparison (bar chart)
    Figure 2 — Scatter: ERA5 GHI vs NASA GHI  (hourly, daytime only)
    Figure 3 — Temperature and RH comparison (2×1 subplots, monthly mean)
    Figure 4 — 14-day time series zoom  (ERA5 vs NASA GHI, Jan 2022)

    These plots show:
    - How closely the two datasets agree on average (monthly means)
    - Whether ERA5 resolves more day-to-day variability (scatter spread)
    - Any systematic bias between the datasets
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Find common UTC timestamps ────────────────────────────────────────────
    era5_ghi_col = "ALLSKY_SFC_SW_DWN_cal"
    nasa_ghi_col = "ALLSKY_SFC_SW_DWN_cal"
    era5_t2m_col = "T2M_cal"
    nasa_t2m_col = "T2M_cal"
    era5_rh_col  = "RH2M_cal"
    nasa_rh_col  = "RH2M_cal"

    # Align on common index
    common_idx = era5_df.index.intersection(nasa_df.index)
    if len(common_idx) == 0:
        logger.warning("No overlapping timestamps between ERA5 and NASA — skipping plots.")
        return

    era5_c = era5_df.loc[common_idx]
    nasa_c = nasa_df.loc[common_idx]

    # Daytime mask (GHI > 5 W/m²)
    day_mask = (era5_c[era5_ghi_col] > 5) & (nasa_c[nasa_ghi_col] > 5)

    # ── Figure 1 — Monthly mean GHI ───────────────────────────────────────────
    era5_monthly = era5_c[era5_ghi_col].groupby(era5_c.index.month).mean()
    nasa_monthly = nasa_c[nasa_ghi_col].groupby(nasa_c.index.month).mean()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(12)
    w = 0.38
    ax1.bar(x - w/2, era5_monthly.values, width=w, label="ERA5  (0.25°)", color="#2196F3", alpha=0.85)
    ax1.bar(x + w/2, nasa_monthly.values, width=w, label="NASA POWER (0.5°)", color="#FF9800", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels)
    ax1.set_ylabel("Mean GHI  [W/m²]", fontsize=10)
    ax1.set_title(
        "Monthly Mean GHI — ERA5 vs NASA POWER\n"
        "Positive bias in ERA5 during monsoon months would indicate finer cloud resolution",
        fontsize=10, fontweight="bold",
    )
    ax1.legend()
    fig1.tight_layout()
    p1 = fig_dir / "era5_vs_nasa_monthly_ghi.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info(f"Saved → {p1}")

    # ── Figure 2 — Hourly scatter ERA5 vs NASA GHI ───────────────────────────
    e_ghi = era5_c.loc[day_mask, era5_ghi_col].values
    n_ghi = nasa_c.loc[day_mask, nasa_ghi_col].values

    # Subsample for plot readability (max 5000 points)
    if len(e_ghi) > 5000:
        idx = np.random.default_rng(42).choice(len(e_ghi), 5000, replace=False)
        e_ghi, n_ghi = e_ghi[idx], n_ghi[idx]

    slope, intercept = np.polyfit(n_ghi, e_ghi, 1)
    r2 = float(np.corrcoef(n_ghi, e_ghi)[0, 1] ** 2)
    rmse = float(np.sqrt(np.mean((e_ghi - n_ghi) ** 2)))
    mbe  = float(np.mean(e_ghi - n_ghi))

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.hexbin(n_ghi, e_ghi, gridsize=50, cmap="Blues", mincnt=1)
    lim = max(n_ghi.max(), e_ghi.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 line")
    fit_x = np.array([0, lim])
    ax2.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.5,
             label=f"OLS: ERA5 = {slope:.3f}·NASA + {intercept:.1f}")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("NASA POWER GHI  [W/m²]", fontsize=10)
    ax2.set_ylabel("ERA5 GHI  [W/m²]", fontsize=10)
    ax2.set_title(
        "Hourly GHI: ERA5 vs NASA POWER  (daytime only)",
        fontsize=10, fontweight="bold",
    )
    ax2.text(
        0.04, 0.96,
        f"R² = {r2:.3f}\nRMSE = {rmse:.1f} W/m²\nMBE = {mbe:+.1f} W/m²\nn = {len(e_ghi):,}",
        transform=ax2.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = fig_dir / "era5_vs_nasa_ghi_scatter.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved → {p2}")

    # ── Figure 3 — Temperature + RH monthly means ────────────────────────────
    fig3, axes = plt.subplots(1, 2, figsize=(11, 4))

    era5_t = era5_c[era5_t2m_col].groupby(era5_c.index.month).mean()
    nasa_t = nasa_c[nasa_t2m_col].groupby(nasa_c.index.month).mean()
    ax = axes[0]
    ax.plot(range(1, 13), era5_t.values, "o-", color="#2196F3", label="ERA5")
    ax.plot(range(1, 13), nasa_t.values, "s-", color="#FF9800", label="NASA POWER")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=7)
    ax.set_ylabel("Mean T2M  [°C]", fontsize=9)
    ax.set_title("Monthly Mean Temperature", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    if era5_rh_col in era5_c.columns and nasa_rh_col in nasa_c.columns:
        era5_rh = era5_c[era5_rh_col].groupby(era5_c.index.month).mean()
        nasa_rh = nasa_c[nasa_rh_col].groupby(nasa_c.index.month).mean()
        ax2b = axes[1]
        ax2b.plot(range(1, 13), era5_rh.values, "o-", color="#2196F3", label="ERA5")
        ax2b.plot(range(1, 13), nasa_rh.values, "s-", color="#FF9800", label="NASA POWER")
        ax2b.set_xticks(range(1, 13))
        ax2b.set_xticklabels(month_labels, fontsize=7)
        ax2b.set_ylabel("Mean RH2M  [%]", fontsize=9)
        ax2b.set_title("Monthly Mean Relative Humidity", fontsize=9, fontweight="bold")
        ax2b.legend(fontsize=8)

    fig3.tight_layout()
    p3 = fig_dir / "era5_vs_nasa_met_monthly.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info(f"Saved → {p3}")

    # ── Figure 4 — 14-day GHI time series  (January 2022) ────────────────────
    try:
        zoom_start = pd.Timestamp("2022-01-01", tz="UTC")
        zoom_end   = pd.Timestamp("2022-01-14 23:00", tz="UTC")
        era5_z = era5_c[era5_ghi_col].loc[zoom_start:zoom_end]
        nasa_z = nasa_c[nasa_ghi_col].loc[zoom_start:zoom_end]

        if len(era5_z) >= 24:
            fig4, ax4 = plt.subplots(figsize=(13, 4))
            ax4.plot(era5_z.index, era5_z.values, lw=1.5, color="#2196F3",
                     label="ERA5  (0.25°)", alpha=0.9)
            ax4.plot(nasa_z.index, nasa_z.values, lw=1.5, color="#FF9800",
                     label="NASA POWER (0.5°)", alpha=0.9, linestyle="--")
            ax4.fill_between(era5_z.index, era5_z.values, nasa_z.values,
                             alpha=0.12, color="gray", label="difference")
            ax4.set_ylabel("GHI  [W/m²]", fontsize=10)
            ax4.set_title(
                "14-Day GHI Time Series — ERA5 vs NASA POWER  (Jan 2022)\n"
                "ERA5 tends to show sharper cloud edges due to finer spatial resolution",
                fontsize=10, fontweight="bold",
            )
            ax4.legend(fontsize=8)
            fig4.tight_layout()
            p4 = fig_dir / "era5_vs_nasa_timeseries_zoom.png"
            fig4.savefig(p4, dpi=150, bbox_inches="tight")
            plt.close(fig4)
            logger.info(f"Saved → {p4}")
    except Exception as exc:
        logger.warning(f"Zoom plot skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Setup guide
# ─────────────────────────────────────────────────────────────────────────────

def _print_setup_guide() -> None:
    guide = """
╔══════════════════════════════════════════════════════════════════════════╗
║               ERA5 CDS API — One-Time Setup Required                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Register (free) at:                                                  ║
║     https://cds.climate.copernicus.eu                                   ║
║                                                                          ║
║  2. Accept the ERA5 licence at:                                          ║
║     CDS website → Datasets → ERA5 hourly single levels → Licence        ║
║     (without accepting the licence, downloads will queue but fail)       ║
║                                                                          ║
║  3. Get your API key:                                                    ║
║     CDS profile page (top-right) → API key → copy uid:key               ║
║                                                                          ║
║  4. Create  ~/.cdsapirc  with this content:                              ║
║                                                                          ║
║       url: https://cds.climate.copernicus.eu/api                        ║
║       key: <your-uid>:<your-api-key>                                     ║
║                                                                          ║
║  5. Install dependencies:                                                ║
║       pip install cdsapi xarray netcdf4                                  ║
║                                                                          ║
║  6. Re-run:  python scripts/fetch_era5.py                               ║
║                                                                          ║
║  Note: Each year download is ~40–80 MB; expect 5–30 min per year        ║
║  depending on CDS queue. Total ~400 MB for 2020–2025.                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process ERA5 reanalysis data for the PV site."
    )
    parser.add_argument("--config",       default="configs/site.yaml")
    parser.add_argument("--start",        type=int, default=2020,
                        help="First year to download (inclusive)")
    parser.add_argument("--end",          type=int, default=2025,
                        help="Last year to download (inclusive)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip CDS download; load existing NetCDF files only")
    parser.add_argument("--skip-plots",    action="store_true",
                        help="Skip ERA5 vs NASA comparison plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Site     : {cfg['site']['name']}")
    logger.info(f"Location : {cfg['site']['latitude']}° N, {cfg['site']['longitude']}° E")
    logger.info(f"Years    : {args.start} – {args.end}")

    # ── 1. Download ERA5 NetCDF files from CDS ────────────────────────────────
    era5_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "era5"

    if args.skip_download:
        logger.info(f"--skip-download set; loading existing files from {era5_dir}")
    else:
        try:
            era5_dir = download_era5(cfg, start_year=args.start, end_year=args.end)
        except ImportError:
            _print_setup_guide()
            sys.exit(1)
        except Exception as exc:
            # CDS credentials or network error
            logger.error(f"ERA5 download failed: {exc}")
            _print_setup_guide()
            sys.exit(1)

    # ── 2. Load and process ERA5 ──────────────────────────────────────────────
    logger.info("Loading and processing ERA5 NetCDF files …")
    try:
        era5_raw = load_era5(era5_dir, cfg)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        _print_setup_guide()
        sys.exit(1)

    # ── 3. Rename to NASA POWER *_cal schema ─────────────────────────────────
    logger.info("Converting to NASA POWER *_cal column schema …")
    era5_cal = era5_to_nasa_schema(era5_raw)
    logger.info(f"  ERA5 columns: {list(era5_cal.columns)}")

    # ── 4. Save processed CSV ─────────────────────────────────────────────────
    from src.data.era5_loader import save_era5_processed
    out_path = save_era5_processed(era5_cal, cfg)

    # ── 5. Comparison plots vs NASA POWER ────────────────────────────────────
    if not args.skip_plots:
        nasa_cal_path = resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv"
        if nasa_cal_path.exists():
            logger.info("Loading NASA calibrated data for comparison plots …")
            nasa_cal = pd.read_csv(
                nasa_cal_path,
                index_col="timestamp_utc",
                parse_dates=True,
            )
            nasa_cal.index = pd.to_datetime(nasa_cal.index, utc=True)
            # Keep only _cal columns for a fair comparison
            nasa_cal_cols = [c for c in nasa_cal.columns if c.endswith("_cal")]
            nasa_cal = nasa_cal[nasa_cal_cols]

            logger.info("Generating ERA5 vs NASA comparison plots …")
            plot_era5_vs_nasa(era5_cal, nasa_cal, cfg)
        else:
            logger.warning(
                f"  {nasa_cal_path} not found — skipping comparison plots.\n"
                "  Run  python scripts/run_calibration.py  first."
            )

    # ── 6. Preview ────────────────────────────────────────────────────────────
    print("\n── ERA5 processed dataset ────────────────────────────────────────────")
    print(f"  Shape        : {era5_cal.shape}")
    print(f"  Date range   : {era5_cal.index.min()}  →  {era5_cal.index.max()}")
    print(f"  Columns      : {list(era5_cal.columns)}")
    print()
    print("  Statistics (daytime only, GHI > 5 W/m²):")
    day = era5_cal[era5_cal["ALLSKY_SFC_SW_DWN_cal"] > 5]
    stats_cols = ["ALLSKY_SFC_SW_DWN_cal", "T2M_cal", "RH2M_cal", "WS10M_cal"]
    stats_cols = [c for c in stats_cols if c in day.columns]
    if stats_cols:
        print(day[stats_cols].describe().round(2).to_string())
    print()
    print(f"  Saved to     : {out_path}")
    print()
    print("── Next steps ────────────────────────────────────────────────────────")
    print("  Run PV simulation with ERA5:")
    print("    python scripts/run_pv_model.py --era5")
    print()
    print("  Then re-run feature engineering:")
    print("    python scripts/run_feature_engineering.py --era5")


if __name__ == "__main__":
    main()
