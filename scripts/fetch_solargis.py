"""
scripts/fetch_solargis.py
--------------------------
Process SolarGIS irradiance data for the University of Moratuwa site.

SolarGIS data cannot be downloaded automatically (commercial product).
This script processes locally saved SolarGIS CSV files.

How to obtain SolarGIS data
----------------------------
Option A — Research request (free):
    Email  info@solargis.com  with subject "Research data request"
    Provide: site lat/lon, study period, institution name, research purpose.
    Many universities have existing agreements.

Option B — SolarGIS Prospect (manual download, web UI):
    https://solargis.com/maps-and-gis-data/overview/
    Site: Lat=6.7912  Lon=79.9005 (University of Moratuwa, Sri Lanka)
    Parameters: GHI, DNI, DIF, Clearsky GHI, TEMP, WS
    Period: 2020-01-01 to 2025-12-31
    Resolution: Hourly, UTC

Option C — SolarGIS API (commercial):
    https://solargis.com/docs/api
    Set: export SOLARGIS_API_KEY=<your-key>
    Contact sales for API access pricing.

Once you have the CSV file(s), place them in:
    data/external/solargis/

Then run:
    python scripts/fetch_solargis.py --local

Outputs
-------
  data/processed/solargis_processed.csv           — cleaned, UTC-indexed
  results/figures/solargis_vs_nasa_monthly_ghi.png
  results/figures/solargis_vs_nasa_scatter.png
  results/figures/solargis_vs_nasa_timeseries.png
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
from src.data.solargis_loader import (
    check_solargis_files,
    load_solargis,
    solargis_to_nasa_schema,
    save_solargis_processed,
)

logger = get_logger("fetch_solargis")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_solargis_vs_nasa(solargis_df: pd.DataFrame, nasa_df: pd.DataFrame,
                          cfg: dict) -> None:
    """Monthly GHI bar chart, hourly scatter, and 14-day time-series zoom."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    ghi_sg = solargis_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    ghi_p  = nasa_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    common = ghi_sg.index.intersection(ghi_p.index)
    ghi_sg = ghi_sg.reindex(common)
    ghi_p  = ghi_p.reindex(common)

    # Monthly GHI
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_sg = ghi_sg.resample("ME").mean()
    monthly_p  = ghi_p.resample("ME").mean()
    x = np.arange(len(monthly_sg))
    w = 0.38
    ax.bar(x - w/2, monthly_p.values,  width=w, label="NASA POWER (0.5°)",
           color="#1565C0", alpha=0.75)
    ax.bar(x + w/2, monthly_sg.values, width=w, label="SolarGIS HelioSat-4 (~90 m)",
           color="#6A1B9A", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %Y") for d in monthly_sg.index], rotation=35, fontsize=8)
    ax.set_ylabel("Monthly Mean GHI (W/m²)", fontsize=10)
    ax.set_title("Monthly Mean GHI — NASA POWER vs SolarGIS", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "solargis_vs_nasa_monthly_ghi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter
    daytime = (ghi_sg > 50) & (ghi_p > 50)
    sample  = pd.concat([ghi_p[daytime], ghi_sg[daytime]], axis=1,
                        keys=["nasa","solargis"]).dropna().sample(
                            min(3000, daytime.sum()), random_state=42)
    r2   = float(1 - ((sample["solargis"] - sample["nasa"])**2).sum()
                 / ((sample["nasa"] - sample["nasa"].mean())**2).sum())
    rmse = float(np.sqrt(((sample["solargis"] - sample["nasa"])**2).mean()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(sample["nasa"], sample["solargis"], gridsize=60, cmap="Purples", mincnt=1)
    lims = [0, max(sample.max().max() * 1.05, 200)]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlabel("NASA POWER GHI (W/m²)", fontsize=10)
    ax.set_ylabel("SolarGIS GHI (W/m²)", fontsize=10)
    ax.set_title(f"Hourly GHI: NASA vs SolarGIS\nR²={r2:.4f}  RMSE={rmse:.1f} W/m²",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(fig_dir / "solargis_vs_nasa_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 14-day zoom
    t0 = pd.Timestamp("2023-01-09", tz="UTC")
    t1 = pd.Timestamp("2023-01-23", tz="UTC")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ghi_p.loc[t0:t1].index, ghi_p.loc[t0:t1].values,
            lw=1.2, color="#1565C0", alpha=0.9, label="NASA POWER")
    ax.plot(ghi_sg.loc[t0:t1].index, ghi_sg.loc[t0:t1].values,
            lw=1.2, color="#6A1B9A", alpha=0.85, linestyle="--", label="SolarGIS")
    ax.set_ylabel("GHI (W/m²)", fontsize=10)
    ax.set_title("14-Day GHI Comparison — Jan 9–23, 2023",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "solargis_vs_nasa_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved SolarGIS vs NASA plots → results/figures/solargis_vs_nasa_*.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process locally saved SolarGIS CSV files for the site."
    )
    parser.add_argument("--config",      default="configs/site.yaml")
    parser.add_argument("--local",       action="store_true",
                        help="Process CSV files already in data/external/solargis/")
    parser.add_argument("--skip-plots",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Check files exist ─────────────────────────────────────────────────────
    # check_solargis_files raises FileNotFoundError with setup instructions
    # if no files are found.
    check_solargis_files(cfg)

    # ── Load and process ──────────────────────────────────────────────────────
    logger.info("Loading SolarGIS data …")
    raw_df    = load_solargis(cfg)
    schema_df = solargis_to_nasa_schema(raw_df)
    save_solargis_processed(schema_df, cfg)

    logger.info(f"SolarGIS: {len(schema_df):,} rows  "
                f"({schema_df.index.min().date()} → {schema_df.index.max().date()})")

    # ── Comparison plots ──────────────────────────────────────────────────────
    if not args.skip_plots:
        try:
            nasa_path = resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv"
            if nasa_path.exists():
                nasa_df = pd.read_csv(nasa_path, index_col="timestamp_utc", parse_dates=True)
                nasa_df.index = pd.to_datetime(nasa_df.index, utc=True)
                plot_solargis_vs_nasa(schema_df, nasa_df, cfg)
            else:
                logger.info("NASA calibrated data not found — skipping comparison plots.")
        except Exception as exc:
            logger.warning(f"Comparison plots failed: {exc}")


if __name__ == "__main__":
    main()
