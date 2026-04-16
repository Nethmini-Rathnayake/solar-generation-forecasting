"""
scripts/fetch_nsrdb.py
-----------------------
Download, process, and save NSRDB PSM v3 data for the University of Moratuwa
site (2020–2025).  NSRDB uses ~4 km satellite resolution vs NASA POWER's 0.5°.

Prerequisites (one-time)
-------------------------
1. Register for a free API key at  https://developer.nrel.gov/signup/
2. Set environment variable:  export NREL_API_KEY=<your-key>
   OR pass  --api-key <key>  flag below.

Run from project root:
    python scripts/fetch_nsrdb.py                    # download + process
    python scripts/fetch_nsrdb.py --skip-download    # process existing files
    python scripts/fetch_nsrdb.py --skip-plots       # skip comparison figures

Outputs
-------
  data/external/nsrdb/nsrdb_psm3_<year>_*.csv   — raw NSRDB yearly CSVs
  data/processed/nsrdb_processed.csv             — cleaned, UTC-indexed
  results/figures/nsrdb_vs_nasa_monthly_ghi.png  — monthly GHI comparison
  results/figures/nsrdb_vs_nasa_scatter.png      — hourly GHI scatter
  results/figures/nsrdb_vs_nasa_timeseries.png   — 14-day GHI zoom
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
from src.data.nsrdb_loader import (
    download_nsrdb,
    load_nsrdb,
    nsrdb_to_nasa_schema,
    save_nsrdb_processed,
)

logger = get_logger("fetch_nsrdb")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_nsrdb_vs_nasa(nsrdb_df: pd.DataFrame, nasa_df: pd.DataFrame,
                       cfg: dict) -> None:
    """
    Three comparison figures: monthly GHI bar chart, hourly scatter, 14-day zoom.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    ghi_n = nsrdb_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    ghi_p = nasa_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)

    common = ghi_n.index.intersection(ghi_p.index)
    ghi_n  = ghi_n.reindex(common)
    ghi_p  = ghi_p.reindex(common)

    # ── Figure 1: Monthly mean GHI ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_n = ghi_n.resample("ME").mean()
    monthly_p = ghi_p.resample("ME").mean()
    x = np.arange(len(monthly_n))
    w = 0.38
    ax.bar(x - w/2, monthly_p.values, width=w, label="NASA POWER (0.5°)",
           color="#1565C0", alpha=0.75)
    ax.bar(x + w/2, monthly_n.values, width=w, label="NSRDB PSM v3 (4 km)",
           color="#E65100", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %Y") for d in monthly_n.index], rotation=35, fontsize=8)
    ax.set_ylabel("Monthly Mean GHI (W/m²)", fontsize=10)
    ax.set_title("Monthly Mean GHI — NASA POWER vs NSRDB PSM v3", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "nsrdb_vs_nasa_monthly_ghi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Hourly scatter ──────────────────────────────────────────────
    daytime = (ghi_n > 50) & (ghi_p > 50)
    sample  = pd.concat([ghi_p[daytime], ghi_n[daytime]], axis=1,
                        keys=["nasa","nsrdb"]).dropna().sample(min(3000, daytime.sum()),
                                                                random_state=42)
    r2 = float(1 - ((sample["nsrdb"] - sample["nasa"])**2).sum()
               / ((sample["nasa"] - sample["nasa"].mean())**2).sum())
    rmse = float(np.sqrt(((sample["nsrdb"] - sample["nasa"])**2).mean()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(sample["nasa"], sample["nsrdb"], gridsize=60, cmap="YlOrRd", mincnt=1)
    lims = [0, max(sample.max().max() * 1.05, 200)]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlabel("NASA POWER GHI (W/m²)", fontsize=10)
    ax.set_ylabel("NSRDB PSM v3 GHI (W/m²)", fontsize=10)
    ax.set_title(f"Hourly GHI: NASA vs NSRDB\nR²={r2:.4f}  RMSE={rmse:.1f} W/m²",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(fig_dir / "nsrdb_vs_nasa_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: 14-day zoom ─────────────────────────────────────────────────
    t0 = pd.Timestamp("2023-01-09", tz="UTC")
    t1 = pd.Timestamp("2023-01-23", tz="UTC")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ghi_p.loc[t0:t1].index, ghi_p.loc[t0:t1].values,
            lw=1.2, color="#1565C0", alpha=0.9, label="NASA POWER")
    ax.plot(ghi_n.loc[t0:t1].index, ghi_n.loc[t0:t1].values,
            lw=1.2, color="#E65100", alpha=0.85, linestyle="--", label="NSRDB PSM v3")
    ax.set_ylabel("GHI (W/m²)", fontsize=10)
    ax.set_title("14-Day GHI Comparison — Jan 9–23, 2023  (Clear Dry Season)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "nsrdb_vs_nasa_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved NSRDB vs NASA comparison plots → results/figures/nsrdb_vs_nasa_*.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and process NSRDB PSM v3 data for the site."
    )
    parser.add_argument("--config",        default="configs/site.yaml")
    parser.add_argument("--api-key",       default=None,
                        help="NREL developer API key (or set NREL_API_KEY env var)")
    parser.add_argument("--start",         type=int, default=2020)
    parser.add_argument("--end",           type=int, default=2025)
    parser.add_argument("--full-name",     default="Researcher")
    parser.add_argument("--email",         default="researcher@university.edu")
    parser.add_argument("--affiliation",   default="University of Moratuwa")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download; process existing CSV files.")
    parser.add_argument("--skip-plots",    action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download:
        download_nsrdb(
            cfg,
            api_key=args.api_key,
            start_year=args.start,
            end_year=args.end,
            full_name=args.full_name,
            email=args.email,
            affiliation=args.affiliation,
        )

    # ── Load and process ──────────────────────────────────────────────────────
    logger.info("Loading and processing NSRDB data …")
    raw_df    = load_nsrdb(cfg)
    schema_df = nsrdb_to_nasa_schema(raw_df)
    save_nsrdb_processed(schema_df, cfg)

    logger.info(f"NSRDB: {len(schema_df):,} rows  "
                f"({schema_df.index.min().date()} → {schema_df.index.max().date()})")
    logger.info(f"  GHI range  : {schema_df['ALLSKY_SFC_SW_DWN_cal'].min():.0f} "
                f"– {schema_df['ALLSKY_SFC_SW_DWN_cal'].max():.0f} W/m²")
    logger.info(f"  Temp range : {schema_df['T2M_cal'].min():.1f} "
                f"– {schema_df['T2M_cal'].max():.1f} °C")

    # ── Comparison plots ──────────────────────────────────────────────────────
    if not args.skip_plots:
        try:
            nasa_path = resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv"
            if nasa_path.exists():
                nasa_df = pd.read_csv(nasa_path, index_col="timestamp_utc", parse_dates=True)
                nasa_df.index = pd.to_datetime(nasa_df.index, utc=True)
                plot_nsrdb_vs_nasa(schema_df, nasa_df, cfg)
            else:
                logger.info("NASA calibrated data not found — skipping comparison plots.")
        except Exception as exc:
            logger.warning(f"Comparison plots failed: {exc}")


if __name__ == "__main__":
    main()
