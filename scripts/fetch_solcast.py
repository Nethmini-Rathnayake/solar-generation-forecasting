"""
scripts/fetch_solcast.py
-------------------------
Download, process, and save Solcast historical radiation data for the
University of Moratuwa site.  Solcast uses ~1–2 km satellite resolution.

Prerequisites (one-time)
-------------------------
1. Register for a free hobbyist API key at:
       https://toolkit.solcast.com.au/register/hobbyist
2. Set environment variable:  export SOLCAST_API_KEY=<your-key>
   OR pass  --api-key <key>  flag below.

Rate limit
----------
Free tier: 10 API calls/day.  Full 2020–2025 history needs ~72 calls.
Run this script once per day — it caches monthly chunks and skips already
downloaded months.

    python scripts/fetch_solcast.py          # download next batch + process
    python scripts/fetch_solcast.py --status # show download progress
    python scripts/fetch_solcast.py --tmy    # download TMY (1 call, quick)

Outputs
-------
  data/external/solcast/solcast_<year>_<month>.csv  — monthly raw files
  data/processed/solcast_processed.csv              — combined, UTC-indexed
  data/processed/solcast_tmy_processed.csv          — TMY (if --tmy used)
  results/figures/solcast_vs_nasa_monthly_ghi.png
  results/figures/solcast_vs_nasa_scatter.png
  results/figures/solcast_vs_nasa_timeseries.png
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
from src.data.solcast_loader import (
    download_solcast_monthly,
    download_solcast_tmy,
    load_solcast,
    solcast_to_nasa_schema,
    save_solcast_processed,
    download_status,
)

logger = get_logger("fetch_solcast")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_solcast_vs_nasa(solcast_df: pd.DataFrame, nasa_df: pd.DataFrame,
                         cfg: dict) -> None:
    """Monthly GHI bar chart, hourly scatter, and 14-day time-series zoom."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    ghi_s = solcast_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    ghi_p = nasa_df["ALLSKY_SFC_SW_DWN_cal"].clip(lower=0)
    common = ghi_s.index.intersection(ghi_p.index)
    ghi_s  = ghi_s.reindex(common)
    ghi_p  = ghi_p.reindex(common)

    # Monthly GHI
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_s = ghi_s.resample("ME").mean()
    monthly_p = ghi_p.resample("ME").mean()
    x = np.arange(len(monthly_s))
    w = 0.38
    ax.bar(x - w/2, monthly_p.values, width=w, label="NASA POWER (0.5°)",
           color="#1565C0", alpha=0.75)
    ax.bar(x + w/2, monthly_s.values, width=w, label="Solcast (~1–2 km)",
           color="#2E7D32", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %Y") for d in monthly_s.index], rotation=35, fontsize=8)
    ax.set_ylabel("Monthly Mean GHI (W/m²)", fontsize=10)
    ax.set_title("Monthly Mean GHI — NASA POWER vs Solcast", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "solcast_vs_nasa_monthly_ghi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter
    daytime = (ghi_s > 50) & (ghi_p > 50)
    sample  = pd.concat([ghi_p[daytime], ghi_s[daytime]], axis=1,
                        keys=["nasa","solcast"]).dropna().sample(
                            min(3000, daytime.sum()), random_state=42)
    r2   = float(1 - ((sample["solcast"] - sample["nasa"])**2).sum()
                 / ((sample["nasa"] - sample["nasa"].mean())**2).sum())
    rmse = float(np.sqrt(((sample["solcast"] - sample["nasa"])**2).mean()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(sample["nasa"], sample["solcast"], gridsize=60, cmap="YlGn", mincnt=1)
    lims = [0, max(sample.max().max() * 1.05, 200)]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlabel("NASA POWER GHI (W/m²)", fontsize=10)
    ax.set_ylabel("Solcast GHI (W/m²)", fontsize=10)
    ax.set_title(f"Hourly GHI: NASA vs Solcast\nR²={r2:.4f}  RMSE={rmse:.1f} W/m²",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(fig_dir / "solcast_vs_nasa_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 14-day zoom
    t0 = pd.Timestamp("2023-01-09", tz="UTC")
    t1 = pd.Timestamp("2023-01-23", tz="UTC")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ghi_p.loc[t0:t1].index, ghi_p.loc[t0:t1].values,
            lw=1.2, color="#1565C0", alpha=0.9, label="NASA POWER")
    ax.plot(ghi_s.loc[t0:t1].index, ghi_s.loc[t0:t1].values,
            lw=1.2, color="#2E7D32", alpha=0.85, linestyle="--", label="Solcast")
    ax.set_ylabel("GHI (W/m²)", fontsize=10)
    ax.set_title("14-Day GHI Comparison — Jan 9–23, 2023",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "solcast_vs_nasa_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved Solcast vs NASA plots → results/figures/solcast_vs_nasa_*.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Solcast historical radiation data for the site."
    )
    parser.add_argument("--config",    default="configs/site.yaml")
    parser.add_argument("--api-key",   default=None,
                        help="Solcast API key (or set SOLCAST_API_KEY env var)")
    parser.add_argument("--start",     type=int, default=2020)
    parser.add_argument("--end",       type=int, default=2025)
    parser.add_argument("--tmy",       action="store_true",
                        help="Download TMY (1 API call) instead of full history.")
    parser.add_argument("--status",    action="store_true",
                        help="Show download progress and exit.")
    parser.add_argument("--calls",     type=int, default=10,
                        help="Max API calls to make today (default: 10 = daily limit).")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.status:
        download_status(cfg, args.start, args.end)
        return

    if args.tmy:
        # ── TMY path ──────────────────────────────────────────────────────────
        download_solcast_tmy(cfg, api_key=args.api_key)
        raw_df    = load_solcast(cfg, tmy=True)
        schema_df = solcast_to_nasa_schema(raw_df)
        save_solcast_processed(schema_df, cfg, tmy=True)
        logger.info("Solcast TMY processed and saved.")
    else:
        # ── Historical monthly path ───────────────────────────────────────────
        calls_made = 0
        for year in range(args.start, args.end + 1):
            for month in range(1, 13):
                if calls_made >= args.calls:
                    logger.info(
                        f"Daily call limit ({args.calls}) reached. "
                        "Run again tomorrow to continue."
                    )
                    break
                result = download_solcast_monthly(
                    cfg, api_key=args.api_key, year=year, month=month
                )
                if result is not None:   # None means skipped (already cached)
                    calls_made += 1
            else:
                continue
            break   # inner break propagates

        logger.info(f"Made {calls_made} new API calls today.")
        download_status(cfg, args.start, args.end)

        # Process all cached months into a single file
        try:
            raw_df    = load_solcast(cfg, tmy=False)
            schema_df = solcast_to_nasa_schema(raw_df)
            save_solcast_processed(schema_df, cfg, tmy=False)
        except FileNotFoundError as e:
            logger.warning(f"{e}\nProcessing will complete when all months are downloaded.")
            return

    # ── Comparison plots ──────────────────────────────────────────────────────
    if not args.skip_plots:
        try:
            nasa_path = resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv"
            if nasa_path.exists():
                nasa_df = pd.read_csv(nasa_path, index_col="timestamp_utc", parse_dates=True)
                nasa_df.index = pd.to_datetime(nasa_df.index, utc=True)
                tmy_flag = args.tmy
                plot_solcast_vs_nasa(schema_df, nasa_df, cfg)
            else:
                logger.info("NASA calibrated data not found — skipping comparison plots.")
        except Exception as exc:
            logger.warning(f"Comparison plots failed: {exc}")


if __name__ == "__main__":
    main()
