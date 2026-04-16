"""
scripts/run_wind_cmv_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Wind-based Cloud Motion Vector pipeline.

Replaces the Himawari-8 CMV pipeline (archived satellite data too slow to
download).  Uses Open-Meteo ERA5-reanalysis surface wind + Solcast
cloud_opacity to compute equivalent CMV features.

Stages
──────
1. Fetch hourly 10 m wind from Open-Meteo (Apr 2022 – Mar 2023, free API)
2. Build CMV features (cloud speed/direction, shadow arrivals, opacity lags)
3. Merge into existing 5-min feature matrix
4. Diagnostic plots

Outputs
───────
    data/interim/wind_hourly_openmeteo.csv          — raw wind download
    data/interim/wind_cmv_features_5min.csv         — CMV feature table
    data/processed/feature_matrix_cmv.parquet       — merged feature matrix

Run
───
    python scripts/run_wind_cmv_pipeline.py
    python scripts/run_wind_cmv_pipeline.py --plot
    python scripts/run_wind_cmv_pipeline.py --skip-fetch   # reuse cached wind
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cmv.wind_cmv import fetch_openmeteo_wind, build_wind_cmv_features
from src.utils.logger import get_logger

logger = get_logger("wind_cmv_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
_WIND_CACHE     = Path("data/interim/wind_hourly_openmeteo.csv")
_CMV_OUT        = Path("data/interim/wind_cmv_features_5min.csv")
_SOLCAST_5MIN   = Path("data/interim/solcast_5min_aligned.csv")
_FEATURE_MATRIX = Path("data/processed/feature_matrix.parquet")
_CMV_MATRIX_OUT = Path("data/processed/feature_matrix_cmv.parquet")
_FIG_DIR        = Path("results/figures/cmv")

# Overlap period (Solcast data covers Apr 2022 – Mar 2023)
_START = "2022-04-01"
_END   = "2023-03-31"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — fetch wind
# ─────────────────────────────────────────────────────────────────────────────

def stage_fetch_wind(skip_fetch: bool) -> pd.DataFrame:
    if skip_fetch and _WIND_CACHE.exists():
        logger.info(f"[1] Loading cached wind: {_WIND_CACHE}")
        df = pd.read_csv(_WIND_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    logger.info("[1] Fetching wind from Open-Meteo …")
    df = fetch_openmeteo_wind(_START, _END)

    _WIND_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_WIND_CACHE)
    logger.info(f"    Saved → {_WIND_CACHE}  ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — build CMV features
# ─────────────────────────────────────────────────────────────────────────────

def stage_build_features(wind_hourly: pd.DataFrame) -> pd.DataFrame:
    logger.info("[2] Loading Solcast 5-min data …")
    solcast = pd.read_csv(_SOLCAST_5MIN, index_col=0, parse_dates=True)
    solcast.index = pd.to_datetime(solcast.index, utc=True)

    logger.info(f"    Solcast: {solcast.shape}  {solcast.index[0]} → {solcast.index[-1]}")
    logger.info("[2] Building wind CMV features …")
    cmv = build_wind_cmv_features(solcast, wind_hourly)

    _CMV_OUT.parent.mkdir(parents=True, exist_ok=True)
    cmv.to_csv(_CMV_OUT)
    logger.info(f"    Saved → {_CMV_OUT}  ({cmv.shape})")
    return cmv


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — merge into feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def stage_merge(cmv: pd.DataFrame) -> pd.DataFrame:
    logger.info("[3] Loading feature matrix …")
    fm = pd.read_parquet(_FEATURE_MATRIX)
    fm.index = pd.to_datetime(fm.index, utc=True)
    logger.info(f"    Feature matrix: {fm.shape}")

    # Drop columns already in feature matrix to avoid conflicts
    existing = set(fm.columns)
    cmv_new  = [c for c in cmv.columns if c not in existing]
    logger.info(f"    Adding {len(cmv_new)} new CMV columns: {cmv_new}")
    merged = fm.join(cmv[cmv_new], how="left")

    _CMV_MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(_CMV_MATRIX_OUT)
    logger.info(f"    Merged → {_CMV_MATRIX_OUT}  ({merged.shape})")

    # Coverage report
    for col in ["cloud_speed_kmh", "shadow_arrival_5km", "opacity_lag_10km", "site_cloud_opacity"]:
        if col in merged.columns:
            n = merged[col].notna().sum()
            logger.info(f"      {col}: {n:,}/{len(merged):,} non-null ({100*n/len(merged):.1f}%)")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def stage_plots(cmv: pd.DataFrame, wind_hourly: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import warnings
    warnings.filterwarnings("ignore")

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[4] Generating diagnostic plots …")

    daytime = cmv[cmv["solar_zenith_deg"] < 90].copy()

    # ── Fig 1: Cloud speed time series (monthly box) ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Daily mean cloud speed
    daily_spd = daytime["cloud_speed_kmh"].resample("D").mean()
    ax = axes[0]
    ax.plot(daily_spd.index, daily_spd.values, lw=0.8, color="#1976D2", alpha=0.7)
    ax.set_ylabel("Cloud speed [km/h]", fontsize=10)
    ax.set_title("Daily mean cloud speed (850 hPa equivalent)", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.3)

    # Monthly wind rose summary
    monthly_dir = daytime.groupby(daytime.index.month)["cloud_direction_deg"].mean()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax2 = axes[1]
    valid_months = sorted(monthly_dir.index)
    ax2.bar([months[m-1] for m in valid_months],
            [daily_spd.resample("ME").mean().groupby(daily_spd.resample("ME").mean().index.month).mean().get(m, np.nan)
             for m in valid_months],
            color="#42A5F5", alpha=0.85)
    ax2.set_ylabel("Mean cloud speed [km/h]", fontsize=10)
    ax2.set_title("Monthly mean cloud speed", fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    p = _FIG_DIR / "01_cloud_speed.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    → {p}")

    # ── Fig 2: Wind rose (polar plot of cloud direction) ─────────────────────
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")
    dirs = np.radians(daytime["cloud_direction_deg"].dropna().values)
    bins = np.linspace(0, 2*np.pi, 37)
    counts, _ = np.histogram(dirs, bins=bins)
    theta  = (bins[:-1] + bins[1:]) / 2
    width  = bins[1] - bins[0]
    ax.bar(theta, counts, width=width, color="#42A5F5", alpha=0.8, align="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Cloud motion direction\n(daytime, Apr 2022 – Mar 2023)",
                 fontsize=11, fontweight="bold", pad=20)
    p = _FIG_DIR / "02_cloud_direction_rose.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    → {p}")

    # ── Fig 3: Shadow arrival times ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    dists = [5, 10, 20, 40]
    for ax, d in zip(axes.flat, dists):
        col = f"shadow_arrival_{d}km"
        vals = daytime[col].dropna()
        ax.hist(vals, bins=60, color="#26A69A", alpha=0.8, range=(0, 120))
        ax.axvline(vals.median(), color="red", lw=1.5, ls="--",
                   label=f"median={vals.median():.0f} min")
        ax.set_xlabel("Minutes until shadow arrival", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Shadow arrival from {d} km upstream", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Shadow arrival time distributions\n(assumes 1500 m cloud base)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _FIG_DIR / "03_shadow_arrival.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    → {p}")

    # ── Fig 4: Cloud opacity + speed correlation ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: opacity vs speed
    ax = axes[0]
    sample = daytime.dropna(subset=["site_cloud_opacity", "cloud_speed_kmh"]).sample(
        min(5000, len(daytime)), random_state=42)
    ax.hexbin(sample["cloud_speed_kmh"], sample["site_cloud_opacity"],
              gridsize=40, cmap="Blues", mincnt=1)
    ax.set_xlabel("Cloud speed [km/h]", fontsize=10)
    ax.set_ylabel("Cloud opacity [%]", fontsize=10)
    ax.set_title("Cloud speed vs opacity (daytime)", fontsize=10, fontweight="bold")

    # Time series zoom: 7 days in November 2022 (wet season)
    ax = axes[1]
    try:
        zoom = daytime["2022-11-01":"2022-11-07"]
        ax2r = ax.twinx()
        ax.plot(zoom.index, zoom["site_cloud_opacity"], lw=1.2, color="#7B1FA2",
                alpha=0.8, label="Cloud opacity [%]")
        ax2r.plot(zoom.index, zoom["cloud_speed_kmh"], lw=1.2, color="#1976D2",
                  alpha=0.8, label="Cloud speed [km/h]")
        ax.set_ylabel("Cloud opacity [%]", fontsize=9, color="#7B1FA2")
        ax2r.set_ylabel("Cloud speed [km/h]", fontsize=9, color="#1976D2")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_title("7-day zoom: Nov 2022 (NE monsoon)", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax2r.legend(loc="upper right", fontsize=8)
    except Exception:
        pass

    fig.tight_layout()
    p = _FIG_DIR / "04_opacity_speed.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    → {p}")

    # ── Fig 5: Feature coverage heatmap by month ──────────────────────────────
    cmv_cols_to_check = [
        "cloud_speed_kmh", "shadow_arrival_5km", "shadow_arrival_10km",
        "shadow_arrival_20km", "shadow_arrival_40km",
        "opacity_lag_5km", "opacity_lag_10km", "opacity_lag_20km", "opacity_lag_40km",
        "site_cloud_opacity", "cloud_opacity_trend",
    ]
    available = [c for c in cmv_cols_to_check if c in cmv.columns]
    monthly_cov = pd.DataFrame(index=range(1, 13))
    months_present = sorted(cmv.index.month.unique())
    for col in available:
        for m in months_present:
            mask = cmv.index.month == m
            pct  = cmv.loc[mask, col].notna().mean() * 100
            monthly_cov.loc[m, col] = pct

    fig, ax = plt.subplots(figsize=(13, 5))
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    cmap.set_bad("lightgray")
    data = monthly_cov.loc[months_present, available].values.astype(float)
    im = ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% non-null")
    ax.set_xticks(range(len(months_present)))
    ax.set_xticklabels([["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
                         for m in months_present], fontsize=9)
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available, fontsize=8)
    ax.set_title("CMV feature coverage by month (%)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = _FIG_DIR / "05_feature_coverage.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    → {p}")

    logger.info(f"  All plots saved to {_FIG_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wind-based CMV feature pipeline"
    )
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Reuse cached wind CSV instead of re-fetching")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging into feature matrix")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    args = parser.parse_args()

    logger.info("═" * 60)
    logger.info("  Wind-Based CMV Pipeline")
    logger.info("═" * 60)

    wind    = stage_fetch_wind(args.skip_fetch)
    cmv     = stage_build_features(wind)

    if not args.skip_merge:
        merged  = stage_merge(cmv)

    if args.plot:
        stage_plots(cmv, wind)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n── Summary ──────────────────────────────────────────────────")
    logger.info(f"  CMV features  : {cmv.shape[1]} columns × {len(cmv):,} rows")

    daytime = cmv[cmv["solar_zenith_deg"] < 90]
    logger.info(f"  Daytime rows  : {len(daytime):,}")
    logger.info(f"  Cloud speed   : {daytime['cloud_speed_kmh'].mean():.1f} ± "
                f"{daytime['cloud_speed_kmh'].std():.1f} km/h")
    logger.info(f"  Cloud dir     : {daytime['cloud_direction_deg'].mean():.0f}° mean")
    logger.info(f"  Shadow 10km   : median {daytime['shadow_arrival_10km'].median():.0f} min")
    logger.info(f"  Opacity lags  : {daytime['opacity_lag_10km'].notna().sum():,} valid values")
    logger.info("")
    logger.info(f"  Outputs:")
    logger.info(f"    {_WIND_CACHE}")
    logger.info(f"    {_CMV_OUT}")
    if not args.skip_merge:
        logger.info(f"    {_CMV_MATRIX_OUT}")
    if args.plot:
        logger.info(f"    {_FIG_DIR}/ (5 plots)")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
