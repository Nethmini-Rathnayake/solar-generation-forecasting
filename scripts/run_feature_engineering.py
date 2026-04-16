"""
scripts/run_feature_engineering.py
------------------------------------
Assembles the full feature matrix for the 24h-ahead XGBoost forecaster.

Pipeline
--------
1. Load synthetic PV (6 years, hourly) + calibrated NASA POWER (or ERA5)
2. Merge into one DataFrame
3. Add time/solar features      (src/features/time_features.py)
4. Add lag features              (src/features/lag_features.py)
5. Add rolling stats + diffs     (src/features/rolling_stats.py)
6. Build 24 target columns       (target_h1 … target_h24)
7. Drop rows with any NaN in features or targets
8. Save to data/processed/feature_matrix.parquet (+ .csv for inspection)
9. Plot feature overview (correlation heatmap + feature importance preview)

Output
------
  data/processed/feature_matrix.parquet   — full matrix (fast load for model)
  data/processed/feature_matrix_sample.csv — first 200 rows (human inspection)
  results/figures/feature_correlation.png
  results/figures/feature_target_correlation.png

Run from project root:
    python scripts/run_feature_engineering.py          # NASA POWER (default)
    python scripts/run_feature_engineering.py --era5   # ERA5 reanalysis
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
from src.features.time_features  import add_time_features
from src.features.lag_features   import (
    add_lag_features, build_target_matrix,
    get_feature_cols, get_target_cols,
)
from src.features.rolling_stats  import add_rolling_features, add_diff_features
from src.preproccesing.clean     import flag_unavailable_hours

logger = get_logger("run_feature_engineering")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Feature→Target correlation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_target_corr(df: pd.DataFrame, feature_cols: list[str],
                              cfg: dict) -> None:
    """
    Horizontal bar chart: Pearson |r| of each feature vs target_h1.

    Shows which features are most predictive of next-hour PV output.
    Top-20 shown.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    if "target_h1" not in df.columns:
        logger.warning("target_h1 not in df — skipping feature-target plot")
        return

    corr = (
        df[feature_cols + ["target_h1"]]
        .corr()["target_h1"]
        .drop("target_h1")
        .abs()
        .sort_values(ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("Blues_d", len(corr))
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, corr.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, corr.max() * 1.18)
    ax.set_xlabel("|Pearson r|  vs  target_h1  (next hour PV power)", fontsize=10)
    ax.set_title(
        "Top-20 Features by Correlation with 1h-Ahead PV Power\n"
        "Higher = stronger linear relationship",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "feature_target_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Correlation with each horizon (h1…h24)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lag_vs_horizon(df: pd.DataFrame, cfg: dict) -> None:
    """
    Line plot: how the correlation of top lag features decays across h1…h24.

    Illustrates why we need horizon-specific models:
    pv_ac_W_lag1 is very predictive for h=1 but much less so for h=24.
    pv_ac_W_lag24 becomes relatively more predictive for h=24.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # Choose the 5 most informative lag/rolling features vs target_h1
    lag_cols = [c for c in df.columns
                if ("_lag" in c or "_roll" in c or "_diff" in c)
                and "target" not in c]
    if len(lag_cols) == 0:
        return

    top5 = (
        df[lag_cols + ["target_h1"]]
        .corr()["target_h1"]
        .drop("target_h1")
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    target_cols = get_target_cols(24)
    horizons    = list(range(1, 25))
    corr_rows   = {}
    for feat in top5:
        corr_rows[feat] = [
            df[[feat, tc]].corr().iloc[0, 1] if tc in df.columns else np.nan
            for tc in target_cols
        ]

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = sns.color_palette("tab10", len(top5))
    for (feat, vals), color in zip(corr_rows.items(), palette):
        label = feat.replace("pv_ac_W_", "")
        ax.plot(horizons, vals, marker="o", ms=4, lw=1.8,
                color=color, label=label)

    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("Forecast Horizon  h  (hours ahead)", fontsize=10)
    ax.set_ylabel("Pearson r  (feature vs target_h{h})", fontsize=10)
    ax.set_xticks(horizons)
    ax.set_title(
        "Feature Predictive Power vs Forecast Horizon  (h=1…24)\n"
        "This is why we train 24 separate models — importance changes with h",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", title="Feature")
    fig.tight_layout()

    out = resolve_path(cfg["paths"]["figures"]) / "feature_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build feature matrix for 24h-ahead PV forecasting."
    )
    parser.add_argument("--config", default="configs/site.yaml")
    parser.add_argument(
        "--era5", action="store_true",
        help="Use ERA5 reanalysis instead of calibrated NASA POWER. "
             "Requires  python scripts/fetch_era5.py  to have been run first.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Site: {cfg['site']['name']}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading synthetic PV …")
    pv = pd.read_csv(
        resolve_path(cfg["paths"]["synthetic"]) / "pv_synthetic_6yr.csv",
        index_col="timestamp_utc", parse_dates=True,
    )
    pv.index = pd.to_datetime(pv.index, utc=True)

    logger.info("Loading local measurements (for unavailability detection) …")
    local = pd.read_csv(
        resolve_path(cfg["paths"]["interim"]) / "local_hourly_utc.csv",
        index_col="timestamp_utc", parse_dates=True,
    )
    local.index = pd.to_datetime(local.index, utc=True)

    if args.era5:
        from src.data.era5_loader import load_era5_processed
        logger.info("Loading ERA5 processed data …")
        nasa = load_era5_processed(cfg)
    else:
        logger.info("Loading calibrated NASA POWER …")
        nasa = pd.read_csv(
            resolve_path(cfg["paths"]["processed"]) / "nasa_calibrated.csv",
            index_col="timestamp_utc", parse_dates=True,
        )
        nasa.index = pd.to_datetime(nasa.index, utc=True)

    # Use only _cal columns from NASA (the raw duplicates add noise)
    nasa_cal_cols = [c for c in nasa.columns if c.endswith("_cal")]
    nasa_keep     = nasa[nasa_cal_cols]

    # ── 2. Merge on common timestamps ─────────────────────────────────────────
    df = pv.join(nasa_keep, how="inner")
    logger.info(f"Merged: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── 3. Time / solar features ──────────────────────────────────────────────
    logger.info("Adding time/solar features …")
    df = add_time_features(df, cfg)

    # ── 4. Lag features ───────────────────────────────────────────────────────
    logger.info("Adding lag features …")
    df = add_lag_features(df, target_col="pv_ac_W")

    # Also lag the key weather inputs — GHI and temperature 24h back are
    # useful predictors for the same-time-tomorrow irradiance
    df = add_lag_features(df, target_col="ALLSKY_SFC_SW_DWN_cal",
                          lags=[1, 2, 3, 24, 48])
    df = add_lag_features(df, target_col="T2M_cal",
                          lags=[1, 24])

    # ── 5. Rolling stats + diffs ──────────────────────────────────────────────
    logger.info("Adding rolling statistics …")
    df = add_rolling_features(df, target_col="pv_ac_W")
    df = add_diff_features(df,    target_col="pv_ac_W")

    # ── 6. Build 24 target columns ────────────────────────────────────────────
    logger.info("Building target matrix (target_h1 … target_h24) …")
    df = build_target_matrix(df, target_col="pv_ac_W", horizons=24)

    # ── 7. Drop rows with any NaN ─────────────────────────────────────────────
    n_before = len(df)
    feature_cols = get_feature_cols(df, target_col="pv_ac_W")
    target_cols  = get_target_cols(24)

    df = df.dropna(subset=feature_cols + target_cols)
    n_nan_dropped = n_before - len(df)
    logger.info(
        f"Dropped {n_nan_dropped:,} rows with NaN  "
        f"({n_nan_dropped/n_before*100:.1f}% of {n_before:,}) — "
        f"mostly early rows missing lag history"
    )

    # ── 7b. Flag and remove unavailability hours ──────────────────────────────
    # Hours where the physics model says "sunny" but the real system was off
    # (grid outage, inverter trip, maintenance). Teaching the model these hours
    # causes systematic under-prediction on clear days.
    logger.info("Flagging system unavailability hours …")
    unavail_mask = flag_unavailable_hours(pv, local)

    # Keep the flag as a column in the full matrix (useful for analysis)
    df["unavailable"] = unavail_mask.reindex(df.index, fill_value=False).astype(int)

    # Split into clean training set and held-out unavailability set
    df_clean = df[df["unavailable"] == 0].drop(columns=["unavailable"])
    df_unavail = df[df["unavailable"] == 1]

    n_unavail = len(df_unavail)
    logger.info(
        f"Removed {n_unavail} unavailability hours from training set  "
        f"({100*n_unavail/len(df):.1f}% of post-NaN rows)"
    )

    # Save unavailability index for diagnostics
    unavail_path = resolve_path(cfg["paths"]["processed"]) / "unavailable_hours.csv"
    df_unavail.index.to_frame().to_csv(unavail_path, index=False)
    logger.info(f"Saved unavailability index → {unavail_path}")

    df = df_clean

    logger.info(f"Final training matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"  NaN rows dropped    : {n_nan_dropped:,}")
    logger.info(f"  Unavail rows removed: {n_unavail}")
    logger.info(f"Feature cols : {len(feature_cols)}")
    logger.info(f"Target  cols : {len(target_cols)}")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out_dir = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "feature_matrix.parquet"
    csv_sample   = out_dir / "feature_matrix_sample.csv"

    df.to_parquet(parquet_path)
    df.head(200).to_csv(csv_sample)

    logger.info(f"Saved → {parquet_path}  ({parquet_path.stat().st_size/1024:.0f} KB)")
    logger.info(f"Saved → {csv_sample}  (first 200 rows, CSV)")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    logger.info("Generating feature plots …")
    resolve_path(cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)

    plot_feature_target_corr(df, feature_cols, cfg)
    plot_lag_vs_horizon(df, cfg)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── Feature matrix preview ───────────────────────────────────────────")
    print(f"  Shape        : {df.shape}")
    print(f"  Date range   : {df.index.min()}  →  {df.index.max()}")
    print(f"  Feature cols : {len(feature_cols)}")
    print(f"  Target cols  : {len(target_cols)}  (target_h1 … target_h24)")
    print()
    print("  Feature groups:")
    groups = {
        "PV physics (poa, temp_cell, solar_elev)": [c for c in feature_cols if c in ("poa_global","temp_cell","solar_elevation","solar_elevation_deg","solar_azimuth_deg","cos_solar_zenith","is_daytime")],
        "Time cyclic (sin/cos)":   [c for c in feature_cols if c.startswith("sin_") or c.startswith("cos_")],
        "Calendar (raw)":          [c for c in feature_cols if c in ("hour","day_of_year","month","weekday","is_weekend","clearness_index")],
        "Lags pv_ac_W":            [c for c in feature_cols if "pv_ac_W_lag" in c],
        "Lags GHI":                [c for c in feature_cols if "ALLSKY" in c and "_lag" in c],
        "Lags T2M":                [c for c in feature_cols if "T2M" in c and "_lag" in c],
        "Rolling stats":           [c for c in feature_cols if "_roll" in c],
        "Diff features":           [c for c in feature_cols if "_diff" in c],
        "NASA cal (current-hour)": [c for c in feature_cols if c.endswith("_cal") and "_lag" not in c],
    }
    for grp, cols in groups.items():
        if cols:
            print(f"    {grp:<38}: {len(cols):>2}  → {cols[:4]}{'…' if len(cols)>4 else ''}")

    print()
    print("  Target sample (first daytime row):")
    first_day = df[df["is_daytime"] == 1].head(1)
    print(first_day[target_cols].T.rename(columns={first_day.index[0]: "pv_ac_W (W)"}).to_string())


if __name__ == "__main__":
    main()
