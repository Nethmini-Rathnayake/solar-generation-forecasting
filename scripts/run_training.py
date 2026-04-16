"""
scripts/run_training.py
------------------------
Train, evaluate, and save the 24h-ahead XGBoost Direct Multi-Step (DMS)
forecasting model for PV power at the University of Moratuwa site.

Pipeline
--------
1. Load feature matrix  (data/processed/feature_matrix.parquet)
2. Chronological train/val/test split  (70 / 15 / 15 %)
3. Train 24 XGBoost models  (one per horizon, with early stopping)
4. Compute baseline forecasts  (day-ahead persistence + climatology)
5. Evaluate all three on the test set  (RMSE, MAE, MBE, MAPE, nRMSE, R²)
6. Save models → results/models/
7. Save metrics → results/metrics/
8. Save predictions → results/predictions/
9. Generate 5 diagnostic plots → results/figures/

Outputs
-------
  results/models/xgb_h01.json … xgb_h24.json
  results/models/model_metadata.json
  results/metrics/metrics_xgboost.csv
  results/metrics/metrics_persistence.csv
  results/metrics/metrics_climatology.csv
  results/predictions/test_predictions.parquet
  results/figures/metrics_vs_horizon.png
  results/figures/scatter_selected_horizons.png
  results/figures/feature_importance.png
  results/figures/feature_importance_by_horizon.png
  results/figures/error_by_hour.png
  results/figures/forecast_sample_days.png
  results/figures/forecast_timeseries.png

Run from project root:
    python scripts/run_training.py
    python scripts/run_training.py --no-plots   # skip figures (faster)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.features.lag_features import get_feature_cols, get_target_cols
from src.models.gradient_boost import (
    train_dms_models, predict_dms,
    save_models, get_feature_importance,
)
from src.models.baseline import day_ahead_persistence, climatological_mean
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_vs_horizon,
    plot_scatter_horizons,
    plot_feature_importance,
    plot_importance_by_horizon,
    plot_error_by_hour,
    plot_sample_days,
    plot_forecast_timeseries,
)

logger = get_logger("run_training")

# Split fractions (chronological — no shuffle)
_TRAIN_FRAC = 0.70
_VAL_FRAC   = 0.15
# test fraction = 1 - 0.70 - 0.15 = 0.15


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = _TRAIN_FRAC,
    val_frac:   float = _VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df chronologically into train / val / test.

    No shuffling — we respect temporal order to avoid leakage of future
    information into training.  The validation set is used for early stopping;
    the test set is held out until final evaluation.

    Parameters
    ----------
    df : pd.DataFrame  time-sorted feature matrix
    train_frac, val_frac : float

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrames
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_train + n_val]
    df_test  = df.iloc[n_train + n_val:]

    logger.info(
        f"Split:  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}  "
        f"({100*train_frac:.0f}/{100*val_frac:.0f}/{100*(1-train_frac-val_frac):.0f}%)"
    )
    logger.info(
        f"  Train : {df_train.index.min()} → {df_train.index.max()}"
    )
    logger.info(
        f"  Val   : {df_val.index.min()} → {df_val.index.max()}"
    )
    logger.info(
        f"  Test  : {df_test.index.min()} → {df_test.index.max()}"
    )
    return df_train, df_val, df_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 24h-ahead XGBoost DMS PV forecaster."
    )
    parser.add_argument("--config",    default="configs/site.yaml")
    parser.add_argument("--no-plots",  action="store_true",
                        help="Skip diagnostic plot generation (faster).")
    parser.add_argument("--matrix",    default=None,
                        help="Path to feature_matrix.parquet (overrides config path).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Site: {cfg['site']['name']}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    processed_dir   = resolve_path(cfg["paths"]["processed"])
    models_dir      = Path("results/models")
    metrics_dir     = resolve_path(cfg["paths"]["metrics"])
    pred_dir        = Path("results/predictions")
    fig_dir         = resolve_path(cfg["paths"]["figures"])

    for d in [models_dir, metrics_dir, pred_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load feature matrix ────────────────────────────────────────────────
    matrix_path = Path(args.matrix) if args.matrix else (processed_dir / "feature_matrix.parquet")
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {matrix_path}\n"
            "Run  python scripts/run_feature_engineering.py  first."
        )

    logger.info(f"Loading feature matrix: {matrix_path} …")
    df = pd.read_parquet(matrix_path)
    df = df.sort_index()
    logger.info(f"  Shape: {df.shape}  ({df.index.min()} → {df.index.max()})")

    # ── 2. Identify feature and target columns ────────────────────────────────
    feature_cols = get_feature_cols(df, target_col="pv_ac_W")
    target_cols  = get_target_cols(24)

    # Guard: ensure all expected columns are present
    missing_feats   = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols  if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats[:5]} …")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    logger.info(f"  Feature cols : {len(feature_cols)}")
    logger.info(f"  Target  cols : {len(target_cols)}")

    # ── 3. Chronological split ────────────────────────────────────────────────
    df_train, df_val, df_test = chronological_split(df)

    X_train = df_train[feature_cols]
    Y_train = df_train[target_cols]
    X_val   = df_val[feature_cols]
    Y_val   = df_val[target_cols]
    X_test  = df_test[feature_cols]
    Y_test  = df_test[target_cols]

    # ── 4. Train XGBoost DMS models ───────────────────────────────────────────
    logger.info("Training XGBoost Direct Multi-Step models …")
    models = train_dms_models(X_train, Y_train, X_val, Y_val)

    # ── 5. Compute XGBoost predictions on test set ───────────────────────────
    logger.info("Generating test-set predictions …")
    pred_xgb = predict_dms(models, X_test)

    # Rename pred columns to match target prefix for metrics
    # (target_h1 → target_h1, pred_h1 → pred_h1 — already aligned)

    # ── 6. Baseline forecasts ─────────────────────────────────────────────────
    logger.info("Computing baseline forecasts …")
    pred_pers = day_ahead_persistence(df_test, target_col="pv_ac_W")
    pred_clim = climatological_mean(df_train, df_test, target_col="pv_ac_W")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating forecasts …")
    metrics_xgb  = compute_metrics(Y_test, pred_xgb)
    metrics_pers = compute_metrics(Y_test, pred_pers)
    metrics_clim = compute_metrics(Y_test, pred_clim)

    summarise_metrics(metrics_xgb,  label="XGBoost")
    summarise_metrics(metrics_pers, label="Persistence")
    summarise_metrics(metrics_clim, label="Climatology")

    # ── 8. Save models ────────────────────────────────────────────────────────
    save_models(models, models_dir)

    # ── 9. Save metrics ───────────────────────────────────────────────────────
    metrics_xgb.to_csv(metrics_dir / "metrics_xgboost.csv")
    metrics_pers.to_csv(metrics_dir / "metrics_persistence.csv")
    metrics_clim.to_csv(metrics_dir / "metrics_climatology.csv")
    logger.info(f"Saved metrics → {metrics_dir}")

    # ── 10. Save predictions ──────────────────────────────────────────────────
    # Combine actual + predictions into one parquet for analysis
    test_out = pd.concat(
        [
            df_test[["pv_ac_W"] + target_cols],
            pred_xgb.rename(columns=lambda c: c.replace("pred_h", "xgb_h")),
            pred_pers.rename(columns=lambda c: c.replace("pred_h", "pers_h")),
            pred_clim.rename(columns=lambda c: c.replace("pred_h", "clim_h")),
        ],
        axis=1,
    )
    pred_path = pred_dir / "test_predictions.parquet"
    test_out.to_parquet(pred_path)
    logger.info(f"Saved predictions → {pred_path}")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("Generating diagnostic plots …")

        plot_metrics_vs_horizon(
            {
                "XGBoost":     metrics_xgb,
                "Persistence": metrics_pers,
                "Climatology": metrics_clim,
            },
            fig_dir,
        )

        plot_scatter_horizons(Y_test, pred_xgb, fig_dir)

        imp_summary, imp_by_horizon = get_feature_importance(models, feature_cols)
        plot_feature_importance(imp_summary, fig_dir)
        plot_importance_by_horizon(imp_by_horizon, fig_dir)

        plot_error_by_hour(Y_test, pred_xgb, fig_dir)

        plot_sample_days(df_test, pred_xgb, target_col="pv_ac_W", fig_dir=fig_dir)

        plot_forecast_timeseries(pred_dir / "test_predictions.parquet", fig_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Training Summary ──────────────────────────────────────────────────")
    print(f"  Feature cols    : {len(feature_cols)}")
    print(f"  Train rows      : {len(df_train):,}  ({df_train.index.min().date()} → {df_train.index.max().date()})")
    print(f"  Val   rows      : {len(df_val):,}")
    print(f"  Test  rows      : {len(df_test):,}  ({df_test.index.min().date()} → {df_test.index.max().date()})")
    print()

    # Key horizons
    print("  XGBoost vs Baselines — test RMSE (kW):")
    print(f"  {'Horizon':>8}  {'XGBoost':>9}  {'Persist.':>9}  {'Climatol.':>10}  {'Improve vs Pers':>16}")
    for h in [1, 3, 6, 12, 24]:
        if h not in metrics_xgb.index:
            continue
        xgb_rmse  = metrics_xgb.loc[h, "RMSE_W"] / 1000
        pers_rmse = metrics_pers.loc[h, "RMSE_W"] / 1000 if h in metrics_pers.index else float("nan")
        clim_rmse = metrics_clim.loc[h, "RMSE_W"] / 1000 if h in metrics_clim.index else float("nan")
        improve   = (pers_rmse - xgb_rmse) / pers_rmse * 100 if not np.isnan(pers_rmse) else float("nan")
        print(
            f"  h={h:>2}         {xgb_rmse:9.2f}  {pers_rmse:9.2f}  {clim_rmse:10.2f}"
            f"  {improve:+14.1f}%"
        )

    print()
    mean_r2  = metrics_xgb["R2"].mean()
    mean_nrmse = metrics_xgb["nRMSE_pct"].mean()
    print(f"  Mean XGBoost R²  (h=1..24): {mean_r2:.4f}")
    print(f"  Mean nRMSE       (h=1..24): {mean_nrmse:.2f}%")
    print()
    print(f"  Models saved → {models_dir}/")
    print(f"  Metrics saved → {metrics_dir}/")
    if not args.no_plots:
        print(f"  Plots  saved → {fig_dir}/")


if __name__ == "__main__":
    main()
