"""
src/evaluation/metrics.py
--------------------------
Forecast evaluation metrics for 24h-ahead PV power.

Metrics computed
----------------
  RMSE  — root mean squared error  [W]
  MAE   — mean absolute error       [W]
  MBE   — mean bias error           [W]  (positive = over-prediction)
  MAPE  — mean absolute % error     [%]  (daytime only, avoids division by zero)
  nRMSE — RMSE normalised by mean observed power  [%]  (scale-free comparison)
  R²    — coefficient of determination (Pearson r²)

All metrics are computed per-horizon (h=1…24) and returned as a DataFrame.

Usage
-----
    from src.evaluation.metrics import compute_metrics, summarise_metrics

    metrics_df = compute_metrics(y_true_df, y_pred_df)
    summarise_metrics(metrics_df)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum observed power to include in MAPE computation (avoid / zero)
_MAPE_MIN_W: float = 1_000.0


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
) -> dict:
    """
    Compute all metrics for a single forecast horizon.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  (1-D, same length, no NaN)
    horizon : int

    Returns
    -------
    dict with keys: horizon, n, RMSE, MAE, MBE, MAPE, nRMSE, R2
    """
    err   = y_pred - y_true
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    mae   = float(np.mean(np.abs(err)))
    mbe   = float(np.mean(err))

    # MAPE — daytime only
    day_mask = y_true >= _MAPE_MIN_W
    if day_mask.sum() > 0:
        mape = float(np.mean(np.abs(err[day_mask]) / y_true[day_mask]) * 100)
    else:
        mape = float("nan")

    mean_obs = float(y_true.mean())
    nrmse = (rmse / mean_obs * 100) if mean_obs > 0 else float("nan")
    r2    = _safe_r2(y_true, y_pred)

    return {
        "horizon": horizon,
        "n":       int(len(y_true)),
        "RMSE_W":  round(rmse, 1),
        "MAE_W":   round(mae, 1),
        "MBE_W":   round(mbe, 1),
        "MAPE_pct": round(mape, 2),
        "nRMSE_pct": round(nrmse, 2),
        "R2":      round(r2, 4),
    }


def compute_metrics(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    target_prefix: str = "target_h",
    pred_prefix:   str = "pred_h",
    n_horizons:    int = 24,
) -> pd.DataFrame:
    """
    Compute per-horizon metrics for all 24 horizons.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        Must contain columns {target_prefix}1 … {target_prefix}{n_horizons}.
    y_pred_df : pd.DataFrame
        Must contain columns {pred_prefix}1 … {pred_prefix}{n_horizons}.
    target_prefix, pred_prefix : str
    n_horizons : int

    Returns
    -------
    pd.DataFrame  with one row per horizon and columns:
        horizon, n, RMSE_W, MAE_W, MBE_W, MAPE_pct, nRMSE_pct, R2
    """
    rows = []
    for h in range(1, n_horizons + 1):
        t_col = f"{target_prefix}{h}"
        p_col = f"{pred_prefix}{h}"

        if t_col not in y_true_df.columns or p_col not in y_pred_df.columns:
            continue

        # Align and drop NaN
        df = pd.DataFrame({
            "y": y_true_df[t_col],
            "p": y_pred_df[p_col],
        }).dropna()

        if len(df) == 0:
            continue

        rows.append(compute_horizon_metrics(df["y"].values, df["p"].values, h))

    return pd.DataFrame(rows).set_index("horizon")


def summarise_metrics(metrics_df: pd.DataFrame, label: str = "") -> None:
    """
    Log a compact per-horizon metrics table to the console.

    Parameters
    ----------
    metrics_df : pd.DataFrame  output of compute_metrics()
    label : str  prefix for the log header (e.g., model name)
    """
    header = f"── {label} Metrics ──" if label else "── Forecast Metrics ──"
    logger.info(header)
    logger.info(
        f"{'h':>3}  {'RMSE(kW)':>9}  {'MAE(kW)':>8}  {'MBE(kW)':>8}"
        f"  {'MAPE%':>6}  {'nRMSE%':>7}  {'R²':>7}"
    )
    for h, row in metrics_df.iterrows():
        logger.info(
            f"{h:3d}  {row['RMSE_W']/1000:9.2f}  {row['MAE_W']/1000:8.2f}"
            f"  {row['MBE_W']/1000:8.2f}  {row['MAPE_pct']:6.2f}"
            f"  {row['nRMSE_pct']:7.2f}  {row['R2']:7.4f}"
        )

    # Aggregate summary
    logger.info("─" * 60)
    logger.info(
        f"Mean RMSE: {metrics_df['RMSE_W'].mean()/1000:.2f} kW  |  "
        f"Mean MAE: {metrics_df['MAE_W'].mean()/1000:.2f} kW  |  "
        f"Mean R²: {metrics_df['R2'].mean():.4f}"
    )
