"""
src/evaluation/plots.py
------------------------
Diagnostic plots for evaluating the 24h-ahead XGBoost DMS forecaster.

Figures generated
------------------
  1. metrics_vs_horizon.png
     Line chart: RMSE, MAE, nRMSE, R² vs forecast horizon h=1..24 for XGBoost
     and both baselines side by side.

  2. scatter_selected_horizons.png
     Scatter: observed vs predicted for h=1, h=6, h=12, h=24  (2×2 grid).
     Shows prediction quality at short, medium, and long horizons.

  3. feature_importance.png
     Horizontal bar chart: mean feature gain across h=1,6,12,24.
     Top-20 features shown.

  4. error_by_hour.png
     Mean error (MBE) and RMSE by hour-of-day for h=1 and h=24.
     Identifies systematic biases at specific times of day.

  5. forecast_sample_days.png
     Actual vs predicted 24h forecast for 4 representative days:
     clear, partly cloudy, overcast, monsoon.

Usage
-----
    from src.evaluation.plots import (
        plot_metrics_vs_horizon,
        plot_scatter_horizons,
        plot_feature_importance,
        plot_error_by_hour,
        plot_sample_days,
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PALETTE = {"XGBoost": "#2196F3", "Persistence": "#FF9800", "Climatology": "#9C27B0"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics vs horizon
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_vs_horizon(
    metrics_dict: dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    """
    Line chart of RMSE (kW), MAE (kW), nRMSE (%), R² vs horizon h=1..24.

    Parameters
    ----------
    metrics_dict : dict  model_name → metrics DataFrame (output of compute_metrics)
    fig_dir : Path
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    horizons = list(range(1, 25))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Forecast Performance vs Horizon  (h = 1 … 24 hours ahead)\n"
        "XGBoost Direct Multi-Step vs Baselines",
        fontsize=11, fontweight="bold",
    )

    metrics_plot = [
        ("RMSE_W",    "RMSE  [kW]",    1000, False),
        ("MAE_W",     "MAE   [kW]",    1000, False),
        ("nRMSE_pct", "nRMSE  [%]",      1, False),
        ("R2",        "R²",               1, True),
    ]

    for ax, (col, ylabel, divisor, invert) in zip(axes.flat, metrics_plot):
        for name, df in metrics_dict.items():
            if col not in df.columns:
                continue
            vals = df[col] / divisor
            color = _PALETTE.get(name, None)
            lw    = 2.5 if name == "XGBoost" else 1.5
            ls    = "-"  if name == "XGBoost" else "--"
            ax.plot(horizons[:len(vals)], vals.values, lw=lw, ls=ls,
                    color=color, label=name, marker="o", ms=3)

        ax.set_xlabel("Horizon h  (hours ahead)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.legend(fontsize=8)
        if invert:
            ax.set_ylim(top=1.0)

    fig.tight_layout()
    out = fig_dir / "metrics_vs_horizon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scatter at selected horizons
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_horizons(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    fig_dir:   Path,
    horizons:  list[int] | None = None,
) -> None:
    """
    2×2 scatter: observed vs predicted for h = 1, 6, 12, 24.

    Parameters
    ----------
    y_true_df : DataFrame  columns target_h1 … target_h24
    y_pred_df : DataFrame  columns pred_h1   … pred_h24
    fig_dir : Path
    horizons : list[int]  four horizons to plot (default [1,6,12,24])
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    horizons = horizons or [1, 6, 12, 24]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Observed vs Predicted PV Power — XGBoost DMS\n"
        "Daytime hours only  (pv_ac_W > 1 kW)",
        fontsize=11, fontweight="bold",
    )

    for ax, h in zip(axes.flat, horizons):
        t_col = f"target_h{h}"
        p_col = f"pred_h{h}"
        if t_col not in y_true_df.columns or p_col not in y_pred_df.columns:
            ax.set_visible(False)
            continue

        df = pd.DataFrame({"y": y_true_df[t_col], "p": y_pred_df[p_col]}).dropna()
        df = df[df["y"] > 1000]   # daytime only

        # Subsample for plot clarity
        if len(df) > 4000:
            df = df.sample(4000, random_state=42)

        y, p = df["y"].values / 1000, df["p"].values / 1000

        ax.hexbin(y, p, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.2)
        lim = max(y.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")

        # OLS fit
        m, c = np.polyfit(y, p, 1)
        fit_x = np.array([0, lim])
        ax.plot(fit_x, m * fit_x + c, "r-", lw=1.5, alpha=0.8)

        # Metrics
        err  = p - y
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2   = float(1 - np.sum(err**2) / np.sum((y - y.mean())**2))

        ax.text(0.04, 0.96,
                f"h = {h}h ahead\nRMSE = {rmse:.1f} kW\nR² = {r2:.3f}\nn = {len(df):,}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Observed  [kW]", fontsize=9)
        ax.set_ylabel("Predicted  [kW]", fontsize=9)
        ax.set_title(f"h = {h} hours ahead", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = fig_dir / "scatter_selected_horizons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    Horizontal bar chart of mean feature gain (top-20).

    Parameters
    ----------
    importance_df : DataFrame  output of gradient_boost.get_feature_importance()
    fig_dir : Path
    """
    if importance_df.empty:
        logger.warning("Feature importance DataFrame is empty — skipping plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=0.9)

    df = importance_df.sort_values("mean_gain", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("Blues_d", len(df))
    bars = ax.barh(df["feature"], df["mean_gain"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df["mean_gain"]):
        ax.text(bar.get_width() + df["mean_gain"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5)

    ax.set_xlim(0, df["mean_gain"].max() * 1.18)
    ax.set_xlabel("Mean Gain  (averaged across h = 1, 6, 12, 24)", fontsize=10)
    ax.set_title(
        "Top-20 Feature Importances — XGBoost DMS\n"
        "Higher gain = larger contribution to prediction accuracy",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = fig_dir / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Feature importance heatmap by horizon
# ─────────────────────────────────────────────────────────────────────────────

def plot_importance_by_horizon(
    importance_by_horizon: pd.DataFrame,
    fig_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Heatmap: top-N features × h=1..24 showing how gain shifts across horizons.

    Answers: "Does the model rely on lag features at h=24, or do weather/time
    features take over?"  Each column is normalised 0→1 (per-horizon max=1)
    so relative importance is visible despite different absolute gain scales.

    Features are colour-grouped by type:
      - Lag/rolling/diff → left of dashed divider
      - Weather/solar/time → right

    Parameters
    ----------
    importance_by_horizon : pd.DataFrame
        Full gain matrix, index=feature_cols, columns=h1..h24.
        Output of get_feature_importance()[1].
    fig_dir : Path
    top_n : int  top features by mean gain across all horizons. Default 15.
    """
    if importance_by_horizon.empty:
        logger.warning("importance_by_horizon is empty — skipping heatmap.")
        return

    sns.set_theme(style="white", font_scale=0.85)

    # Select top_n features by mean gain across all horizons
    top_features = (
        importance_by_horizon
        .mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    # Column-normalise so each horizon's max = 1
    heat = importance_by_horizon.loc[top_features].copy()
    col_max = heat.max(axis=0).replace(0, 1)
    heat_norm = heat.div(col_max, axis=1)

    # Rename columns h1→1, h2→2 etc. for cleaner x-axis
    heat_norm.columns = [int(c.replace("h", "")) for c in heat_norm.columns]
    heat_norm = heat_norm[sorted(heat_norm.columns)]

    # Classify features for divider line
    lag_types  = {"_lag", "_roll", "_diff"}
    is_lag = [any(t in f for t in lag_types) for f in top_features]
    n_lag  = sum(is_lag)   # number of lag-type features in top_n

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.45)))
    sns.heatmap(
        heat_norm,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Normalised gain  (per-horizon max = 1)", "shrink": 0.6},
    )

    # Dashed horizontal line separating lag features from weather/time features
    if 0 < n_lag < top_n:
        ax.axhline(n_lag, color="#333", lw=1.5, ls="--", alpha=0.7)
        ax.text(
            heat_norm.shape[1] + 0.2, n_lag - 0.4,
            "← lag / rolling", fontsize=7.5, color="#333", va="center",
        )
        ax.text(
            heat_norm.shape[1] + 0.2, n_lag + 0.4,
            "← weather / time", fontsize=7.5, color="#333", va="center",
        )

    ax.set_xlabel("Forecast horizon  h  (hours ahead)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_title(
        f"Top-{top_n} Feature Gain by Horizon  (h = 1 … 24)\n"
        "Warm = high gain. Shows whether weather features dominate at longer horizons.",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    out = Path(fig_dir) / "feature_importance_by_horizon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Error by hour of day
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_by_hour(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    fig_dir:   Path,
    horizons:  list[int] | None = None,
) -> None:
    """
    Mean Bias Error and RMSE grouped by UTC hour, for h=1 and h=24.

    Shows whether specific hours of day have systematic biases
    (e.g., morning ramp-up underestimation, afternoon overestimation).

    Parameters
    ----------
    y_true_df, y_pred_df : DataFrames with target_h*/pred_h* columns
    fig_dir : Path
    horizons : list[int]  default [1, 24]
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    horizons = horizons or [1, 24]

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5),
                              sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle(
        "Forecast Bias by Hour of Day  (UTC)\n"
        "Positive = over-prediction  |  Negative = under-prediction",
        fontsize=11, fontweight="bold",
    )

    for ax, h in zip(axes, horizons):
        t_col = f"target_h{h}"
        p_col = f"pred_h{h}"
        if t_col not in y_true_df.columns or p_col not in y_pred_df.columns:
            continue

        df = pd.DataFrame({
            "y": y_true_df[t_col],
            "p": y_pred_df[p_col],
        }, index=y_true_df.index).dropna()

        # Hour of the TARGET timestamp (index + h hours)
        target_hour = (df.index + pd.Timedelta(hours=h)).hour
        df["target_hour"] = target_hour
        df["err"]  = (df["p"] - df["y"]) / 1000      # kW
        df["abserr"] = df["err"].abs()

        mbe_h  = df.groupby("target_hour")["err"].mean()
        rmse_h = df.groupby("target_hour")["abserr"].apply(
            lambda x: float(np.sqrt((x**2).mean()))
        )

        hrs = mbe_h.index
        ax.bar(hrs, mbe_h.values, color=["#f44336" if v > 0 else "#2196F3"
                                          for v in mbe_h.values],
               alpha=0.75, label="MBE (kW)")
        ax.plot(hrs, rmse_h.values, "k-o", ms=4, lw=1.5, label="RMSE (kW)")
        ax.axhline(0, color="black", lw=0.7, alpha=0.5)
        ax.set_xlabel("Hour of Day (UTC)  [local noon ≈ 06:30]", fontsize=9)
        ax.set_ylabel("Power  [kW]", fontsize=9)
        ax.set_title(f"h = {h} hours ahead", fontsize=10, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = fig_dir / "error_by_hour.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sample forecast days
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_days(
    df_test:      pd.DataFrame,
    y_pred_df:    pd.DataFrame,
    target_col:   str = "pv_ac_W",
    fig_dir:      Path | None = None,
    sample_dates: list[str] | None = None,
) -> None:
    """
    Show actual 24h PV profile vs 24h XGBoost forecast for 4 sample days.

    For each sample date d, uses the forecast issued at 00:00 UTC on that date
    (i.e., row with index == d 00:00 UTC), and reads pred_h1…pred_h24 to get
    the 24-hour forecast trace. The actual trace is taken from pv_ac_W at
    d+1h … d+24h.

    Parameters
    ----------
    df_test : DataFrame  must contain target_col column, UTC index
    y_pred_df : DataFrame  pred_h1…pred_h24, same index as df_test
    target_col : str
    fig_dir : Path
    sample_dates : list of YYYY-MM-DD strings  (4 dates)
        Default: auto-selected as clear/cloudy/overcast/monsoon days from test set.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    if sample_dates is None:
        sample_dates = _auto_select_sample_days(df_test, target_col)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "24h-Ahead XGBoost Forecast vs Observed — Representative Days\n"
        "Forecast issued at 00:00 UTC; horizon h=1…24",
        fontsize=11, fontweight="bold",
    )

    titles = ["Clear Day", "Partly Cloudy", "Overcast / Monsoon", "Best Available"]

    for ax, date_str, title in zip(axes.flat, sample_dates, titles):
        try:
            issue_ts = pd.Timestamp(date_str, tz="UTC")
        except Exception:
            ax.set_visible(False)
            continue

        if issue_ts not in y_pred_df.index:
            # Find nearest available forecast timestamp
            diffs = np.abs((y_pred_df.index - issue_ts).total_seconds())
            issue_ts = y_pred_df.index[diffs.argmin()]

        if issue_ts not in y_pred_df.index:
            ax.set_visible(False)
            continue

        # Forecast trace: pred_h1…pred_h24 issued at issue_ts
        forecast_hours = pd.date_range(
            issue_ts + pd.Timedelta(hours=1),
            periods=24, freq="1h", tz="UTC",
        )
        pred_vals = y_pred_df.loc[issue_ts, [f"pred_h{h}" for h in range(1, 25)]].values / 1000

        # Actual trace
        actual_vals = df_test[target_col].reindex(forecast_hours).values / 1000

        ax.plot(range(1, 25), actual_vals, "o-", color="steelblue", lw=2, ms=5,
                label="Observed")
        ax.plot(range(1, 25), pred_vals, "s--", color="darkorange", lw=2, ms=5,
                label="XGBoost forecast")
        ax.fill_between(range(1, 25), actual_vals, pred_vals, alpha=0.12, color="gray")

        # Error metrics for this day
        mask = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        if mask.sum() > 0:
            rmse_d = float(np.sqrt(np.mean((pred_vals[mask] - actual_vals[mask])**2)))
            ax.text(0.97, 0.97, f"RMSE = {rmse_d:.1f} kW",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_title(f"{title}\n{date_str}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Forecast hour (h)", fontsize=8)
        ax.set_ylabel("PV Power  [kW]", fontsize=8)
        ax.set_xticks(range(1, 25, 2))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

    fig.tight_layout()
    if fig_dir:
        out = Path(fig_dir) / "forecast_sample_days.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {out}")


def _auto_select_sample_days(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Auto-select 4 representative forecast-issue dates from df:
    clear, partly cloudy, overcast, plus one extra.
    """
    daily_max = df[target_col].resample("1D").max().dropna()
    if len(daily_max) < 4:
        return [str(daily_max.index[0].date())] * 4

    q75  = daily_max.quantile(0.75)
    q50  = daily_max.quantile(0.50)
    q25  = daily_max.quantile(0.25)
    q10  = daily_max.quantile(0.10)

    clear    = daily_max[daily_max >= q75].index[0]
    partial  = daily_max[(daily_max >= q50) & (daily_max < q75)].index[0]
    overcast = daily_max[(daily_max >= q25) & (daily_max < q50)].index[0]
    monsoon  = daily_max[daily_max <= q10].index[0]

    return [
        str(clear.date()),
        str(partial.date()),
        str(overcast.date()),
        str(monsoon.date()),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Continuous forecast time series (full test period)
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_timeseries(
    test_predictions_path: Path,
    fig_dir: Path,
    horizons: list[int] | None = None,
) -> None:
    """
    Continuous actual vs predicted time series over the full test period.

    Three stacked subplots — one per forecast horizon (h=1, h=6, h=24).
    Each panel shows:
      - Actual PV power (blue)
      - XGBoost prediction (orange)
    Only daytime hours are plotted (target > 0 W) to avoid cluttering with zeros.

    A fourth panel below shows the monthly mean RMSE for h=1 and h=24 as
    a bar chart — useful for seeing which months are hardest to forecast.

    Parameters
    ----------
    test_predictions_path : Path
        Path to test_predictions.parquet produced by run_training.py.
    fig_dir : Path
    horizons : list[int]  default [1, 6, 24]
    """
    horizons = horizons or [1, 6, 24]
    test_predictions_path = Path(test_predictions_path)

    if not test_predictions_path.exists():
        logger.warning(f"  {test_predictions_path} not found — skipping timeseries plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=0.9)
    df = pd.read_parquet(test_predictions_path)

    n_panels = len(horizons) + 1   # forecast panels + monthly RMSE panel
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(18, 4 * n_panels),
        sharex=False,
        gridspec_kw={"height_ratios": [2] * len(horizons) + [1.5]},
    )
    fig.suptitle(
        "XGBoost 24h-Ahead Forecast vs Observed — Full Test Period\n"
        "Daytime hours only  (pv_ac_W > 0 W)",
        fontsize=12, fontweight="bold",
    )

    for ax, h in zip(axes[:len(horizons)], horizons):
        t_col   = f"target_h{h}"
        xgb_col = f"xgb_h{h}"

        if t_col not in df.columns or xgb_col not in df.columns:
            ax.set_visible(False)
            continue

        # Align: the prediction xgb_h{h} at row t is for timestamp t+h.
        # Shift both forward by h so the x-axis represents the *target* time.
        actual = df[t_col].shift(-h)
        pred   = df[xgb_col].shift(-h)

        # Daytime mask on the TARGET value
        day_mask = actual > 0
        actual_day = actual[day_mask] / 1000   # kW
        pred_day   = pred[day_mask]   / 1000

        ax.plot(actual_day.index, actual_day.values,
                color="#1565C0", lw=0.7, alpha=0.85, label="Observed")
        ax.plot(pred_day.index, pred_day.values,
                color="#E65100", lw=0.7, alpha=0.75, label=f"XGBoost h={h}")

        # Error shading
        ax.fill_between(
            actual_day.index,
            actual_day.values,
            pred_day.values,
            where=(pred_day.values > actual_day.values),
            alpha=0.12, color="#E65100", label="Over-prediction",
        )
        ax.fill_between(
            actual_day.index,
            actual_day.values,
            pred_day.values,
            where=(pred_day.values <= actual_day.values),
            alpha=0.12, color="#1565C0", label="Under-prediction",
        )

        rmse_all = float(np.sqrt(np.nanmean((pred_day.values - actual_day.values) ** 2)))
        r2_all = float(
            1 - np.nansum((pred_day.values - actual_day.values) ** 2)
              / np.nansum((actual_day.values - np.nanmean(actual_day.values)) ** 2)
        )
        ax.text(
            0.01, 0.97,
            f"h = {h}h ahead  |  RMSE = {rmse_all:.1f} kW  |  R² = {r2_all:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88),
        )
        ax.set_ylabel("PV Power  [kW]", fontsize=9)
        ax.set_title(f"h = {h} hours ahead", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=20, labelsize=7)

    # ── Monthly RMSE panel ────────────────────────────────────────────────────
    ax_rmse = axes[-1]
    palette = {"h1": "#1976D2", "h24": "#E64A19"}
    month_labels_short = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

    for h_bar, color in [(1, palette["h1"]), (24, palette["h24"])]:
        t_col   = f"target_h{h_bar}"
        xgb_col = f"xgb_h{h_bar}"
        if t_col not in df.columns or xgb_col not in df.columns:
            continue
        err = (df[xgb_col] - df[t_col]).dropna()
        err = err[df[t_col] > 0]   # daytime only
        monthly_rmse = (
            err.groupby(err.index.month)
               .apply(lambda x: float(np.sqrt((x**2).mean())))
            / 1000  # kW
        )
        months = monthly_rmse.index
        offset = -0.2 if h_bar == 1 else 0.2
        ax_rmse.bar(
            months + offset, monthly_rmse.values,
            width=0.38, color=color, alpha=0.85,
            label=f"RMSE h={h_bar}",
        )

    ax_rmse.set_xlabel("Month", fontsize=9)
    ax_rmse.set_ylabel("RMSE  [kW]", fontsize=9)
    ax_rmse.set_title(
        "Monthly RMSE — h=1 vs h=24  (higher = harder month to forecast)",
        fontsize=9, fontweight="bold",
    )
    ax_rmse.set_xticks(range(1, 13))
    ax_rmse.set_xticklabels(month_labels_short, fontsize=8)
    ax_rmse.legend(fontsize=8)

    fig.tight_layout()
    out = Path(fig_dir) / "forecast_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")
