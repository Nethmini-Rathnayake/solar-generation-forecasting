"""
scripts/run_ml_cmv.py
──────────────────────────────────────────────────────────────────────────────
CMV-augmented ML pipeline: measures the forecast improvement from adding
wind-based Cloud Motion Vector features to the 4-year model.

Runs each model twice — without and with CMV features — then reports the
Δ metrics so it's clear what the cloud-tracking features contribute.

Models
──────
  XGBoost   — gradient boosted trees (handles NaN natively)
  GRU       — recurrent, 30-min lookback
  CNN-GRU   — convolutional + recurrent, 1-hour lookback

CMV features added (14 columns)
────────────────────────────────
  cloud_speed_kmh           — 850 hPa equivalent wind speed [km/h]
  cloud_direction_deg       — direction clouds move toward [°]
  shadow_offset_km          — solar geometry shadow displacement [km]
  solar_zenith_deg          — solar zenith [°]
  shadow_arrival_{5,10,20,40}km  — minutes until shadow arrives
  opacity_lag_{5,10,20,40}km     — cloud opacity at upstream location
  site_cloud_opacity        — current Solcast cloud opacity [0–100]
  cloud_opacity_trend       — 30-min change in cloud opacity

Run
───
    python scripts/run_ml_cmv.py
    python scripts/run_ml_cmv.py --xgb-only      # fast test (~5 min)
    python scripts/run_ml_cmv.py --no-baseline    # skip re-running baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse all shared helpers from run_ml_4yr
from scripts.run_ml_4yr import (
    build_4yr_labels,
    build_feature_matrix,
    split_4yr,
    train_xgboost,
    _metrics,
    _GRUModel,
    _CNNLSTMModel,
)
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.data.solcast_loader import load_solcast_local_files, solcast_to_nasa_schema
from src.features.time_features import add_time_features
from src.features.weather_patterns import (
    add_weather_pattern_features,
    add_pv_lag_features_5min,
)

logger = get_logger("run_ml_cmv")

_OUT_DIR = Path("results/figures/ml_cmv")
_MET_DIR = Path("results/metrics/ml_cmv")
_MOD_DIR = Path("results/models/ml_cmv")

# CMV feature columns to add
_CMV_FEATURES = [
    "cloud_speed_kmh",
    "cloud_direction_deg",
    "shadow_offset_km",
    "solar_zenith_deg",
    "shadow_arrival_5km",
    "shadow_arrival_10km",
    "shadow_arrival_20km",
    "shadow_arrival_40km",
    "opacity_lag_5km",
    "opacity_lag_10km",
    "opacity_lag_20km",
    "opacity_lag_40km",
    "site_cloud_opacity",
    "cloud_opacity_trend",
]


# ─────────────────────────────────────────────────────────────────────────────
# Load and attach CMV features
# ─────────────────────────────────────────────────────────────────────────────

def load_cmv_features() -> pd.DataFrame:
    """Load wind CMV features (5-min, UTC index)."""
    path = Path("data/interim/wind_cmv_features_5min.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"CMV features not found at {path}.\n"
            "Run:  python scripts/run_wind_cmv_pipeline.py"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"  CMV features: {df.shape}  "
                f"({df.index.min().date()} → {df.index.max().date()})")
    return df


def attach_cmv(feat: pd.DataFrame, cmv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge CMV features into feature matrix, return (augmented_df, cmv_cols_added).

    CMV features are available only for Apr 2022 – Mar 2023.
    NaNs in the training period (2020–Apr 2022) are filled with the
    monthly-hour climatological median computed from the CMV-available period.
    This lets the model learn from CMV features even on synthetic rows.

    Columns that track instantaneous state (shadow_arrival, opacity_lag) are
    filled this way; cloud_speed/direction use monthly-hour medians too.
    """
    avail = [c for c in _CMV_FEATURES if c in cmv.columns]
    new   = [c for c in avail if c not in feat.columns]

    feat_cmv = feat.join(cmv[new], how="left")
    n_orig = feat_cmv[new[0]].notna().sum() if new else 0

    # Build climatological fill table: median by (month, hour) over CMV period
    cmv_period = cmv[new].copy()
    cmv_period["month"] = cmv_period.index.month
    cmv_period["hour"]  = cmv_period.index.hour
    clim = cmv_period.groupby(["month", "hour"])[new].median()

    # Fill NaN rows in feat_cmv
    nan_mask = feat_cmv[new[0]].isna()
    logger.info(f"  Filling {nan_mask.sum():,} NaN rows with climatological medians …")

    for col in new:
        nan_idx = feat_cmv.index[feat_cmv[col].isna()]
        months  = nan_idx.month
        hours   = nan_idx.hour
        fills   = [
            clim.loc[(m, h), col] if (m, h) in clim.index else np.nan
            for m, h in zip(months, hours)
        ]
        feat_cmv.loc[nan_idx, col] = fills

    # Mark filled rows (so model can learn "this is imputed")
    feat_cmv["cmv_is_observed"] = (~nan_mask).astype(np.float32)

    n_filled = feat_cmv[new[0]].notna().sum() if new else 0
    logger.info(
        f"  Attached {len(new)+1} CMV columns  "
        f"(observed={n_orig:,}  clim-filled={n_filled-n_orig:,}  "
        f"still-NaN={(feat_cmv[new[0]].isna()).sum():,})"
    )
    return feat_cmv, new + ["cmv_is_observed"]


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment (with or without CMV features)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    feat:         pd.DataFrame,
    feature_cols: list[str],
    physics_sim:  pd.Series,
    label_suffix: str,
    xgb_only:     bool = False,
) -> dict:
    """
    Train XGBoost (and optionally GRU + CNN-GRU), return metric dict.

    Returns
    -------
    dict with keys: xgb, gru, cnnlstm  (each a metrics dict from _metrics)
    """
    train, val, test = split_4yr(feat)
    obs_test = test["pv_ac_kW"].values
    phy_test = (physics_sim.reindex(test.index) / 1000).clip(lower=0).fillna(0).values

    def _prepend(block, source, n):
        return pd.concat([source.iloc[-n:], block])

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info(f"\n── XGBoost [{label_suffix}] ─────────────────────────────────")
    # XGBoost handles NaN natively — use CMV columns directly
    xgb_feat_cols = feature_cols   # already includes CMV for CMV run

    # For GRU/CNN-GRU, remaining NaNs (nighttime shadow_arrival) → 0
    feat_imputed = feat.copy()
    cmv_in_feat  = [c for c in _CMV_FEATURES + ["cmv_is_observed"]
                    if c in feat_imputed.columns]
    if cmv_in_feat:
        feat_imputed[cmv_in_feat] = feat_imputed[cmv_in_feat].fillna(0.0)

    train_imp, val_imp, test_imp = split_4yr(feat_imputed)

    xgb_model = train_xgboost(train, val, xgb_feat_cols, label=f"XGBoost [{label_suffix}]")
    pred_xgb  = xgb_model.predict(test[xgb_feat_cols].values).clip(min=0)
    xgb_model.save_model(str(_MOD_DIR / f"xgb_{label_suffix}.json"))
    r_xgb = _metrics(obs_test, pred_xgb, f"XGBoost [{label_suffix}]")

    if xgb_only:
        return {"xgb": r_xgb, "gru": None, "cnnlstm": None,
                "test": test, "obs": obs_test, "pred_xgb": pred_xgb,
                "pred_gru": pred_xgb, "pred_cnnlstm": pred_xgb, "phy": phy_test}

    # ── GRU ───────────────────────────────────────────────────────────────────
    logger.info(f"\n── GRU [{label_suffix}] ──────────────────────────────────────")
    lookback_gru = 6
    X_tr = train_imp[feature_cols].values.astype(np.float32)
    y_tr = train_imp["pv_ac_kW"].values.astype(np.float32)
    va_df = _prepend(val_imp, train_imp, lookback_gru)
    X_va  = va_df[feature_cols].values.astype(np.float32)
    y_va  = va_df["pv_ac_kW"].values.astype(np.float32)
    te_df = _prepend(test_imp, val_imp, lookback_gru)
    X_te  = te_df[feature_cols].values.astype(np.float32)

    gru = _GRUModel(hidden=64)
    gru.fit(X_tr, y_tr, X_va, y_va, lookback=lookback_gru)
    gru_raw  = gru.predict(X_te)
    pred_gru = gru_raw[lookback_gru:][:len(test)]
    pred_gru = np.where(np.isnan(pred_gru), pred_xgb, pred_gru)
    r_gru    = _metrics(obs_test, pred_gru, f"GRU [{label_suffix}]")

    # ── CNN-GRU ───────────────────────────────────────────────────────────────
    logger.info(f"\n── CNN-GRU [{label_suffix}] ──────────────────────────────────")
    lookback_cnn = 12
    t_cnn_val_start = pd.Timestamp("2022-12-01", tz="UTC")
    t_cnn_test      = pd.Timestamp("2023-02-01", tz="UTC")
    train_cnn = feat_imputed[
        (feat_imputed["is_actual"] == 1) & (feat_imputed.index < t_cnn_val_start)]
    val_cnn   = feat_imputed[
        (feat_imputed["is_actual"] == 1) &
        (feat_imputed.index >= t_cnn_val_start) & (feat_imputed.index < t_cnn_test)]
    logger.info(f"  train_cnn: {len(train_cnn):,}  val_cnn: {len(val_cnn):,}")

    va_df_c = _prepend(val_cnn, train_cnn, lookback_cnn)
    te_df_c = _prepend(test_imp, val_cnn, lookback_cnn)
    X_va_c  = va_df_c[feature_cols].values.astype(np.float32)
    y_va_c  = va_df_c["pv_ac_kW"].values.astype(np.float32)
    X_te_c  = te_df_c[feature_cols].values.astype(np.float32)
    X_tr_c  = train_cnn[feature_cols].values.astype(np.float32)
    y_tr_c  = train_cnn["pv_ac_kW"].values.astype(np.float32)

    cnnlstm = _CNNLSTMModel(cnn_filters=64, lstm_hidden=64)
    cnnlstm.fit(X_tr_c, y_tr_c, X_va_c, y_va_c,
                lookback=lookback_cnn, epochs=80, patience=15)
    cnn_raw      = cnnlstm.predict(X_te_c)
    pred_cnnlstm = cnn_raw[lookback_cnn:][:len(test)]
    pred_cnnlstm = np.where(np.isnan(pred_cnnlstm), pred_xgb, pred_cnnlstm)
    r_cnn = _metrics(obs_test, pred_cnnlstm, f"CNN-GRU [{label_suffix}]")

    return {
        "xgb":        r_xgb,
        "gru":        r_gru,
        "cnnlstm":    r_cnn,
        "test":       test,
        "obs":        obs_test,
        "pred_xgb":   pred_xgb,
        "pred_gru":   pred_gru,
        "pred_cnnlstm": pred_cnnlstm,
        "phy":        phy_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_cmv_comparison(
    base: dict,
    cmv:  dict,
    feature_cols_base: list[str],
    feature_cols_cmv:  list[str],
    xgb_model_cmv,
) -> None:
    import matplotlib.dates as mdates

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Δ metrics bar chart ────────────────────────────────────────────
    models = ["XGBoost", "GRU", "CNN-GRU"]
    base_r2   = [base["xgb"]["R2"], base["gru"]["R2"] if base["gru"] else np.nan,
                 base["cnnlstm"]["R2"] if base["cnnlstm"] else np.nan]
    cmv_r2    = [cmv["xgb"]["R2"],  cmv["gru"]["R2"]  if cmv["gru"]  else np.nan,
                 cmv["cnnlstm"]["R2"] if cmv["cnnlstm"] else np.nan]
    base_rmse = [base["xgb"]["RMSE_kW"], base["gru"]["RMSE_kW"] if base["gru"] else np.nan,
                 base["cnnlstm"]["RMSE_kW"] if base["cnnlstm"] else np.nan]
    cmv_rmse  = [cmv["xgb"]["RMSE_kW"],  cmv["gru"]["RMSE_kW"]  if cmv["gru"]  else np.nan,
                 cmv["cnnlstm"]["RMSE_kW"] if cmv["cnnlstm"] else np.nan]

    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    b1 = ax.bar(x - w/2, base_r2, w, label="Baseline", color="#455A64", alpha=0.85)
    b2 = ax.bar(x + w/2, cmv_r2,  w, label="+ CMV",    color="#26A69A", alpha=0.85)
    for bar in b2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#26A69A")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("R² — Baseline vs + CMV Features", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - w/2, base_rmse, w, label="Baseline", color="#455A64", alpha=0.85)
    b3 = ax.bar(x + w/2, cmv_rmse,  w, label="+ CMV",    color="#26A69A", alpha=0.85)
    for bar in b3:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#26A69A")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("RMSE [kW]", fontsize=10)
    ax.set_title("RMSE — Baseline vs + CMV Features", fontsize=10, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "CMV Feature Impact on Forecast Skill\nTest set: Feb–Mar 2023",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    p = _OUT_DIR / "01_cmv_metric_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {p}")

    # ── Fig 2: XGBoost CMV feature importance ────────────────────────────────
    score  = xgb_model_cmv.get_booster().get_score(importance_type="gain")
    imp    = {feature_cols_cmv[int(k[1:])]: v for k, v in score.items()
              if int(k[1:]) < len(feature_cols_cmv)}
    top25  = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:25]
    names  = [k for k, _ in top25]
    vals   = [v for _, v in top25]
    cmv_set = set(_CMV_FEATURES)
    colors  = ["#26A69A" if n in cmv_set else
               "#e63946" if any(p in n for p in ["kt","ghi","cloud","clearness","sky","diffuse","physics"])
               else "#457b9d" for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost + CMV Feature Importance (Top 25)\n"
                 "Teal = CMV features  |  Red = weather/physics  |  Blue = time/lag",
                 fontsize=11)
    p = _OUT_DIR / "02_cmv_feature_importance.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {p}")

    # ── Fig 3: Residual comparison (XGBoost base vs +CMV) ────────────────────
    test = cmv["test"]
    obs  = cmv["obs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pred, label, color in [
        (axes[0], base["pred_xgb"], "Baseline XGBoost", "#455A64"),
        (axes[1], cmv["pred_xgb"],  "+ CMV XGBoost",    "#26A69A"),
    ]:
        err = pred - obs
        ax.scatter(obs, err, alpha=0.2, s=8, color=color)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(np.nanmean(err), color="red", lw=1.5, ls="--",
                   label=f"MBE={np.nanmean(err):+.2f} kW")
        ax.set_xlabel("Actual PV [kW]", fontsize=9)
        ax.set_ylabel("Residual [kW]", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Residual Analysis — Test Set (Feb–Mar 2023)", fontsize=11)
    fig.tight_layout()
    p = _OUT_DIR / "03_residual_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {p}")

    # ── Fig 4: 7-day time series (CMV model vs baseline) ─────────────────────
    try:
        zoom_s = pd.Timestamp("2023-02-01", tz="UTC")
        zoom_e = pd.Timestamp("2023-02-07 23:55", tz="UTC")
        mask_z = (test.index >= zoom_s) & (test.index <= zoom_e)
        idx_z  = test.index[mask_z]
        obs_z  = obs[mask_z]
        base_z = base["pred_xgb"][mask_z]
        cmv_z  = cmv["pred_xgb"][mask_z]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(idx_z, obs_z,   lw=1.5, color="black",   alpha=0.9,  label="Actual PV")
        ax.plot(idx_z, base_z,  lw=1.2, color="#455A64", alpha=0.8,  ls="--", label="Baseline XGB")
        ax.plot(idx_z, cmv_z,   lw=1.2, color="#26A69A", alpha=0.85, label="+ CMV XGB")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylabel("Power [kW]", fontsize=10)
        ax.set_title("7-day forecast: Feb 2023  (Baseline vs + CMV XGBoost)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = _OUT_DIR / "04_timeseries_zoom.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {p}")
    except Exception as e:
        logger.warning(f"Zoom plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMV-augmented ML pipeline")
    parser.add_argument("--config",       default="configs/site.yaml")
    parser.add_argument("--xgb-only",    action="store_true",
                        help="Train XGBoost only (skip GRU and CNN-GRU, ~5 min)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip re-training baseline (load from metrics CSV)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MET_DIR.mkdir(parents=True, exist_ok=True)
    _MOD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data (same as run_ml_4yr.py) ────────────────────────────────────
    logger.info("Loading 4-year Solcast data …")
    raw_solcast = load_solcast_local_files(cfg)
    solcast_cal = solcast_to_nasa_schema(raw_solcast)
    solcast_cal = add_weather_pattern_features(solcast_cal)
    logger.info(f"  Solcast: {len(solcast_cal):,} rows")

    interim = resolve_path(cfg["paths"]["interim"])
    local_5min = pd.read_csv(interim / "local_5min_utc.csv",
                             index_col="timestamp_utc", parse_dates=True)
    local_5min.index = pd.to_datetime(local_5min.index, utc=True)

    synth_path  = resolve_path(cfg["paths"]["synthetic"]) / "solcast_pv_synthetic_5min.csv"
    sim_df      = pd.read_csv(synth_path, index_col="timestamp_utc", parse_dates=True)
    sim_df.index = pd.to_datetime(sim_df.index, utc=True)
    physics_sim  = sim_df["pv_ac_W"]

    pv_labels = build_4yr_labels(local_5min, physics_sim)

    # ── Build base feature matrix ─────────────────────────────────────────────
    logger.info("\n── Building feature matrix ──────────────────────────────────")
    feat_base = build_feature_matrix(solcast_cal, pv_labels, physics_sim, use_lags=True)
    feature_cols_base = [c for c in feat_base.columns
                         if c not in ("pv_ac_kW", "is_actual")]
    logger.info(f"  Base features: {len(feature_cols_base)}")

    # ── Load and attach CMV features ──────────────────────────────────────────
    logger.info("\n── Attaching CMV features ───────────────────────────────────")
    cmv_data = load_cmv_features()
    feat_cmv, new_cmv_cols = attach_cmv(feat_base, cmv_data)
    feature_cols_cmv = feature_cols_base + new_cmv_cols
    logger.info(f"  CMV features added: {new_cmv_cols}")
    logger.info(f"  Total features (with CMV): {len(feature_cols_cmv)}")

    # ── Experiment 1: Baseline (no CMV) ──────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  EXPERIMENT 1: Baseline (no CMV features)")
    logger.info("═"*60)
    result_base = run_experiment(
        feat_base, feature_cols_base, physics_sim,
        label_suffix="base", xgb_only=args.xgb_only,
    )

    # ── Experiment 2: With CMV features ──────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  EXPERIMENT 2: + CMV features")
    logger.info("═"*60)
    result_cmv = run_experiment(
        feat_cmv, feature_cols_cmv, physics_sim,
        label_suffix="cmv", xgb_only=args.xgb_only,
    )

    # ── Compare ───────────────────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  RESULTS — TEST SET (Feb–Mar 2023)")
    logger.info("═"*60)

    rows = []
    model_keys = [("XGBoost", "xgb"), ("GRU", "gru"), ("CNN-GRU", "cnnlstm")]
    for mname, key in model_keys:
        if result_base[key] is None:
            continue
        b = result_base[key]
        c = result_cmv[key]
        delta_r2   = c["R2"]       - b["R2"]
        delta_rmse = c["RMSE_kW"]  - b["RMSE_kW"]
        delta_mae  = c["MAE_kW"]   - b["MAE_kW"]
        rows.append({
            "model":          mname,
            "base_R2":        round(b["R2"],      4),
            "cmv_R2":         round(c["R2"],      4),
            "delta_R2":       round(delta_r2,     4),
            "base_RMSE_kW":   round(b["RMSE_kW"], 2),
            "cmv_RMSE_kW":    round(c["RMSE_kW"], 2),
            "delta_RMSE_kW":  round(delta_rmse,   2),
            "base_MAE_kW":    round(b["MAE_kW"],  2),
            "cmv_MAE_kW":     round(c["MAE_kW"],  2),
            "delta_MAE_kW":   round(delta_mae,    2),
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(_MET_DIR / "cmv_impact_comparison.csv", index=False)

    print("\n" + "═"*80)
    print("  CMV FEATURE IMPACT  |  Test set: Feb–Mar 2023")
    print("═"*80)
    print(f"  {'Model':<12} {'Base R²':>8} {'CMV R²':>8} {'ΔR²':>8} "
          f"{'Base RMSE':>10} {'CMV RMSE':>10} {'ΔRMSE':>8}")
    print("  " + "-"*70)
    for _, row in comp_df.iterrows():
        dr2  = f"{row['delta_R2']:+.4f}"
        drms = f"{row['delta_RMSE_kW']:+.2f}"
        r2_marker  = "▲" if row['delta_R2']       > 0 else "▼"
        rms_marker = "▼" if row['delta_RMSE_kW']  < 0 else "▲"
        print(f"  {row['model']:<12} {row['base_R2']:>8.4f} {row['cmv_R2']:>8.4f} "
              f"{dr2:>8} {r2_marker}  "
              f"{row['base_RMSE_kW']:>8.2f}   {row['cmv_RMSE_kW']:>8.2f}  "
              f"{drms:>7} {rms_marker}")
    print("═"*80)
    print(f"\n  Metrics → {_MET_DIR}/cmv_impact_comparison.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("\nGenerating comparison plots …")
    xgb_model_path = _MOD_DIR / "xgb_cmv.json"
    from xgboost import XGBRegressor
    xgb_cmv = XGBRegressor(); xgb_cmv.load_model(str(xgb_model_path))

    plot_cmv_comparison(
        result_base, result_cmv,
        feature_cols_base, feature_cols_cmv,
        xgb_model_cmv=xgb_cmv,
    )

    print(f"\n  Figures → {_OUT_DIR}/")
    print("═"*80)


if __name__ == "__main__":
    main()
