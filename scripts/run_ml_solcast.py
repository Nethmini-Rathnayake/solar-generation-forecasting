"""
scripts/run_ml_solcast.py
--------------------------
Weather-pattern-aware XGBoost model for PV generation prediction.

Pipeline
--------
1. Load aligned 5-min Solcast + actual PV data
2. Identify & visualise weather patterns:
     - Daily cycles (average diurnal profiles by sky condition)
     - Seasonal patterns (monthly GHI, kt, cloud distributions)
     - Cloud events (high kt variability = unpredicted ramp events)
3. Engineer features (weather + time + pattern + PV lags)
4. Chronological train / val / test split
5. Train XGBoost (single model predicting PV at time t)
6. Evaluate: ML vs physics simulation vs persistence baseline
7. Generate diagnostic plots

Run
---
    python scripts/run_ml_solcast.py
    python scripts/run_ml_solcast.py --no-lag   # exclude PV lag features
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.features.time_features import add_time_features
from src.features.weather_patterns import (
    add_weather_pattern_features,
    add_pv_lag_features_5min,
)

logger = get_logger("run_ml_solcast")

_PV_OBS_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
_SKY_LABELS  = {0: "Clear", 1: "PartlyCloudy", 2: "MostlyCloudy", 3: "Overcast"}
_SKY_COLORS  = {0: "#f4a261", 1: "#90be6d", 2: "#577590", 3: "#4d4d4d"}
_OUT_DIR     = Path("results/figures/ml_solcast")
_MET_DIR     = Path("results/metrics/ml_solcast")
_MOD_DIR     = Path("results/models/ml_solcast")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pattern analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_weather_patterns(solcast: pd.DataFrame, local: pd.DataFrame) -> None:
    """Three-panel weather pattern identification figure."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    pv_kw = local[_PV_OBS_COL].clip(lower=0) / 1000
    ghi   = solcast["ALLSKY_SFC_SW_DWN_cal"]
    kt    = solcast["clearness_index"]
    sky   = solcast["sky_condition"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Weather Pattern Identification — Solcast 5-min  (2022–2023)\n"
        "University of Moratuwa  |  Daily cycles · Seasonality · Cloud events",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Diurnal PV by sky condition ─────────────────────────────────
    ax = axes[0, 0]
    df_join = pd.concat([pv_kw.rename("pv"), sky.rename("sky")], axis=1).dropna()
    df_join["hour"] = df_join.index.hour + df_join.index.minute / 60

    for code, label in _SKY_LABELS.items():
        sub = df_join[df_join["sky"] == code]
        if len(sub) < 50:
            continue
        profile = sub.groupby(sub["hour"].round(1))["pv"].mean()
        ax.plot(profile.index, profile.values, lw=2,
                color=_SKY_COLORS[code], label=label)
        ax.fill_between(profile.index, 0, profile.values,
                        alpha=0.12, color=_SKY_COLORS[code])

    ax.set_title("Daily Cycle: Mean PV by Sky Condition", fontsize=10)
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8); ax.set_xlim(0, 24); ax.set_ylim(0)

    # ── Panel 2: Monthly kt distribution (seasonality) ───────────────────────
    ax = axes[0, 1]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    kt_day = kt[(ghi > 50)]   # daytime only
    monthly_kt = [kt_day[kt_day.index.month == m].values for m in range(1, 13)]
    bp = ax.boxplot(monthly_kt, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    colors = plt.cm.RdYlGn([0.9, 0.85, 0.75, 0.4, 0.35, 0.6,
                             0.7, 0.65, 0.55, 0.3, 0.25, 0.7])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(month_names, fontsize=8)
    ax.set_title("Seasonal Pattern: Clearness Index (kt) by Month\n"
                 "(daytime only, GHI > 50 W/m²)", fontsize=10)
    ax.set_ylabel("Clearness Index kt")
    ax.axhline(0.75, ls="--", lw=0.8, color="green", alpha=0.7, label="Clear threshold")
    ax.axhline(0.40, ls="--", lw=0.8, color="orange", alpha=0.7, label="PartlyCloudy threshold")
    ax.legend(fontsize=7)

    # ── Panel 3: Sky condition share by month ─────────────────────────────────
    ax = axes[0, 2]
    sky_day = sky[ghi > 50]
    sky_month = pd.crosstab(sky_day.index.month, sky_day,
                            normalize="index") * 100
    sky_month.columns = [_SKY_LABELS[c] for c in sky_month.columns]
    sky_month.index   = month_names
    sky_month.plot(kind="bar", stacked=True, ax=ax,
                   color=[_SKY_COLORS[i] for i in range(4)
                          if _SKY_LABELS[i] in sky_month.columns],
                   alpha=0.85, width=0.8)
    ax.set_title("Sky Condition Share by Month (daytime)", fontsize=10)
    ax.set_ylabel("% of daytime intervals")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel 4: Cloud variability (ramp events) ──────────────────────────────
    ax = axes[1, 0]
    kt_std  = solcast["kt_roll15_std"]
    ramp_thr = float(kt_std.quantile(0.90))
    ax.hist(kt_std[kt_std > 0], bins=80, color="steelblue",
            alpha=0.7, edgecolor="none", density=True)
    ax.axvline(ramp_thr, color="red", lw=1.5, ls="--",
               label=f"90th percentile ({ramp_thr:.3f})\n= cloud ramp event")
    ax.set_title("Cloud Variability: 15-min kt Rolling Std\n"
                 "(unpredicted ramp events = right tail)", fontsize=10)
    ax.set_xlabel("kt std (15 min)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # ── Panel 5: kt vs PV scatter coloured by sky condition ──────────────────
    ax = axes[1, 1]
    df_kt = pd.concat([kt.rename("kt"), pv_kw.rename("pv"),
                       sky.rename("sky"), ghi.rename("ghi")], axis=1).dropna()
    df_kt = df_kt[df_kt["ghi"] > 50]   # daytime
    for code, label in _SKY_LABELS.items():
        sub = df_kt[df_kt["sky"] == code]
        if len(sub) < 20:
            continue
        ax.scatter(sub["kt"], sub["pv"], s=1, alpha=0.3,
                   color=_SKY_COLORS[code], label=f"{label} (n={len(sub):,})")
    ax.set_title("Clearness Index vs Actual PV Output\n"
                 "(coloured by sky condition)", fontsize=10)
    ax.set_xlabel("Clearness Index kt"); ax.set_ylabel("PV Power (kW)")
    leg = ax.legend(fontsize=7, markerscale=5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # ── Panel 6: Ramp event example ───────────────────────────────────────────
    ax = axes[1, 2]
    t0 = pd.Timestamp("2022-07-14 01:00:00", tz="UTC")
    t1 = pd.Timestamp("2022-07-14 08:00:00", tz="UTC")
    pv_w  = pv_kw.loc[t0:t1]
    kt_w  = kt.loc[t0:t1]
    std_w = kt_std.loc[t0:t1]

    ax2 = ax.twinx()
    ax.plot(pv_w.index, pv_w.values, lw=1.2, color="steelblue", label="Actual PV (kW)")
    ax2.plot(kt_w.index, kt_w.values, lw=1.2, color="darkorange",
             ls="--", alpha=0.8, label="kt")
    ax2.fill_between(std_w.index, 0, std_w.values * 3,
                     alpha=0.25, color="red", label="kt variability ×3")
    ax.set_title("Cloud Ramp Event Example — Jul 14 2022\n"
                 "(rapid kt oscillation = sub-5-min cloud transients)", fontsize=10)
    ax.set_ylabel("PV Power (kW)", color="steelblue")
    ax2.set_ylabel("kt / variability", color="darkorange")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="x", rotation=30)
    lines  = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7, loc="upper left")

    fig.tight_layout()
    out = _OUT_DIR / "ml_weather_patterns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    solcast:    pd.DataFrame,
    local:      pd.DataFrame,
    use_lags:   bool = True,
    physics_sim: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Merge Solcast weather + pattern features + time features + PV lags
    into a single feature matrix aligned to the actual PV observations.

    physics_sim : optional pd.Series of physics-simulated PV in Watts (index=UTC).
                  When provided, added as `pv_physics_kW` feature — gives the model
                  the best physics estimate so it only needs to learn the residual.
    """
    logger.info("Building feature matrix …")

    pv_kw = (local[_PV_OBS_COL].clip(lower=0) / 1000).rename("pv_ac_kW")

    # Start from Solcast (already has weather pattern features)
    feat = solcast.copy()

    # Select only the columns we want as features from Solcast
    weather_cols = [
        "ALLSKY_SFC_SW_DWN_cal",    # GHI
        "CLRSKY_SFC_SW_DWN_cal",    # clear-sky GHI
        "ALLSKY_SFC_SW_DIFF_cal",   # DHI
        "ALLSKY_SFC_SW_DNI_cal",    # DNI
        "T2M_cal",                  # temperature
        "WS10M_cal",                # wind speed
        # Pattern features (added by add_weather_pattern_features)
        "clearness_index",
        "diffuse_fraction",
        "sky_condition",
        "sky_is_clear",
        "sky_is_overcast",
        "kt_roll15_std",
        "kt_roll30_std",
        "kt_trend_15min",
        "ghi_roll15_mean",
        "ghi_roll30_mean",
        "ghi_roll15_std",
        "cloud_opacity",
        "cloud_opacity_trend",
        "relative_humidity",
    ]
    weather_cols = [c for c in weather_cols if c in feat.columns]
    feat = feat[weather_cols].copy()

    # Add time / solar features
    feat = add_time_features(feat)

    # Add physics simulation output as a feature (most powerful single feature)
    if physics_sim is not None:
        phy_kw = (physics_sim.reindex(feat.index) / 1000).clip(lower=0).fillna(0)
        feat["pv_physics_kW"] = phy_kw.astype(np.float32)
        # Residual proxy: physics may over/under-predict — expose the normalised ratio
        clrsky = feat["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan)
        feat["pv_physics_per_clrsky"] = (phy_kw / clrsky).clip(0, 2).fillna(0).astype(np.float32)
        logger.info("  Added pv_physics_kW and pv_physics_per_clrsky features")

    # Add PV target
    feat["pv_ac_kW"] = pv_kw

    # Add PV lag features including 7-day lag (2016 × 5min = 1 week)
    if use_lags:
        feat = add_pv_lag_features_5min(
            feat, target_col="pv_ac_kW",
            lags=[1, 3, 6, 12, 24, 288],
        )

    # Drop rows where target is NaN or negative (night / missing)
    feat = feat.dropna(subset=["pv_ac_kW"])

    # Daytime only: solar elevation > 0 AND GHI > 10
    day_mask = (feat["solar_elevation_deg"] > 0) & \
               (feat["ALLSKY_SFC_SW_DWN_cal"] > 10)
    feat = feat[day_mask]

    # Drop rows with any NaN in features
    feat_cols = [c for c in feat.columns if c != "pv_ac_kW"]
    feat = feat.dropna(subset=feat_cols)

    logger.info(
        f"  Feature matrix: {feat.shape[0]:,} rows × {feat.shape[1]} cols  "
        f"(daytime, obs > 0 W, no NaN)"
    )
    return feat


def split_chronological(
    feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — no shuffling (prevents data leakage).

    Train : Apr–Oct 2022  (~70%)
    Val   : Nov 2022–Jan 2023  (~20%)
    Test  : Feb–Mar 2023  (~10%)
    """
    train = feat[feat.index < pd.Timestamp("2022-11-01", tz="UTC")]
    val   = feat[(feat.index >= pd.Timestamp("2022-11-01", tz="UTC")) &
                 (feat.index <  pd.Timestamp("2023-02-01", tz="UTC"))]
    test  = feat[feat.index >= pd.Timestamp("2023-02-01", tz="UTC")]

    logger.info(
        f"  Split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    return train, val, test


def split_dry_season(
    feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dry-season-only split.  Test period (Feb–Mar) is dry season;
    training on monsoon data causes distribution mismatch.

    Dry months: Jan, Feb, Mar, Jun, Jul, Aug (bimodal Sri Lanka).
    Train : Jun–Aug 2022 dry spell + Nov–Dec 2022 (NE monsoon = drier)
    Val   : Jan 2023
    Test  : Feb–Mar 2023  (same as main split)
    """
    _DRY = {1, 2, 3, 6, 7, 8}
    dry_mask = feat.index.month.isin(_DRY)
    dry      = feat[dry_mask]

    train = dry[dry.index < pd.Timestamp("2023-01-01", tz="UTC")]
    val   = dry[(dry.index >= pd.Timestamp("2023-01-01", tz="UTC")) &
                (dry.index <  pd.Timestamp("2023-02-01", tz="UTC"))]
    test  = feat[feat.index >= pd.Timestamp("2023-02-01", tz="UTC")]

    logger.info(
        f"  Dry-season split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "pv_ac_kW",
) -> object:
    """Train a single XGBoost regressor to predict `target_col`."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("Run: pip install xgboost")

    X_tr, y_tr = train[feature_cols].values, train[target_col].values
    X_va, y_va = val[feature_cols].values,   val[target_col].values

    model = XGBRegressor(
        n_estimators      = 6000,
        max_depth         = 8,
        learning_rate     = 0.01,
        subsample         = 0.85,
        colsample_bytree  = 0.75,
        colsample_bylevel = 0.75,
        min_child_weight  = 5,
        gamma             = 0.1,
        reg_alpha         = 0.05,
        reg_lambda        = 1.0,
        objective         = "reg:squarederror",
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = -1,
        early_stopping_rounds = 80,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    val_rmse = float(np.sqrt(np.mean(
        (model.predict(X_va) - y_va) ** 2
    )))
    logger.info(
        f"  XGBoost trained: best_iter={model.best_iteration}  "
        f"val RMSE={val_rmse:.2f} kW"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(obs: np.ndarray, pred: np.ndarray, label: str) -> dict:
    mask  = obs > 1.0   # daytime only (> 1 kW)
    o, p  = obs[mask], pred[mask]
    rmse  = float(np.sqrt(((o - p) ** 2).mean()))
    mae   = float(np.abs(o - p).mean())
    mbe   = float((p - o).mean())
    nrmse = rmse / float(o.mean()) * 100
    ss_res = ((o - p) ** 2).sum()
    ss_tot = ((o - o.mean()) ** 2).sum()
    r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    logger.info(
        f"  [{label:>20}]  R²={r2:.4f}  RMSE={rmse:.2f} kW  "
        f"nRMSE={nrmse:.1f}%  MAE={mae:.2f} kW  MBE={mbe:+.2f} kW  n={mask.sum():,}"
    )
    return {"model": label, "R2": r2, "RMSE_kW": rmse,
            "nRMSE_pct": nrmse, "MAE_kW": mae, "MBE_kW": mbe, "n": int(mask.sum())}


def evaluate_by_sky(
    test:         pd.DataFrame,
    feature_cols: list[str],
    model:        object,
) -> pd.DataFrame:
    """Per-sky-condition accuracy on the test set."""
    pred = model.predict(test[feature_cols].values).clip(min=0)
    obs  = test["pv_ac_kW"].values
    rows = []
    for code, label in _SKY_LABELS.items():
        mask = test["sky_condition"].values == code
        if mask.sum() < 10:
            continue
        rows.append(_metrics(obs[mask], pred[mask], label))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Result plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list[str]) -> None:
    """Horizontal bar chart of top-30 XGBoost gain importances."""
    booster = model.get_booster()
    score   = booster.get_score(importance_type="gain")
    imp     = {feature_cols[int(k[1:])]: v
               for k, v in score.items() if int(k[1:]) < len(feature_cols)}
    imp_s   = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:30]

    names, vals = zip(*imp_s)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e63946" if any(p in n for p in
              ["kt", "ghi", "cloud", "clearness", "sky", "diffuse"])
              else "#457b9d" for n in names]
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGBoost Feature Importance (Top 30 by Gain)\n"
                 "Red = weather/pattern features  |  Blue = time/lag features",
                 fontsize=11, fontweight="bold")

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e63946", label="Weather / pattern"),
                        Patch(color="#457b9d", label="Time / lag")],
              fontsize=8, loc="lower right")

    fig.tight_layout()
    out = _OUT_DIR / "ml_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_predictions_vs_actual(
    test:         pd.DataFrame,
    feature_cols: list[str],
    model:        object,
    physics_sim:  pd.Series | None = None,
) -> None:
    """Four-panel: yearly, weekly zoom, scatter, error by sky condition."""
    sns.set_theme(style="whitegrid", font_scale=0.9)

    pred_kw = pd.Series(
        model.predict(test[feature_cols].values).clip(min=0),
        index=test.index, name="ml_pred_kW"
    )
    obs_kw = test["pv_ac_kW"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "XGBoost ML Model: Predicted vs Actual PV — Test Set (Feb–Mar 2023)\n"
        "University of Moratuwa  |  5-min resolution",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Full test period daily means ─────────────────────────────────
    ax = axes[0, 0]
    obs_d  = obs_kw.where(obs_kw > 0.5).resample("1D").mean()
    pred_d = pred_kw.where(pred_kw > 0.5).resample("1D").mean()
    ax.fill_between(obs_d.index, 0, obs_d.values, alpha=0.3, color="steelblue")
    ax.fill_between(pred_d.index, 0, pred_d.values, alpha=0.3, color="darkorange")
    ax.plot(obs_d.index, obs_d.values, lw=1.5, color="steelblue", label="Actual")
    ax.plot(pred_d.index, pred_d.values, lw=1.5, color="darkorange",
            ls="--", label="ML predicted")
    if physics_sim is not None:
        phy = physics_sim.reindex(obs_kw.index)
        phy_d = (phy / 1000).where(phy > 500).resample("1D").mean()
        ax.plot(phy_d.index, phy_d.values, lw=1.2, color="green",
                ls=":", alpha=0.8, label="Physics simulation")
    ax.set_title("Daily Mean Power — Test Period", fontsize=10)
    ax.set_ylabel("Mean Daytime Power (kW)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # ── Panel 2: 5-min zoom — one week ───────────────────────────────────────
    ax = axes[0, 1]
    t0 = pd.Timestamp("2023-02-06", tz="UTC")
    t1 = pd.Timestamp("2023-02-13", tz="UTC")
    o2 = obs_kw.loc[t0:t1]
    p2 = pred_kw.loc[t0:t1]
    ax.fill_between(o2.index, 0, o2.values, alpha=0.3, color="steelblue")
    ax.plot(o2.index, o2.values, lw=0.8, color="steelblue", label="Actual (5-min)")
    ax.plot(p2.index, p2.values, lw=0.8, color="darkorange",
            ls="--", alpha=0.85, label="ML predicted (5-min)")
    if physics_sim is not None:
        p_phy = (physics_sim.reindex(o2.index) / 1000).clip(lower=0)
        ax.plot(p_phy.index, p_phy.values, lw=0.8, color="green",
                ls=":", alpha=0.7, label="Physics")
    ax.set_title("5-min Resolution — Week of Feb 6–12 2023", fontsize=10)
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # ── Panel 3: Scatter ML vs actual ─────────────────────────────────────────
    ax = axes[1, 0]
    sky_test = test["sky_condition"].values
    for code, label in _SKY_LABELS.items():
        mask = (sky_test == code) & (obs_kw.values > 1.0)
        if mask.sum() < 10:
            continue
        ax.scatter(obs_kw.values[mask], pred_kw.values[mask],
                   s=2, alpha=0.3, color=_SKY_COLORS[code], label=label)
    lim = max(obs_kw.max(), pred_kw.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual PV (kW)"); ax.set_ylabel("ML Predicted PV (kW)")
    ax.set_title("Scatter: ML Predictions vs Actual\n(coloured by sky condition)",
                 fontsize=10)
    leg = ax.legend(fontsize=7, markerscale=5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    o_all = obs_kw.values[obs_kw.values > 1]
    p_all = pred_kw.values[obs_kw.values > 1]
    r2  = float(1 - ((o_all - p_all) ** 2).sum() /
                ((o_all - o_all.mean()) ** 2).sum())
    rmse = float(np.sqrt(((o_all - p_all) ** 2).mean()))
    ax.set_title(
        f"Scatter: ML vs Actual  R²={r2:.4f}  RMSE={rmse:.1f} kW", fontsize=10
    )

    # ── Panel 4: RMSE by sky condition — ML vs Physics ────────────────────────
    ax = axes[1, 1]
    sky_results = []
    for code, label in _SKY_LABELS.items():
        mask_d = (sky_test == code) & (obs_kw.values > 1.0)
        if mask_d.sum() < 10:
            continue
        o_s = obs_kw.values[mask_d]
        p_s = pred_kw.values[mask_d]
        ml_rmse  = float(np.sqrt(((o_s - p_s) ** 2).mean()))
        sky_results.append({"Sky": label, "ML RMSE (kW)": ml_rmse, "n": mask_d.sum()})
        if physics_sim is not None:
            phy_s = (physics_sim.reindex(test.index).values[mask_d] / 1000).clip(0)
            phy_rmse = float(np.sqrt(((o_s - phy_s) ** 2).mean()))
            sky_results[-1]["Physics RMSE (kW)"] = phy_rmse

    sky_df = pd.DataFrame(sky_results).set_index("Sky")
    rmse_cols = [c for c in sky_df.columns if "RMSE" in c]
    x = np.arange(len(sky_df))
    w = 0.35
    for i, col in enumerate(rmse_cols):
        color = "darkorange" if "ML" in col else "steelblue"
        bars = ax.bar(x + i * w, sky_df[col].values, w,
                      label=col, color=color, alpha=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(sky_df.index, fontsize=9)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("RMSE by Sky Condition: ML vs Physics Simulation", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0)

    fig.tight_layout()
    out = _OUT_DIR / "ml_predictions_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_accuracy_summary(metrics_df: pd.DataFrame) -> None:
    """Bar chart comparing ML vs physics vs persistence."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Model Comparison — Test Set (Feb–Mar 2023)\n"
        "XGBoost ML  vs  Physics Simulation  vs  Persistence",
        fontsize=12, fontweight="bold",
    )
    palette = {"XGBoost ML": "#e63946", "Physics (Solcast)": "#457b9d",
               "Persistence (t-288)": "#6d6875"}
    metrics_plot = [
        ("R2",       "R²",          axes[0]),
        ("RMSE_kW",  "RMSE (kW)",   axes[1]),
        ("nRMSE_pct","nRMSE (%)",   axes[2]),
    ]
    for col, ylabel, ax in metrics_plot:
        vals   = metrics_df.set_index("model")[col]
        colors = [palette.get(m, "grey") for m in vals.index]
        bars   = ax.bar(vals.index, vals.values, color=colors, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if col == "R2" else 0.3),
                    f"{bar.get_height():.3f}" if col == "R2"
                    else f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        if col == "R2":
            ax.set_ylim(0, 1)

    fig.tight_layout()
    out = _OUT_DIR / "ml_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM / GRU sequence models (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    X_data: np.ndarray,    # (N, n_features) — feature-only, already normalised
    y_data: np.ndarray,    # (N,)            — target, already normalised
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over X_data / y_data.
    Returns X: (N-lookback, lookback, n_features)  y: (N-lookback,).
    The target y[i] corresponds to the step *after* X[i].
    """
    X_list, y_list = [], []
    for i in range(lookback, len(X_data)):
        X_list.append(X_data[i - lookback: i])
        y_list.append(y_data[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class _RNNModel:
    """
    LSTM or GRU regressor for 5-min PV power prediction.

    Design decisions
    ----------------
    * Feature normalisation (zero-mean, unit-variance) is fitted on training
      data only — applied to features, NOT the target.
    * Target is normalised independently (divide by max_train_target) so the
      model outputs values in [0, 1] and we recover kW by multiplying back.
    * The input sequence X contains only *features* (not the target).
      Past PV information is already encoded in the lag features
      (pv_ac_kW_lag5m, lag15m …).
    """

    def __init__(self, model_type: str = "lstm",
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.15):
        self.model_type  = model_type.lower()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.net         = None
        self.feat_mean   = None   # (n_features,)
        self.feat_std    = None   # (n_features,)
        self.target_scale = None  # scalar — max training target
        self.lookback    = None

    def _make_net(self, n_features: int):
        import torch.nn as nn

        rnn_type    = self.model_type
        hidden_size = self.hidden_size
        num_layers  = self.num_layers
        dropout     = self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
                self.rnn = rnn_cls(
                    input_size  = n_features,
                    hidden_size = hidden_size,
                    num_layers  = num_layers,
                    dropout     = dropout if num_layers > 1 else 0.0,
                    batch_first = True,
                )
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.rnn(x)
                h = self.norm(out[:, -1, :])
                return self.head(h).squeeze(-1)

        return Net()

    def fit(
        self,
        X_tr: np.ndarray,   # (N_tr, n_features)
        y_tr: np.ndarray,   # (N_tr,)  kW
        X_va: np.ndarray,   # (N_va, n_features)
        y_va: np.ndarray,   # (N_va,)  kW
        lookback:    int   = 12,
        batch_size:  int   = 256,
        max_epochs:  int   = 100,
        patience:    int   = 12,
        lr:          float = 1e-3,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.lookback = lookback

        # ── Fit feature normaliser on training data ───────────────────────────
        self.feat_mean = X_tr.mean(axis=0).astype(np.float32)
        self.feat_std  = (X_tr.std(axis=0) + 1e-8).astype(np.float32)

        # ── Normalise target to [0, 1] using training max ─────────────────────
        self.target_scale = float(np.percentile(y_tr, 99)) + 1e-6

        def _norm_X(X):
            return ((X - self.feat_mean) / self.feat_std).astype(np.float32)

        def _norm_y(y):
            return (y / self.target_scale).astype(np.float32)

        Xn_tr = _norm_X(X_tr); yn_tr = _norm_y(y_tr)
        Xn_va = _norm_X(X_va); yn_va = _norm_y(y_va)

        X_seq_tr, y_seq_tr = _build_sequences(Xn_tr, yn_tr, lookback)
        X_seq_va, y_seq_va = _build_sequences(Xn_va, yn_va, lookback)

        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
        label = self.model_type.upper()
        logger.info(f"  [{label}] device={device}  "
                    f"train_seqs={len(X_seq_tr):,}  val_seqs={len(X_seq_va):,}")

        self.net = self._make_net(X_seq_tr.shape[2]).to(device)
        opt     = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, mode="min", patience=4, factor=0.5, min_lr=1e-6)
        loss_fn = torch.nn.HuberLoss(delta=0.1)   # robust to cloud-event outliers

        # Keep chronological order (no shuffle) — prevents future leakage
        tr_dl = DataLoader(
            TensorDataset(torch.tensor(X_seq_tr), torch.tensor(y_seq_tr)),
            batch_size=batch_size, shuffle=False,
        )

        best_val, best_state, wait = float("inf"), None, 0

        for epoch in range(1, max_epochs + 1):
            self.net.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                xv = torch.tensor(X_seq_va).to(device)
                yv = torch.tensor(y_seq_va).to(device)
                val_mse = float(loss_fn(self.net(xv), yv))

            # Val RMSE in actual kW
            val_rmse_kw = float(np.sqrt(val_mse)) * self.target_scale
            if val_mse < best_val:
                best_val  = val_mse
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(
                        f"  [{label}] Early stop @ epoch {epoch}  "
                        f"best val RMSE={np.sqrt(best_val)*self.target_scale:.2f} kW"
                    )
                    break

            sched.step(val_mse)
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  [{label}] epoch {epoch:>3}  val RMSE={val_rmse_kw:.2f} kW")

        self.net.load_state_dict(best_state)
        self.net.to(torch.device("cpu"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : (N, n_features) — raw (unnormalised) features.
        Returns (N,) predictions in kW; first `lookback` entries are NaN.
        """
        import torch

        Xn = ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        # Dummy y for sequence builder (not used)
        y_dummy = np.zeros(len(Xn), dtype=np.float32)
        X_seq, _ = _build_sequences(Xn, y_dummy, self.lookback)

        self.net.eval()
        with torch.no_grad():
            preds_norm = self.net(torch.tensor(X_seq)).numpy()

        preds_kw = (preds_norm * self.target_scale).clip(min=0)
        out = np.full(len(X), np.nan, dtype=np.float32)
        out[self.lookback:] = preds_kw
        return out


def train_rnn(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feature_cols: list[str],
    model_type:   str = "lstm",
    lookback:     int = 12,
) -> tuple[_RNNModel, np.ndarray]:
    """
    Train LSTM or GRU; returns (model, test_predictions_kW array).

    Contiguous time-order is preserved for each split.
    `lookback` rows from the previous split are prepended to provide
    warm-start context for the first predictions.
    """
    label = model_type.upper()
    logger.info(
        f"Training {label} (lookback={lookback}×5min={lookback*5}min) …"
    )
    target_col = "pv_ac_kW"

    def _prepend(block, source, n):
        return pd.concat([source.iloc[-n:], block])

    # ── Build arrays ──────────────────────────────────────────────────────────
    X_tr = train[feature_cols].values.astype(np.float32)
    y_tr = train[target_col].values.astype(np.float32)

    va_df   = _prepend(val, train, lookback)
    X_va    = va_df[feature_cols].values.astype(np.float32)
    y_va    = va_df[target_col].values.astype(np.float32)

    te_df   = _prepend(test, val, lookback)
    X_te    = te_df[feature_cols].values.astype(np.float32)
    y_te    = te_df[target_col].values.astype(np.float32)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = _RNNModel(model_type=model_type,
                      hidden_size=64, num_layers=1, dropout=0.0)
    model.fit(X_tr, y_tr, X_va, y_va,
              lookback=lookback, max_epochs=80, patience=10, lr=1e-3)

    # ── Predict on test ───────────────────────────────────────────────────────
    preds_pre  = model.predict(X_te)          # (len(te_df),)  first lookback = NaN
    test_preds = preds_pre[lookback:]         # aligned to test rows
    test_preds = test_preds[:len(test)]       # trim edge case

    # ── Report final val RMSE in kW ───────────────────────────────────────────
    preds_va  = model.predict(X_va)           # (len(va_df),)
    preds_val = preds_va[lookback:][:len(val)]
    val_rmse  = float(np.sqrt(np.nanmean((val[target_col].values - preds_val) ** 2)))
    logger.info(f"  [{label}] final val RMSE={val_rmse:.2f} kW")

    return model, test_preds


def plot_all_models_comparison(
    test:          pd.DataFrame,
    feature_cols:  list[str],
    xgb_model:     object,
    lstm_preds:    np.ndarray,
    gru_preds:     np.ndarray,
    physics_sim:   pd.Series | None,
    metrics_df:    pd.DataFrame,
) -> None:
    """
    Combined comparison plot: XGBoost + LSTM + GRU + Physics + Persistence.
    Three panels:
      1. Daily mean power — full test period (all models overlaid)
      2. 5-min zoom — one week
      3. R² / RMSE bar chart
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    obs_kw   = test["pv_ac_kW"]
    xgb_preds = pd.Series(
        xgb_model.predict(test[feature_cols].values).clip(min=0),
        index=test.index,
    )
    lstm_s = pd.Series(lstm_preds, index=test.index, name="LSTM")
    gru_s  = pd.Series(gru_preds,  index=test.index, name="GRU")

    model_series = {
        "Actual":        (obs_kw,     "steelblue",   "-",  2.0),
        "XGBoost":       (xgb_preds,  "#e63946",     "--", 1.5),
        "LSTM":          (lstm_s,     "#2a9d8f",     "-.", 1.5),
        "GRU":           (gru_s,      "#e9c46a",     ":",  1.5),
    }
    if physics_sim is not None:
        phy_kw = (physics_sim.reindex(test.index) / 1000).clip(lower=0)
        model_series["Physics"] = (phy_kw, "#457b9d", "--", 1.2)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Model Comparison — Test Set (Feb–Mar 2023)  |  XGBoost vs LSTM vs GRU vs Physics\n"
        "University of Moratuwa  |  5-min resolution",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Daily means ──────────────────────────────────────────────────
    ax = axes[0]
    for name, (s, color, ls, lw) in model_series.items():
        d = s.where(s > 0.5).resample("1D").mean()
        ax.plot(d.index, d.values, lw=lw, color=color, ls=ls, label=name, alpha=0.85)
    ax.set_title("Daily Mean Power — Full Test Period", fontsize=10)
    ax.set_ylabel("Mean Daytime Power (kW)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # ── Panel 2: 5-min zoom ───────────────────────────────────────────────────
    ax = axes[1]
    t0 = pd.Timestamp("2023-02-06", tz="UTC")
    t1 = pd.Timestamp("2023-02-13", tz="UTC")
    for name, (s, color, ls, lw) in model_series.items():
        seg = s.loc[t0:t1]
        ax.plot(seg.index, seg.values, lw=max(lw * 0.6, 0.6),
                color=color, ls=ls, label=name, alpha=0.85)
    ax.set_title("5-min Resolution — Feb 6–12 2023", fontsize=10)
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # ── Panel 3: R² and RMSE bar chart ───────────────────────────────────────
    ax = axes[2]
    palette = {
        "XGBoost ML":          "#e63946",
        "LSTM":                "#2a9d8f",
        "GRU":                 "#e9c46a",
        "XGB Residual (Phy+ML)":  "#2d6a4f",
        "XGB Dry-Season":         "#f77f00",
        "Ensemble (4-way)":       "#6a0572",
        "Ensemble (Res+XGB+GRU)": "#6a0572",
        "Ensemble (XGB+GRU)":     "#6a0572",
        "Physics (Solcast)":   "#457b9d",
        "Persistence (t-288)": "#6d6875",
    }
    m = metrics_df.set_index("model")
    x = np.arange(len(m))
    w = 0.35
    colors = [palette.get(n, "grey") for n in m.index]
    bars1 = ax.bar(x - w / 2, m["R2"].values,       w, color=colors, alpha=0.85, label="R²")
    bars2 = ax.bar(x + w / 2, m["RMSE_kW"].values / m["RMSE_kW"].max(),
                   w, color=colors, alpha=0.45, hatch="///", label="RMSE (norm.)")
    ax.set_xticks(x)
    ax.set_xticklabels(m.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R²  (bars)  |  normalised RMSE (hatched)")
    ax.set_title("R² and RMSE by Model", fontsize=10)
    ax.set_ylim(0, 1.05)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = _OUT_DIR / "ml_all_models_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/site.yaml")
    parser.add_argument("--no-lag", action="store_true",
                        help="Exclude PV lag features (model applies to all years)")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    use_lags = not args.no_lag

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MET_DIR.mkdir(parents=True, exist_ok=True)
    _MOD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load aligned 5-min data ───────────────────────────────────────────────
    logger.info("Loading 5-min aligned data …")
    interim = resolve_path(cfg["paths"]["interim"])
    local   = pd.read_csv(interim / "local_5min_utc.csv",
                          index_col="timestamp_utc", parse_dates=True)
    local.index = pd.to_datetime(local.index, utc=True)

    solcast = pd.read_csv(interim / "solcast_5min_aligned.csv",
                          index_col="timestamp_utc", parse_dates=True)
    solcast.index = pd.to_datetime(solcast.index, utc=True)

    logger.info(f"  Local  : {len(local):,} rows  ({local.index.min().date()} → {local.index.max().date()})")
    logger.info(f"  Solcast: {len(solcast):,} rows  ({solcast.index.min().date()} → {solcast.index.max().date()})")

    # ── Map Solcast columns to NASA schema & add pattern features ─────────────
    logger.info("Adding weather pattern features …")
    from src.data.solcast_loader import solcast_to_nasa_schema
    solcast_cal = solcast_to_nasa_schema(solcast)
    solcast_cal = add_weather_pattern_features(solcast_cal)

    # ── Load physics simulation (used as feature AND for comparison) ─────────
    synth_path = resolve_path(cfg["paths"]["synthetic"]) / "solcast_pv_synthetic_5min.csv"
    physics_sim = None
    if synth_path.exists():
        sim_df = pd.read_csv(synth_path, index_col="timestamp_utc", parse_dates=True)
        sim_df.index = pd.to_datetime(sim_df.index, utc=True)
        physics_sim = sim_df["pv_ac_W"]
        logger.info(f"  Physics simulation loaded: {len(sim_df):,} rows")
    else:
        logger.warning(f"  Physics simulation not found at {synth_path} — skipping physics feature")

    # ── Pattern analysis plot ─────────────────────────────────────────────────
    logger.info("Plotting weather pattern analysis …")
    plot_weather_patterns(solcast_cal, local)

    # ── Build feature matrix (physics output included as feature) ─────────────
    feat = build_feature_matrix(solcast_cal, local,
                                use_lags=use_lags, physics_sim=physics_sim)

    feature_cols = [c for c in feat.columns if c != "pv_ac_kW"]
    logger.info(f"  Features ({len(feature_cols)}): {feature_cols[:8]} … ")

    # ── Chronological split ───────────────────────────────────────────────────
    train, val, test = split_chronological(feat)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    logger.info("Training XGBoost …")
    model = train_xgboost(train, val, feature_cols)
    model.save_model(str(_MOD_DIR / "xgb_solcast_5min.json"))
    logger.info(f"Model saved → {_MOD_DIR / 'xgb_solcast_5min.json'}")

    # ── Evaluate XGBoost on test set ──────────────────────────────────────────
    logger.info("\n── Test set accuracy ─────────────────────────────────────────")
    pred_test = model.predict(test[feature_cols].values).clip(min=0)
    obs_test  = test["pv_ac_kW"].values

    rows = [_metrics(obs_test, pred_test, "XGBoost ML")]

    if physics_sim is not None:
        phy_test = (physics_sim.reindex(test.index).values / 1000).clip(min=0)
        rows.append(_metrics(obs_test, phy_test, "Physics (Solcast)"))

    if "pv_ac_kW_lag1440m" in feat.columns:
        pers_test = test["pv_ac_kW_lag1440m"].fillna(0).values
        rows.append(_metrics(obs_test, pers_test, "Persistence (t-288)"))

    # ── Per-sky-condition accuracy ─────────────────────────────────────────────
    logger.info("\n── Accuracy by sky condition ────────────────────────────────")
    sky_metrics = evaluate_by_sky(test, feature_cols, model)
    sky_metrics.to_csv(_MET_DIR / "ml_accuracy_by_sky.csv", index=False)

    # ── Residual XGBoost (physics + ML correction) ────────────────────────────
    # Strategy: train XGBoost on (observed - physics) residuals.
    # Final prediction = physics + ML_correction.
    # If ML explains 70% of residual variance → R² ≈ 0.83 + 0.70×0.17 ≈ 0.95
    logger.info("\n── Residual XGBoost (Physics + ML correction) ───────────────")
    if physics_sim is not None:
        # Build residual target: obs - physics (in kW) for each split
        def _add_residual(df):
            phy = (physics_sim.reindex(df.index) / 1000).clip(lower=0).fillna(0)
            df = df.copy()
            df["pv_residual_kW"] = (df["pv_ac_kW"] - phy).values
            return df

        train_r = _add_residual(train)
        val_r   = _add_residual(val)
        test_r  = _add_residual(test)

        residual_model = train_xgboost(train_r, val_r, feature_cols,
                                       target_col="pv_residual_kW")

        # Predict: physics + correction
        phy_test_kw    = (physics_sim.reindex(test.index) / 1000).clip(lower=0).fillna(0).values
        residual_preds = residual_model.predict(test[feature_cols].values)
        xgb_residual_preds = (phy_test_kw + residual_preds).clip(min=0)
        rows.append(_metrics(obs_test, xgb_residual_preds, "XGB Residual (Phy+ML)"))

        # Also val residual RMSE for ensemble weighting
        phy_val_kw = (physics_sim.reindex(val.index) / 1000).clip(lower=0).fillna(0).values
        res_val    = residual_model.predict(val[feature_cols].values)
        xgb_res_val = (phy_val_kw + res_val).clip(min=0)
        logger.info(f"  XGB Residual val RMSE={float(np.sqrt(np.mean((val['pv_ac_kW'].values - xgb_res_val)**2))):.2f} kW")
    else:
        xgb_residual_preds = pred_test   # fallback
        xgb_res_val = xgb_val

    # ── Dry-season XGBoost (test period is Feb–Mar = dry season) ─────────────
    logger.info("\n── Dry-season XGBoost ───────────────────────────────────────")
    train_dry, val_dry, _ = split_dry_season(feat)
    dry_model = train_xgboost(train_dry, val_dry, feature_cols)
    dry_preds = dry_model.predict(test[feature_cols].values).clip(min=0)
    rows.append(_metrics(obs_test, dry_preds, "XGB Dry-Season"))

    dry_val_preds = dry_model.predict(val[feature_cols].values).clip(min=0)

    # ── Train LSTM ────────────────────────────────────────────────────────────
    logger.info("\n── Training LSTM ────────────────────────────────────────────")
    _lstm_model, lstm_preds = train_rnn(train, val, test, feature_cols,
                                        model_type="lstm", lookback=6)
    lstm_preds_safe = np.where(np.isnan(lstm_preds), 0.0, lstm_preds)
    rows.append(_metrics(obs_test, lstm_preds_safe, "LSTM"))

    # ── Train GRU ─────────────────────────────────────────────────────────────
    logger.info("\n── Training GRU ─────────────────────────────────────────────")
    _gru_model, gru_preds = train_rnn(train, val, test, feature_cols,
                                      model_type="gru", lookback=6)
    gru_preds_safe = np.where(np.isnan(gru_preds), 0.0, gru_preds)
    rows.append(_metrics(obs_test, gru_preds_safe, "GRU"))

    # ── Ensemble: optimise 4-way blend on validation ──────────────────────────
    # Models: XGB-All, XGB-Dry, XGB-Residual, GRU
    logger.info("\n── Ensemble (4-way) ─────────────────────────────────────────")
    obs_val      = val["pv_ac_kW"].values
    xgb_val      = model.predict(val[feature_cols].values).clip(min=0)
    gru_val_full = _gru_model.predict(val[feature_cols].values)
    gru_val_safe = np.where(np.isnan(gru_val_full), xgb_val, gru_val_full)

    # Grid-search blend: w_dry*dry + w_res*res + w_xgb*xgb + remainder*gru
    best_ws, best_rmse = (0.4, 0.3, 0.2), float("inf")
    for w_dry in np.linspace(0.2, 0.7, 10):
        for w_res in np.linspace(0.1, 0.5, 8):
            for w_xgb in np.linspace(0.05, 0.4, 7):
                if w_dry + w_res + w_xgb >= 1.0:
                    continue
                w_gru = 1.0 - w_dry - w_res - w_xgb
                blend = (w_dry * dry_val_preds + w_res * xgb_res_val
                         + w_xgb * xgb_val + w_gru * gru_val_safe)
                rmse  = float(np.sqrt(np.mean((obs_val - blend) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_ws   = (w_dry, w_res, w_xgb)

    w_dry, w_res, w_xgb = best_ws
    w_gru = 1.0 - w_dry - w_res - w_xgb
    logger.info(
        f"  Best blend: Dry={w_dry:.2f}  Residual={w_res:.2f}  "
        f"XGB={w_xgb:.2f}  GRU={w_gru:.2f}  val RMSE={best_rmse:.2f} kW"
    )
    ensemble_preds = (
        w_dry * dry_preds + w_res * xgb_residual_preds
        + w_xgb * pred_test + w_gru * gru_preds_safe
    ).clip(min=0)
    rows.append(_metrics(obs_test, ensemble_preds, "Ensemble (4-way)"))

    # ── Save all metrics ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(_MET_DIR / "ml_model_comparison.csv", index=False)
    logger.info(f"\nSaved metrics → {_MET_DIR / 'ml_model_comparison.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating result plots …")
    plot_feature_importance(model, feature_cols)
    plot_predictions_vs_actual(test, feature_cols, model, physics_sim)
    plot_accuracy_summary(metrics_df)

    ensemble_s = pd.Series(ensemble_preds, index=test.index)
    plot_all_models_comparison(
        test, feature_cols, model,
        lstm_preds_safe, gru_preds_safe,
        physics_sim, metrics_df,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ML MODEL RESULTS — TEST SET (Feb–Mar 2023)")
    print("═" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\n  Figures  → {_OUT_DIR}/")
    print(f"  Metrics  → {_MET_DIR}/")
    print(f"  Model    → {_MOD_DIR}/xgb_solcast_5min.json")
    print("═" * 60)


if __name__ == "__main__":
    main()
