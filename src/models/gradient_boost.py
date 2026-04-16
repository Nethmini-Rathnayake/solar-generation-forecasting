"""
src/models/gradient_boost.py
------------------------------
XGBoost Direct Multi-Step (DMS) forecaster for 24h-ahead PV power.

Strategy: Direct Multi-Step
----------------------------
Train one XGBRegressor per forecast horizon h ∈ {1, 2, …, 24}.
Each model independently predicts ŷ(t+h) = f_h(X_t).

This avoids recursive error propagation: each model uses only ground-truth
lag features observed at time t, so errors at h=1 do not corrupt h=2..24.

Model hyperparameters
----------------------
Chosen for tropical solar time-series (strong diurnal pattern, moderate
autocorrelation, sharp cloud transients):

  max_depth         = 6     — enough depth for GHI × lag interactions
  learning_rate     = 0.05  — slow learning for better generalisation
  n_estimators      = 1000  — paired with early stopping (patience=50)
  subsample         = 0.8   — row subsampling prevents overfitting
  colsample_bytree  = 0.8   — feature subsampling for diversity
  min_child_weight  = 5     — avoids tiny leaf splits on rare clear-sky hours
  reg_alpha         = 0.05  — mild L1 sparsity on lags
  reg_lambda        = 1.0   — default L2

Early stopping on the validation set prevents overfitting and selects the
optimal n_estimators per horizon automatically.

Usage
-----
    from src.models.gradient_boost import train_dms_models, predict_dms, save_models, load_models

    models = train_dms_models(X_train, Y_train, X_val, Y_val)
    preds  = predict_dms(models, X_test)     # DataFrame: cols = target_h1…h24
    save_models(models, Path("results/models"))
    models = load_models(Path("results/models"))
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Default XGBoost hyperparameters ──────────────────────────────────────────
_DEFAULT_PARAMS: dict = {
    "n_estimators":      1000,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "reg_alpha":         0.05,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "tree_method":       "hist",       # fast histogram algorithm
    "random_state":      42,
    "n_jobs":            -1,
}

_EARLY_STOPPING_ROUNDS: int = 50
_N_HORIZONS: int = 24


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_dms_models(
    X_train:    pd.DataFrame,
    Y_train:    pd.DataFrame,
    X_val:      pd.DataFrame,
    Y_val:      pd.DataFrame,
    params:     dict | None = None,
    n_horizons: int = _N_HORIZONS,
) -> dict[int, object]:
    """
    Train one XGBRegressor per forecast horizon h = 1 … n_horizons.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices. Columns must be identical.
    Y_train, Y_val : pd.DataFrame
        Target matrices. Must contain columns target_h1 … target_h{n_horizons}.
    params : dict, optional
        XGBoost hyperparameters. Defaults to _DEFAULT_PARAMS.
    n_horizons : int
        Number of forecast steps. Default 24.

    Returns
    -------
    dict[int, XGBRegressor]
        Maps horizon h (1..n_horizons) → fitted model.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    hp = {**_DEFAULT_PARAMS, **(params or {})}
    models: dict[int, object] = {}

    logger.info(
        f"Training {n_horizons} XGBoost models  "
        f"(train={len(X_train):,}  val={len(X_val):,}  features={X_train.shape[1]})"
    )

    best_iters = []
    for h in range(1, n_horizons + 1):
        col = f"target_h{h}"
        y_tr = Y_train[col].values
        y_va = Y_val[col].values

        model = XGBRegressor(
            **hp,
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        )
        model.fit(
            X_train.values, y_tr,
            eval_set=[(X_val.values, y_va)],
            verbose=False,
        )

        best_iters.append(model.best_iteration)
        if h % 6 == 0 or h == 1:
            rmse_val = float(np.sqrt(np.mean(
                (model.predict(X_val.values) - y_va) ** 2
            )))
            logger.info(
                f"  h={h:2d}: best_iter={model.best_iteration:4d}  "
                f"val RMSE={rmse_val/1000:.2f} kW"
            )
        models[h] = model

    logger.info(
        f"  Mean best_iteration: {np.mean(best_iters):.0f}  "
        f"(range {min(best_iters)}–{max(best_iters)})"
    )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_dms(
    models: dict[int, object],
    X:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for all horizons.

    Parameters
    ----------
    models : dict[int, XGBRegressor]
        Output of train_dms_models().
    X : pd.DataFrame
        Feature matrix (same columns as training).

    Returns
    -------
    pd.DataFrame
        Shape (len(X), n_horizons). Columns: pred_h1 … pred_h{n_horizons}.
        Index matches X.index. All predictions are clipped to ≥ 0.
    """
    preds = {}
    for h, model in sorted(models.items()):
        preds[f"pred_h{h}"] = model.predict(X.values).clip(min=0.0)

    return pd.DataFrame(preds, index=X.index)


# ─────────────────────────────────────────────────────────────────────────────
# Persist
# ─────────────────────────────────────────────────────────────────────────────

def save_models(models: dict[int, object], out_dir: Path) -> None:
    """
    Save all horizon models to out_dir using XGBoost's native JSON format.

    Saves:
        out_dir/xgb_h01.json … xgb_h24.json
        out_dir/model_metadata.json  (feature count, n_horizons, best iters)

    Parameters
    ----------
    models : dict[int, XGBRegressor]
    out_dir : Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"n_horizons": len(models), "best_iterations": {}}
    for h, model in sorted(models.items()):
        path = out_dir / f"xgb_h{h:02d}.json"
        model.save_model(str(path))
        metadata["best_iterations"][str(h)] = int(model.best_iteration)

    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    total_kb = sum(f.stat().st_size for f in out_dir.glob("xgb_h*.json")) / 1024
    logger.info(
        f"Saved {len(models)} models → {out_dir}  "
        f"(total {total_kb:.0f} KB)"
    )


def load_models(out_dir: Path) -> dict[int, object]:
    """
    Load XGBoost models saved by save_models().

    Parameters
    ----------
    out_dir : Path

    Returns
    -------
    dict[int, XGBRegressor]

    Raises
    ------
    FileNotFoundError  if no model files found in out_dir.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    out_dir = Path(out_dir)
    model_files = sorted(out_dir.glob("xgb_h*.json"))
    if not model_files:
        raise FileNotFoundError(
            f"No XGBoost model files (xgb_h*.json) found in {out_dir}.\n"
            "Run  python scripts/run_training.py  first."
        )

    models = {}
    for path in model_files:
        h = int(path.stem.split("_h")[1])
        m = XGBRegressor()
        m.load_model(str(path))
        models[h] = m

    logger.info(f"Loaded {len(models)} XGBoost models from {out_dir}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(
    models:       dict[int, object],
    feature_cols: list[str],
    horizons:     list[int] | None = None,
    top_n:        int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute gain-based feature importances across all 24 horizons.

    Uses XGBoost's ``get_score(importance_type='gain')`` which measures the
    average reduction in loss per split — a far more informative metric than
    the default ``feature_importances_`` (weight = number of splits, which
    inflates frequently-used lag features regardless of their actual value).

    Parameters
    ----------
    models : dict[int, XGBRegressor]
    feature_cols : list[str]
        Column names corresponding to the feature matrix columns (positionally
        aligned: feature_cols[i] ↔ internal XGBoost feature "fi").
    horizons : list[int], optional
        Horizons to include in the summary average. Default: [1, 6, 12, 24].
    top_n : int
        Number of features to return in the summary DataFrame. Default 20.

    Returns
    -------
    importance_summary : pd.DataFrame
        Top-``top_n`` features by mean gain across ``horizons``.
        Columns: feature, mean_gain, h1, h6, h12, h24.
    importance_by_horizon : pd.DataFrame
        Full gain matrix — all features × all 24 horizons.
        Index: feature_cols.  Columns: h1, h2, …, h24.
    """
    summary_horizons = horizons or [1, 6, 12, 24]
    all_horizons     = sorted(models.keys())

    # ── Build full gain matrix (features × all horizons) ─────────────────────
    full_rows: dict[str, np.ndarray] = {}
    for h in all_horizons:
        booster = models[h].get_booster()
        score   = booster.get_score(importance_type="gain")
        # XGBoost internal names are "f0", "f1", … positionally aligned to feature_cols
        imp = np.array([score.get(f"f{i}", 0.0) for i in range(len(feature_cols))])
        full_rows[f"h{h}"] = imp

    if not full_rows:
        return pd.DataFrame(), pd.DataFrame()

    full_df = pd.DataFrame(full_rows, index=feature_cols)

    # ── Summary: top_n features by mean gain across summary_horizons ──────────
    summary_cols = [f"h{h}" for h in summary_horizons if f"h{h}" in full_df.columns]
    summary_df = full_df[summary_cols].copy()
    summary_df["mean_gain"] = summary_df.mean(axis=1)
    summary_df = (
        summary_df
        .sort_values("mean_gain", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    logger.info(
        f"Top-5 features by gain (mean across h={summary_horizons}): "
        + ", ".join(summary_df["feature"].head(5).tolist())
    )

    return summary_df, full_df
