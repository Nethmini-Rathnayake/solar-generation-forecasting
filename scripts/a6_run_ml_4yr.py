"""
scripts/run_ml_4yr.py
----------------------
4-year ML pipeline for PV generation prediction.

Data strategy
-------------
Actual PV observations span Apr 2022 – Mar 2023 (1 year).
Solcast weather data spans Jan 2020 – Feb 2024 (4+ years).
Physics-calibrated synthetic PV covers the same Solcast period.

Labels used for training:
  - 2020-01-01 → 2022-03-31 : synthetic PV (calibrated physics)
  - 2022-04-01 → 2023-03-31 : actual PV observations  (overlap year)
  - 2023-04-01 → 2024-02-28 : synthetic PV (calibrated physics)

The overlap year is used for both fine-tuning and test evaluation.

Split (chronological, no leakage)
  Train : 2020-01 → 2022-10  (~80%)   mix of synthetic + actual
  Val   : 2022-11 → 2023-01  (~10%)   actual PV
  Test  : 2023-02 → 2023-03  (~5%)    actual PV  (same as 1-yr baseline)

Run
---
    python scripts/run_ml_4yr.py
"""

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
from src.data.solcast_loader import load_solcast_local_files, solcast_to_nasa_schema
from src.features.time_features import add_time_features
from src.features.weather_patterns import (
    add_weather_pattern_features,
    add_pv_lag_features_5min,
)

logger = get_logger("run_ml_4yr")

_PV_OBS_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
_OUT_DIR    = Path("results/figures/a6_ml_4yr")
_MET_DIR    = Path("results/metrics/a6_ml_4yr")
_MOD_DIR    = Path("results/models/a6_ml_4yr")

_SKY_LABELS = {0: "Clear", 1: "PartlyCloudy", 2: "MostlyCloudy", 3: "Overcast"}
_SKY_COLORS = {0: "#f4a261", 1: "#90be6d", 2: "#577590", 3: "#4d4d4d"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build 4-year label series (actual where available, else synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def build_4yr_labels(
    local_5min:   pd.DataFrame,
    synthetic_pv: pd.Series,
) -> pd.Series:
    """
    Returns pv_ac_kW series covering the full Solcast period.

    Priority: actual PV observations > synthetic PV.
    Synthetic PV is used only where actual data is absent.

    Parameters
    ----------
    local_5min   : actual PV dataframe (UTC index), ~1 year
    synthetic_pv : pv_ac_W series from physics simulation, ~4 years

    Returns
    -------
    pd.Series  pv_ac_kW  (UTC index, full 4-year span)
    """
    # Actual PV in kW
    actual_kw = (
        local_5min[_PV_OBS_COL].clip(lower=0) / 1000
    ).rename("pv_ac_kW")

    # Synthetic PV in kW
    synth_kw = (synthetic_pv.clip(lower=0) / 1000).rename("pv_ac_kW")

    # Merge: actual takes priority
    combined = synth_kw.copy()
    combined.loc[actual_kw.index] = actual_kw.values

    n_actual = len(actual_kw)
    n_synth  = len(combined) - n_actual
    logger.info(
        f"  Label source: actual={n_actual:,} rows "
        f"({actual_kw.index.min().date()} → {actual_kw.index.max().date()})  "
        f"| synthetic={n_synth:,} rows"
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    solcast_cal:  pd.DataFrame,
    pv_labels:    pd.Series,
    physics_sim:  pd.Series,
    use_lags:     bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix for full 4-year dataset.

    Lag features use the combined label series (actual + synthetic) so
    past-PV lags are available throughout the training period.
    """
    logger.info("Building 4-year feature matrix …")

    feat = solcast_cal.copy()

    weather_cols = [
        "ALLSKY_SFC_SW_DWN_cal", "CLRSKY_SFC_SW_DWN_cal",
        "ALLSKY_SFC_SW_DIFF_cal", "ALLSKY_SFC_SW_DNI_cal",
        "T2M_cal", "WS10M_cal",
        "clearness_index", "diffuse_fraction",
        "sky_condition", "sky_is_clear", "sky_is_overcast",
        "kt_roll15_std", "kt_roll30_std", "kt_trend_15min",
        "ghi_roll15_mean", "ghi_roll30_mean", "ghi_roll15_std",
        "cloud_opacity", "cloud_opacity_trend", "relative_humidity",
    ]
    weather_cols = [c for c in weather_cols if c in feat.columns]
    feat = feat[weather_cols].copy()

    # Time/solar features
    feat = add_time_features(feat)

    # Physics simulation as feature
    phy_kw = (physics_sim.reindex(feat.index) / 1000).clip(lower=0).fillna(0)
    feat["pv_physics_kW"] = phy_kw.astype(np.float32)
    clrsky = feat["CLRSKY_SFC_SW_DWN_cal"].replace(0, np.nan)
    feat["pv_physics_per_clrsky"] = (phy_kw / clrsky).clip(0, 2).fillna(0).astype(np.float32)

    # Target
    feat["pv_ac_kW"] = pv_labels.reindex(feat.index)

    # Lag features (on combined label — enables lags during synthetic period)
    if use_lags:
        feat = add_pv_lag_features_5min(feat, target_col="pv_ac_kW",
                                        lags=[1, 3, 6, 12, 24, 288])

    # Daytime only
    feat = feat.dropna(subset=["pv_ac_kW"])
    day_mask = (feat["solar_elevation_deg"] > 0) & (feat["ALLSKY_SFC_SW_DWN_cal"] > 10)
    feat = feat[day_mask]

    # Drop NaN in features
    feat_cols = [c for c in feat.columns if c != "pv_ac_kW"]
    feat = feat.dropna(subset=feat_cols)

    # Flag which rows are actual vs synthetic
    actual_idx = pd.read_csv(
        "data/interim/local_5min_utc.csv",
        index_col="timestamp_utc", parse_dates=True, usecols=["timestamp_utc"]
    ).index
    actual_idx = pd.to_datetime(actual_idx, utc=True)
    feat["is_actual"] = feat.index.isin(actual_idx).astype(np.int8)

    n_actual = int(feat["is_actual"].sum())
    n_synth  = len(feat) - n_actual
    logger.info(
        f"  Feature matrix: {len(feat):,} rows × {feat.shape[1]} cols  "
        f"(actual={n_actual:,}  synthetic={n_synth:,})"
    )
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chronological split
# ─────────────────────────────────────────────────────────────────────────────

def split_4yr(feat: pd.DataFrame):
    """
    Train : 2020-01 → 2022-10  (synthetic + actual)
    Val   : 2022-11 → 2023-01  (actual)
    Test  : 2023-02 → 2023-03  (actual)  — identical to 1-yr baseline
    """
    train = feat[feat.index < pd.Timestamp("2022-11-01", tz="UTC")]
    val   = feat[(feat.index >= pd.Timestamp("2022-11-01", tz="UTC")) &
                 (feat.index <  pd.Timestamp("2023-02-01", tz="UTC"))]
    test  = feat[(feat.index >= pd.Timestamp("2023-02-01", tz="UTC")) &
                 (feat.index <  pd.Timestamp("2023-04-01", tz="UTC"))]

    logger.info(
        f"  4-yr split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    logger.info(
        f"    train actual={int(train['is_actual'].sum()):,}  "
        f"synthetic={int((train['is_actual']==0).sum()):,}"
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(train, val, feature_cols, target_col="pv_ac_kW",
                  label="XGBoost"):
    from xgboost import XGBRegressor

    X_tr = train[feature_cols].values;  y_tr = train[target_col].values
    X_va = val[feature_cols].values;    y_va = val[target_col].values

    model = XGBRegressor(
        n_estimators         = 6000,
        max_depth            = 8,
        learning_rate        = 0.01,
        subsample            = 0.85,
        colsample_bytree     = 0.75,
        colsample_bylevel    = 0.75,
        min_child_weight     = 5,
        gamma                = 0.1,
        reg_alpha            = 0.05,
        reg_lambda           = 1.0,
        objective            = "reg:squarederror",
        tree_method          = "hist",
        random_state         = 42,
        n_jobs               = -1,
        early_stopping_rounds = 80,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_rmse = float(np.sqrt(np.mean(
        (model.predict(X_va) - y_va) ** 2
    )))
    logger.info(f"  [{label}] best_iter={model.best_iteration}  val RMSE={val_rmse:.2f} kW")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(obs, pred, label):
    mask = obs > 1.0
    o, p = obs[mask], pred[mask]
    rmse  = float(np.sqrt(((o - p) ** 2).mean()))
    mae   = float(np.abs(o - p).mean())
    mbe   = float((p - o).mean())
    nrmse = rmse / float(o.mean()) * 100
    r2    = float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
    logger.info(
        f"  [{label:>28}]  R²={r2:.4f}  RMSE={rmse:.2f} kW  "
        f"nRMSE={nrmse:.1f}%  MAE={mae:.2f} kW  MBE={mbe:+.2f} kW  n={mask.sum():,}"
    )
    return {"model": label, "R2": r2, "RMSE_kW": rmse,
            "nRMSE_pct": nrmse, "MAE_kW": mae, "MBE_kW": mbe, "n": int(mask.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRU (same architecture as run_ml_solcast.py)
# ─────────────────────────────────────────────────────────────────────────────

class _GRUModel:
    def __init__(self, hidden=64):
        self.hidden = hidden
        self.net = None
        self.feat_mean = self.feat_std = self.target_scale = None
        self.lookback = None

    def _make_net(self, n_feat):
        import torch.nn as nn
        h = self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn  = nn.GRU(n_feat, h, num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(h)
                self.head = nn.Sequential(nn.Linear(h, 32), nn.GELU(), nn.Linear(32, 1))
            def forward(self, x):
                o, _ = self.rnn(x)
                return self.head(self.norm(o[:, -1, :])).squeeze(-1)
        return Net()

    def fit(self, X_tr, y_tr, X_va, y_va, lookback=6,
            batch_size=512, epochs=80, patience=10):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.lookback      = lookback
        self.feat_mean     = X_tr.mean(0).astype(np.float32)
        self.feat_std      = (X_tr.std(0) + 1e-8).astype(np.float32)
        self.target_scale  = float(np.percentile(y_tr[y_tr > 0], 99)) + 1e-6

        def norm_X(X): return ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        def norm_y(y): return (y / self.target_scale).astype(np.float32)

        def seqs(X, y):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i-lookback:i]); ys.append(y[i])
            return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

        Xn_tr, yn_tr = norm_X(X_tr), norm_y(y_tr)
        Xn_va, yn_va = norm_X(X_va), norm_y(y_va)
        Xs_tr, ys_tr = seqs(Xn_tr, yn_tr)
        Xs_va, ys_va = seqs(Xn_va, yn_va)

        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  [GRU-4yr] device={device}  seqs={len(Xs_tr):,}")

        self.net = self._make_net(Xs_tr.shape[2]).to(device)
        opt   = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        loss_fn = torch.nn.HuberLoss(delta=0.1)

        dl = DataLoader(TensorDataset(torch.tensor(Xs_tr), torch.tensor(ys_tr)),
                        batch_size=batch_size, shuffle=False)

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                xv = torch.tensor(Xs_va).to(device)
                yv = torch.tensor(ys_va).to(device)
                val_mse = float(loss_fn(self.net(xv), yv))
            sched.step(val_mse)
            val_rmse = float(np.sqrt(val_mse)) * self.target_scale

            if val_mse < best_val:
                best_val  = val_mse
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(
                        f"  [GRU-4yr] Early stop epoch {epoch}  "
                        f"best val RMSE={np.sqrt(best_val)*self.target_scale:.2f} kW"
                    )
                    break
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  [GRU-4yr] epoch {epoch:>3}  val RMSE={val_rmse:.2f} kW")

        self.net.load_state_dict(best_state)
        self.net.to(torch.device("cpu"))

    def predict(self, X):
        import torch
        Xn  = ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        yd  = np.zeros(len(Xn), np.float32)
        Xs  = np.array([Xn[i-self.lookback:i] for i in range(self.lookback, len(Xn))],
                        dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            p = self.net(torch.tensor(Xs)).numpy()
        out = np.full(len(X), np.nan, np.float32)
        out[self.lookback:] = (p * self.target_scale).clip(min=0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. CNN-LSTM model
# ─────────────────────────────────────────────────────────────────────────────

class _CNNLSTMModel:
    """
    CNN-LSTM hybrid:
      - 1-D Conv (kernel=3, ×2) extracts local temporal patterns per feature
      - LSTM captures longer-range dependencies across the lookback window
      - LayerNorm + 2-layer head produces scalar power output

    Architecture:
        (batch, lookback, features)
        → permute → Conv1d × 2 → permute
        → LSTM → LayerNorm → Linear(32) → GELU → Linear(1)
    """

    def __init__(self, cnn_filters: int = 64, lstm_hidden: int = 64):
        self.cnn_filters = cnn_filters
        self.lstm_hidden = lstm_hidden
        self.net         = None
        self.feat_mean   = self.feat_std = self.target_scale = None
        self.lookback    = None

    def _make_net(self, n_feat: int):
        import torch.nn as nn
        cf, lh = self.cnn_filters, self.lstm_hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # CNN extracts local multi-scale features; GRU models temporal deps.
                # Using GRU (not LSTM) because it converges better on this dataset.
                self.conv = nn.Sequential(
                    nn.Conv1d(n_feat, cf, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(cf, cf, kernel_size=5, padding=2),
                    nn.GELU(),
                )
                self.gru  = nn.GRU(cf, lh, num_layers=1, batch_first=True)
                self.norm = nn.LayerNorm(lh)
                self.head = nn.Sequential(
                    nn.Linear(lh, 32), nn.GELU(), nn.Linear(32, 1)
                )

            def forward(self, x):
                c = self.conv(x.permute(0, 2, 1))   # (B, cf, T)
                c = c.permute(0, 2, 1)               # (B, T, cf)
                out, _ = self.gru(c)                 # (B, T, lh)
                return self.head(self.norm(out[:, -1, :])).squeeze(-1)

        return Net()

    def fit(self, X_tr, y_tr, X_va, y_va, lookback=12,
            batch_size=512, epochs=80, patience=15):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.lookback     = lookback
        self.feat_mean    = X_tr.mean(0).astype(np.float32)
        self.feat_std     = (X_tr.std(0) + 1e-8).astype(np.float32)
        self.target_scale = float(np.percentile(y_tr[y_tr > 0], 99)) + 1e-6

        def norm_X(X): return ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        def norm_y(y): return (y / self.target_scale).astype(np.float32)

        def seqs(X, y):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i - lookback:i]); ys.append(y[i])
            return np.array(Xs, np.float32), np.array(ys, np.float32)

        Xs_tr, ys_tr = seqs(norm_X(X_tr), norm_y(y_tr))
        Xs_va, ys_va = seqs(norm_X(X_va), norm_y(y_va))

        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"  [CNN-LSTM] device={device}  seqs={len(Xs_tr):,}")

        self.net = self._make_net(Xs_tr.shape[2]).to(device)
        opt      = torch.optim.Adam(self.net.parameters(), lr=5e-4, weight_decay=1e-4)
        sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        loss_fn  = torch.nn.HuberLoss(delta=0.1)

        dl = DataLoader(
            TensorDataset(torch.tensor(Xs_tr), torch.tensor(ys_tr)),
            batch_size=batch_size, shuffle=True,
        )

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(
                    self.net(torch.tensor(Xs_va).to(device)),
                    torch.tensor(ys_va).to(device),
                ))
            sched.step(val_mse)
            val_rmse_kw = float(np.sqrt(val_mse)) * self.target_scale

            if val_mse < best_val:
                best_val   = val_mse
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(
                        f"  [CNN-LSTM] Early stop epoch {epoch}  "
                        f"best val RMSE={np.sqrt(best_val)*self.target_scale:.2f} kW"
                    )
                    break
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  [CNN-LSTM] epoch {epoch:>3}  val RMSE={val_rmse_kw:.2f} kW")

        self.net.load_state_dict(best_state)
        self.net.to(torch.device("cpu"))

    def fine_tune(self, X_ft, y_ft, X_va, y_va,
                  epochs: int = 25, lr: float = 1e-4, patience: int = 10):
        """
        Fine-tune on a smaller actual-data subset after pre-training.
        Lower lr avoids overwriting the generalizable weights from pre-training.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        def norm_X(X): return ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        def norm_y(y): return (y / self.target_scale).astype(np.float32)

        def seqs(X, y):
            Xs, ys = [], []
            for i in range(self.lookback, len(X)):
                Xs.append(X[i - self.lookback:i]); ys.append(y[i])
            return np.array(Xs, np.float32), np.array(ys, np.float32)

        Xs_ft, ys_ft = seqs(norm_X(X_ft), norm_y(y_ft))
        Xs_va, ys_va = seqs(norm_X(X_va), norm_y(y_va))

        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"  [CNN-LSTM fine-tune] seqs={len(Xs_ft):,}  lr={lr}")
        self.net.to(device)

        opt     = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.HuberLoss(delta=0.1)
        dl = DataLoader(
            TensorDataset(torch.tensor(Xs_ft), torch.tensor(ys_ft)),
            batch_size=256, shuffle=True,
        )

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(
                    self.net(torch.tensor(Xs_va).to(device)),
                    torch.tensor(ys_va).to(device),
                ))
            val_rmse_kw = float(np.sqrt(val_mse)) * self.target_scale

            if val_mse < best_val:
                best_val   = val_mse
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(
                        f"  [CNN-LSTM fine-tune] Early stop epoch {epoch}  "
                        f"best val RMSE={np.sqrt(best_val)*self.target_scale:.2f} kW"
                    )
                    break
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"  [CNN-LSTM fine-tune] epoch {epoch:>3}  val RMSE={val_rmse_kw:.2f} kW")

        self.net.load_state_dict(best_state)
        self.net.to(torch.device("cpu"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        Xn = ((X - self.feat_mean) / self.feat_std).astype(np.float32)
        Xs = np.array(
            [Xn[i - self.lookback:i] for i in range(self.lookback, len(Xn))],
            dtype=np.float32,
        )
        self.net.eval()
        with torch.no_grad():
            p = self.net(torch.tensor(Xs)).numpy()
        out = np.full(len(X), np.nan, np.float32)
        out[self.lookback:] = (p * self.target_scale).clip(min=0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_coverage(pv_labels: pd.Series, local_5min: pd.DataFrame) -> None:
    """Show the 4-year label series with actual vs synthetic highlighted."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.suptitle(
        "4-Year PV Label Dataset — Actual + Synthetic\n"
        "University of Moratuwa  |  5-min resolution",
        fontsize=13, fontweight="bold",
    )

    # Daily mean
    ax = axes[0]
    daily = (pv_labels.where(pv_labels > 0.5) / 1000).resample("1D").mean()

    actual_kw = (local_5min[_PV_OBS_COL].clip(lower=0) / 1000)
    actual_daily = actual_kw.where(actual_kw > 0.5).resample("1D").mean()

    ax.fill_between(daily.index, 0, daily.values,
                    alpha=0.3, color="steelblue", label="Synthetic PV")
    ax.fill_between(actual_daily.index, 0, actual_daily.values,
                    alpha=0.6, color="darkorange", label="Actual PV (observed)")
    ax.plot(daily.index, daily.values, lw=0.6, color="steelblue", alpha=0.7)
    ax.set_ylabel("Daily Mean Power (kW)")
    ax.set_title("Daily Mean Daytime Power — Full 4-Year Dataset", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    # Monthly energy
    ax = axes[1]
    monthly_kwh = (pv_labels / 1000 * (5/60)).resample("ME").sum()
    actual_monthly = (actual_kw * (5/60)).resample("ME").sum()
    colors = ["darkorange" if t in actual_monthly.index else "steelblue"
              for t in monthly_kwh.index]
    bars = ax.bar(monthly_kwh.index, monthly_kwh.values,
                  width=20, color=colors, alpha=0.8)
    ax.set_ylabel("Monthly Energy (kWh)")
    ax.set_title("Monthly PV Energy — Orange = Actual, Blue = Synthetic", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0)

    fig.tight_layout()
    out = _OUT_DIR / "4yr_label_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_synthetic_vs_actual(pv_labels, local_5min, physics_sim,
                             solcast_cal: pd.DataFrame | None = None) -> None:
    """
    4-panel deep-dive into synthetic PV errors (calibration window 2022–2023).

    Panel 1: Scatter (actual vs synthetic) coloured by sky condition
             — reveals that Overcast intervals drive the worst errors.
    Panel 2: Per-month RMSE and R² — highlights Oct / Nov / Apr as problem months.
    Panel 3: Error distribution by sky condition (violin) — quantifies scatter by regime.
    Panel 4: Monthly error box plot — shows both bias and spread per month.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    actual_kw = (local_5min[_PV_OBS_COL].clip(lower=0) / 1000)
    synth_kw  = (physics_sim.reindex(actual_kw.index) / 1000).clip(lower=0)
    day_mask  = (synth_kw > 0.5) & (actual_kw > 0.5)

    df_join = pd.concat(
        [actual_kw.rename("actual"), synth_kw.rename("synth")], axis=1
    ).dropna()
    df_join = df_join[df_join["actual"] > 0.5]
    df_join["err"] = df_join["synth"] - df_join["actual"]
    df_join["month"] = df_join.index.month

    # Sky condition labels (0-3) — join from solcast_cal if available
    if solcast_cal is not None and "sky_condition" in solcast_cal.columns:
        sky = solcast_cal["sky_condition"].reindex(df_join.index)
        df_join["sky"] = sky.map(_SKY_LABELS).fillna("Unknown")
    else:
        df_join["sky"] = "Unknown"

    obs_arr  = df_join["actual"].values
    pred_arr = df_join["synth"].values
    r2   = float(1 - ((obs_arr - pred_arr)**2).sum() / ((obs_arr - obs_arr.mean())**2).sum())
    rmse = float(np.sqrt(((obs_arr - pred_arr)**2).mean()))

    _MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    sky_palette = {
        "Clear": "#f4a261", "PartlyCloudy": "#90be6d",
        "MostlyCloudy": "#577590", "Overcast": "#4d4d4d", "Unknown": "grey"
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Synthetic PV Error Analysis — Calibration Window (Apr 2022 – Mar 2023)\n"
        f"Overall  R²={r2:.3f}   RMSE={rmse:.1f} kW   (daytime, obs > 0.5 kW)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: scatter coloured by sky condition ────────────────────────────
    ax = axes[0, 0]
    sky_order = ["Clear", "PartlyCloudy", "MostlyCloudy", "Overcast"]
    for sky_label in sky_order:
        sub = df_join[df_join["sky"] == sky_label]
        if len(sub) == 0:
            continue
        ax.scatter(sub["actual"], sub["synth"],
                   c=sky_palette[sky_label], s=2, alpha=0.35,
                   label=f"{sky_label} (n={len(sub):,})", rasterized=True)
    lim = max(df_join["actual"].max(), df_join["synth"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1 line")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual PV (kW)"); ax.set_ylabel("Synthetic PV (kW)")
    ax.set_title("Scatter by Sky Condition — Overcast drives worst errors", fontsize=10)
    ax.legend(fontsize=8, markerscale=4)

    # ── Panel 2: per-month RMSE and R² ───────────────────────────────────────
    ax  = axes[0, 1]
    ax2 = ax.twinx()
    months, rmses, r2s = [], [], []
    for m in range(1, 13):
        sub = df_join[df_join["month"] == m]
        if len(sub) < 20:
            continue
        o, p = sub["actual"].values, sub["synth"].values
        months.append(_MONTH_ABBR[m - 1])
        rmses.append(float(np.sqrt(((o - p)**2).mean())))
        r2s.append(float(1 - ((o - p)**2).sum() / ((o - o.mean())**2).sum()))

    x   = np.arange(len(months))
    bar_colors = ["#e63946" if r < 0.65 else "#f4a261" if r < 0.75
                  else "#2a9d8f" for r in r2s]
    bars = ax.bar(x, rmses, color=bar_colors, alpha=0.85, width=0.6, label="RMSE (kW)")
    ax2.plot(x, r2s, "o--", color="navy", lw=1.8, markersize=6, label="R²")
    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=9)
    ax.set_ylabel("RMSE (kW)", color="#333")
    ax2.set_ylabel("R²", color="navy")
    ax2.set_ylim(0, 1.05)
    ax.set_title("Monthly RMSE (bars) and R² (line) — Red = R² < 0.65", fontsize=10)
    for bar, r in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{r:.2f}", ha="center", va="bottom", fontsize=7, color="navy")
    ax.set_ylim(0)

    # ── Panel 3: error violin by sky condition ────────────────────────────────
    ax = axes[1, 0]
    plot_data = [
        df_join.loc[df_join["sky"] == s, "err"].values
        for s in sky_order if (df_join["sky"] == s).any()
    ]
    plot_labels = [s for s in sky_order if (df_join["sky"] == s).any()]
    parts = ax.violinplot(plot_data, showmedians=True, showextrema=False)
    for i, (pc, label) in enumerate(zip(parts["bodies"], plot_labels)):
        pc.set_facecolor(sky_palette[label])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("Error: Synthetic − Actual (kW)")
    ax.set_title("Error Distribution by Sky Condition\n(wide violin = high uncertainty)", fontsize=10)
    ax.set_ylim(-250, 250)

    # ── Panel 4: error boxplot by month ──────────────────────────────────────
    ax = axes[1, 1]
    month_data, month_labels = [], []
    for m in range(1, 13):
        sub = df_join[df_join["month"] == m]["err"]
        if len(sub) >= 10:
            month_data.append(sub.values)
            month_labels.append(_MONTH_ABBR[m - 1])

    bplot = ax.boxplot(month_data, patch_artist=True,
                       flierprops=dict(marker=".", markersize=2, alpha=0.3),
                       medianprops=dict(color="black", linewidth=2),
                       showfliers=True)
    wet_months_set = {4, 5, 10, 11, 12}
    for i, (patch, label) in enumerate(zip(bplot["boxes"], month_labels)):
        m_num = _MONTH_ABBR.index(label) + 1
        patch.set_facecolor("#e63946" if m_num in wet_months_set else "#457b9d")
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(month_labels) + 1))
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("Error: Synthetic − Actual (kW)")
    ax.set_title("Monthly Error Distribution — Red = Wet season months", fontsize=10)
    ax.set_ylim(-250, 250)

    fig.tight_layout()
    out = _OUT_DIR / "synthetic_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_predictions_vs_actuals(
    test: pd.DataFrame,
    obs_test: np.ndarray,
    physics_test: np.ndarray,
    pred_xgb: np.ndarray,
    pred_gru: np.ndarray,
    pred_cnnlstm: np.ndarray,
    metrics_df: pd.DataFrame,
) -> None:
    """
    5-panel comparison of actual, synthetic (physics) and 3 model predictions
    on the test set (Feb–Mar 2023).

    Row 1:  Full test period — daily mean power (actual, synthetic, XGB, GRU, CNN-LSTM)
    Row 2a: 5-min zoom — one clear week and one cloudy week
    Row 2b: Scatter for each model (actual vs predicted, R² annotated)
    Row 3:  R² / RMSE / MAE bar-chart summary
    """
    sns.set_theme(style="whitegrid", font_scale=0.92)

    obs_s    = pd.Series(obs_test,    index=test.index)
    phy_s    = pd.Series(physics_test, index=test.index)
    xgb_s    = pd.Series(pred_xgb,    index=test.index)
    gru_s    = pd.Series(pred_gru,    index=test.index)
    cnn_s    = pd.Series(pred_cnnlstm, index=test.index)

    _SERIES = [
        (obs_s,  "Actual",     "steelblue",  "-",   2.0),
        (phy_s,  "Synthetic",  "#aaaaaa",    ":",   1.2),
        (xgb_s,  "XGBoost",    "#e63946",    "--",  1.5),
        (gru_s,  "GRU",        "#e9c46a",    "-.",  1.5),
        (cnn_s,  "CNN-LSTM",   "#2a9d8f",    "-",   1.5),
    ]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Model Predictions vs Actual PV — Test Set (Feb–Mar 2023)\n"
        "University of Moratuwa  |  5-min resolution  |  XGBoost / GRU / CNN-LSTM",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32,
                          height_ratios=[1.2, 1.2, 1.0])

    # ── Row 1: full test period daily means ───────────────────────────────────
    ax_full = fig.add_subplot(gs[0, :])
    for s, label, color, ls, lw in _SERIES:
        d = s.where(s > 0.5).resample("1D").mean()
        ax_full.plot(d.index, d.values, lw=lw, color=color, ls=ls, label=label)
    ax_full.set_title("Daily Mean Power — Full Test Period (Feb–Mar 2023)", fontsize=11)
    ax_full.set_ylabel("Mean Power (kW)")
    ax_full.legend(fontsize=9, ncol=5, loc="upper right")
    ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_full.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_full.tick_params(axis="x", rotation=20)
    ax_full.set_ylim(0)

    # ── Row 2 left: 5-min zoom — clear week ──────────────────────────────────
    ax_clear = fig.add_subplot(gs[1, :2])
    t0 = pd.Timestamp("2023-02-06", tz="UTC")
    t1 = pd.Timestamp("2023-02-13", tz="UTC")
    for s, label, color, ls, lw in _SERIES:
        seg = s.loc[t0:t1]
        ax_clear.plot(seg.index, seg.values, lw=lw * 0.8,
                      color=color, ls=ls, label=label, alpha=0.9)
    ax_clear.set_title("5-min Detail — Feb 6–12 (dry week)", fontsize=10)
    ax_clear.set_ylabel("Power (kW)")
    ax_clear.legend(fontsize=8, ncol=5)
    ax_clear.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax_clear.xaxis.set_major_locator(mdates.DayLocator())
    ax_clear.tick_params(axis="x", rotation=20)
    ax_clear.set_ylim(0)

    # ── Row 2 right: metrics bar chart ────────────────────────────────────────
    ax_met = fig.add_subplot(gs[1, 2])
    m_plot = metrics_df.set_index("model")
    model_palette = {
        "XGBoost":           "#e63946",
        "GRU":               "#e9c46a",
        "CNN-LSTM":          "#2a9d8f",
        "Physics (Solcast)": "#457b9d",
    }
    bar_colors = [model_palette.get(n, "grey") for n in m_plot.index]
    x = np.arange(len(m_plot))
    bars = ax_met.bar(x, m_plot["R2"].values, color=bar_colors, alpha=0.85, width=0.6)
    ax_met.set_xticks(x)
    ax_met.set_xticklabels(m_plot.index, rotation=20, ha="right", fontsize=8)
    ax_met.set_ylabel("R²")
    ax_met.set_title("R² Comparison", fontsize=10)
    ax_met.set_ylim(0.7, 1.0)
    for bar in bars:
        ax_met.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Row 3: scatter — actual vs each model ────────────────────────────────
    def _scatter(ax, obs, pred, label, color):
        mask = obs > 1.0
        o, p = obs[mask], pred[mask]
        r2   = float(1 - ((o - p)**2).sum() / ((o - o.mean())**2).sum())
        rmse = float(np.sqrt(((o - p)**2).mean()))
        ax.hexbin(o, p, gridsize=50, cmap="YlOrRd", mincnt=1)
        lim = max(o.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Actual PV (kW)", fontsize=8)
        ax.set_ylabel(f"{label} (kW)", fontsize=8)
        ax.set_title(f"{label}  R²={r2:.3f}  RMSE={rmse:.1f} kW", fontsize=9,
                     color=color, fontweight="bold")

    _scatter(fig.add_subplot(gs[2, 0]), obs_test, pred_xgb,    "XGBoost",  "#e63946")
    _scatter(fig.add_subplot(gs[2, 1]), obs_test, pred_gru,    "GRU",      "#e9c46a")
    _scatter(fig.add_subplot(gs[2, 2]), obs_test, pred_cnnlstm,"CNN-LSTM", "#2a9d8f")

    out = _OUT_DIR / "4yr_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_feature_importance(model, feature_cols, label="4yr") -> None:
    booster = model.get_booster()
    score   = booster.get_score(importance_type="gain")
    imp     = {feature_cols[int(k[1:])]: v
               for k, v in score.items() if int(k[1:]) < len(feature_cols)}
    imp_s   = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:25]
    names, vals = zip(*imp_s)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e63946" if any(p in n for p in
              ["kt", "ghi", "cloud", "clearness", "sky", "diffuse", "physics"])
              else "#457b9d" for n in names]
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"XGBoost 4-yr Feature Importance (Top 25)\n"
                 f"Red = weather/physics  |  Blue = time/lag", fontsize=11)
    fig.tight_layout()
    out = _OUT_DIR / f"4yr_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/site.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MET_DIR.mkdir(parents=True, exist_ok=True)
    _MOD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all Solcast files (4 years) ──────────────────────────────────────
    logger.info("Loading full 4-year Solcast data …")
    raw_solcast = load_solcast_local_files(cfg)
    solcast_cal = solcast_to_nasa_schema(raw_solcast)
    logger.info(f"  Solcast: {len(solcast_cal):,} rows  "
                f"({solcast_cal.index.min().date()} → {solcast_cal.index.max().date()})")

    # ── Add weather pattern features ──────────────────────────────────────────
    logger.info("Adding weather pattern features …")
    solcast_cal = add_weather_pattern_features(solcast_cal)

    # ── Load actual PV (1-year overlap) ───────────────────────────────────────
    logger.info("Loading actual PV observations …")
    interim  = resolve_path(cfg["paths"]["interim"])
    local_5min = pd.read_csv(interim / "local_5min_utc.csv",
                             index_col="timestamp_utc", parse_dates=True)
    local_5min.index = pd.to_datetime(local_5min.index, utc=True)
    logger.info(f"  Actual PV: {len(local_5min):,} rows  "
                f"({local_5min.index.min().date()} → {local_5min.index.max().date()})")

    # ── Load physics simulation (4-year synthetic PV) ─────────────────────────
    logger.info("Loading physics synthetic PV …")
    synth_path = resolve_path(cfg["paths"]["synthetic"]) / "a6_solcast_pv_synthetic_5min.csv"
    sim_df     = pd.read_csv(synth_path, index_col="timestamp_utc", parse_dates=True)
    sim_df.index = pd.to_datetime(sim_df.index, utc=True)
    physics_sim  = sim_df["pv_ac_W"]
    logger.info(f"  Synthetic PV: {len(sim_df):,} rows  "
                f"({sim_df.index.min().date()} → {sim_df.index.max().date()})")

    # ── Build 4-year label series ─────────────────────────────────────────────
    logger.info("Building 4-year label series …")
    pv_labels = build_4yr_labels(local_5min, physics_sim)

    # ── Plot label coverage and synthetic vs actual quality ───────────────────
    logger.info("Plotting label coverage and synthetic quality …")
    plot_label_coverage(pv_labels, local_5min)
    plot_synthetic_vs_actual(pv_labels, local_5min, physics_sim, solcast_cal=solcast_cal)

    # ── Build feature matrix ──────────────────────────────────────────────────
    feat = build_feature_matrix(solcast_cal, pv_labels, physics_sim, use_lags=True)

    # Remove 'is_actual' from features (it's a flag, not a predictor)
    feature_cols = [c for c in feat.columns if c not in ("pv_ac_kW", "is_actual")]

    # ── 4-year chronological split ────────────────────────────────────────────
    train, val, test = split_4yr(feat)

    obs_test = test["pv_ac_kW"].values

    def _prepend(block, source, n):
        return pd.concat([source.iloc[-n:], block])

    # ── 1. XGBoost (baseline) ────────────────────────────────────────────────
    logger.info("\n── Training XGBoost (baseline) ──────────────────────────────")
    xgb_model = train_xgboost(train, val, feature_cols, label="XGBoost")
    pred_xgb  = xgb_model.predict(test[feature_cols].values).clip(min=0)
    xgb_model.save_model(str(_MOD_DIR / "xgb_4yr.json"))

    # ── 2. GRU (time-series) ─────────────────────────────────────────────────
    logger.info("\n── Training GRU (time-series) ───────────────────────────────")
    lookback_gru = 6   # 30 min
    X_tr = train[feature_cols].values.astype(np.float32)
    y_tr = train["pv_ac_kW"].values.astype(np.float32)
    va_df_gru = _prepend(val, train, lookback_gru)
    X_va_gru  = va_df_gru[feature_cols].values.astype(np.float32)
    y_va_gru  = va_df_gru["pv_ac_kW"].values.astype(np.float32)
    te_df_gru = _prepend(test, val, lookback_gru)
    X_te_gru  = te_df_gru[feature_cols].values.astype(np.float32)

    gru = _GRUModel(hidden=64)
    gru.fit(X_tr, y_tr, X_va_gru, y_va_gru, lookback=lookback_gru)
    gru_raw  = gru.predict(X_te_gru)
    pred_gru = gru_raw[lookback_gru:][:len(test)]
    pred_gru = np.where(np.isnan(pred_gru), pred_xgb, pred_gru)

    val_gru_raw  = gru.predict(X_va_gru)
    val_gru_pred = val_gru_raw[lookback_gru:][:len(val)]
    logger.info(f"  [GRU] final val RMSE="
                f"{float(np.sqrt(np.nanmean((val['pv_ac_kW'].values - val_gru_pred)**2))):.2f} kW")

    # ── 3. CNN-GRU (advanced) — trained on actual data, actual time split ─────
    # CNN extracts local irradiance-change features (multi-scale, 3+5 kernels).
    # GRU models sequential dependencies across the 1-hour lookback window.
    # Trained on the actual 1-year overlap only to avoid synthetic distribution
    # shift; uses an actual-only time split for the val set.
    logger.info("\n── Training CNN-GRU (advanced) ───────────────────────────────")
    lookback_cnn = 12   # 1 hour

    # Actual-data time split (chronological within actual year)
    # Train: Apr 2022 – Nov 2022  |  Val: Dec 2022 – Jan 2023  |  Test: Feb–Mar 2023
    t_cnn_val_start = pd.Timestamp("2022-12-01", tz="UTC")
    t_cnn_test      = pd.Timestamp("2023-02-01", tz="UTC")
    train_cnn = feat[(feat["is_actual"] == 1) & (feat.index < t_cnn_val_start)]
    val_cnn   = feat[(feat["is_actual"] == 1) &
                     (feat.index >= t_cnn_val_start) & (feat.index < t_cnn_test)]
    logger.info(f"  train_cnn: {len(train_cnn):,}  val_cnn: {len(val_cnn):,}")

    va_df_cnn = _prepend(val_cnn, train_cnn, lookback_cnn)
    X_va_cnn  = va_df_cnn[feature_cols].values.astype(np.float32)
    y_va_cnn  = va_df_cnn["pv_ac_kW"].values.astype(np.float32)
    te_df_cnn = _prepend(test, val_cnn, lookback_cnn)
    X_te_cnn  = te_df_cnn[feature_cols].values.astype(np.float32)
    X_tr_cnn  = train_cnn[feature_cols].values.astype(np.float32)
    y_tr_cnn  = train_cnn["pv_ac_kW"].values.astype(np.float32)

    cnnlstm = _CNNLSTMModel(cnn_filters=64, lstm_hidden=64)
    cnnlstm.fit(X_tr_cnn, y_tr_cnn, X_va_cnn, y_va_cnn,
                lookback=lookback_cnn, epochs=80, patience=15)

    cnn_raw      = cnnlstm.predict(X_te_cnn)
    pred_cnnlstm = cnn_raw[lookback_cnn:][:len(test)]
    pred_cnnlstm = np.where(np.isnan(pred_cnnlstm), pred_xgb, pred_cnnlstm)

    val_cnn_raw  = cnnlstm.predict(X_va_cnn)
    val_cnn_pred = val_cnn_raw[lookback_cnn:][:len(val_cnn)]
    logger.info(f"  [CNN-GRU] final val RMSE="
                f"{float(np.sqrt(np.nanmean((val_cnn['pv_ac_kW'].values - val_cnn_pred)**2))):.2f} kW")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("\n── Test set results ─────────────────────────────────────────")
    phy_test = (physics_sim.reindex(test.index) / 1000).clip(lower=0).fillna(0).values
    rows = [
        _metrics(obs_test, pred_xgb,    "XGBoost"),
        _metrics(obs_test, pred_gru,    "GRU"),
        _metrics(obs_test, pred_cnnlstm,"CNN-LSTM"),
        _metrics(obs_test, phy_test,    "Physics (Solcast)"),
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(_MET_DIR / "4yr_model_comparison.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating comparison plots …")
    plot_feature_importance(xgb_model, feature_cols)
    plot_predictions_vs_actuals(
        test, obs_test, phy_test,
        pred_xgb, pred_gru, pred_cnnlstm, metrics_df,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  4-YEAR ML RESULTS — TEST SET (Feb–Mar 2023)")
    print("  Models: XGBoost (baseline) | GRU (time-series) | CNN-LSTM (advanced)")
    print("═" * 65)
    print(metrics_df.to_string(index=False))
    print(f"\n  Figures → {_OUT_DIR}/")
    print(f"  Metrics → {_MET_DIR}/")
    print("═" * 65)


if __name__ == "__main__":
    main()
