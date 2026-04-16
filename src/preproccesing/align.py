"""
src/preproccesing/align.py
--------------------------
Aligns local PV/weather measurements with NASA POWER satellite data.

Both sources differ in:
  - Timezone  : local data is Asia/Colombo (UTC+5:30); NASA POWER is UTC
  - Frequency : local data is 5-minute intervals; NASA POWER is hourly
  - Coverage  : local ~1 year (2022–2023); NASA POWER 2020–2026

Steps performed
---------------
1. Split columns — measurement cols (power, temp, humidity …) vs. status/flag
   cols vs. all-NaN cols (string fields coerced to NaN by load_local_data)
2. Deduplicate local index
3. Convert local timezone to UTC; rename index to "timestamp_utc"
4. Resample 5-min → hourly
     • measurements : mean(), masking hours with < _MIN_SAMPLES_PER_HOUR as NaN
     • status flags : last() — most recent operational state in the window
5. Clip both DataFrames to their temporal overlap
6. Snap NASA grid to exact hourly boundaries
7. Warn if indices do not align exactly
8. Save to data/interim/ and return

Usage
-----
    from src.utils.config import load_config
    from src.preproccesing.align import align_datasets, save_aligned, load_aligned

    cfg         = load_config()
    local_df    = load_local_data("data/raw/...", cfg)
    nasa_df     = nasa_power.load_raw(cfg)

    local_hourly, nasa_aligned = align_datasets(local_df, nasa_df, cfg)
    save_aligned(local_hourly, nasa_aligned, cfg)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Hours with fewer than this many valid 5-min samples are masked to NaN.
# Data analysis showed the distribution is bimodal: 7,851 hours with all 12
# samples present, and 41 hours with 2–11 samples (partial/corrupted), plus
# 869 hours with 0 (logger down).  Requiring all 12 rejects only 30 more
# hours than threshold=6 (0.3% of data) while guaranteeing every hourly
# mean is computed from a full 12-point window.
_MIN_SAMPLES_PER_HOUR: int = 12

# Column name substrings that identify binary operational-status flags.
_STATUS_COL_PATTERNS = ("Status", "Communication")

# Primary PV power column used for diagnostic plots.
_PV_POWER_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def align_datasets(
    local_df: pd.DataFrame,
    nasa_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align local 5-min measurements with hourly NASA POWER data.

    Parameters
    ----------
    local_df : pd.DataFrame
        Output of load_local_data() — timezone-aware DatetimeIndex named
        "timestamp_local" in Asia/Colombo, 5-minute frequency.
    nasa_df : pd.DataFrame
        Output of nasa_power.load_raw() — UTC DatetimeIndex named
        "timestamp_utc", hourly frequency.
    cfg : dict
        Config dict from load_config().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (local_hourly_utc, nasa_aligned) — both with UTC DatetimeIndex
        named "timestamp_utc", clipped to the same overlap window.
    """
    local = local_df.copy()

    # ── Split columns ─────────────────────────────────────────────────────────
    all_nan_cols = [c for c in local.columns if local[c].isna().all()]
    status_cols  = [
        c for c in local.columns
        if any(p in c for p in _STATUS_COL_PATTERNS) and c not in all_nan_cols
    ]
    meas_cols = [
        c for c in local.columns
        if c not in all_nan_cols and c not in status_cols
    ]

    local = local.drop(columns=all_nan_cols)

    logger.info(
        f"Columns — measurement: {len(meas_cols)}, "
        f"status: {len(status_cols)}, "
        f"dropped (all-NaN): {len(all_nan_cols)}"
    )

    # ── Deduplicate ───────────────────────────────────────────────────────────
    n_before = len(local)
    local = local[~local.index.duplicated(keep="first")]
    if len(local) < n_before:
        logger.warning(f"Removed {n_before - len(local):,} duplicate timestamps")

    # ── Convert local → UTC ───────────────────────────────────────────────────
    if local.index.tz is None:
        raise ValueError(
            "local_df has a timezone-naive index. "
            "Ensure load_local_data() localizes the timezone before calling align_datasets()."
        )
    local_utc = local.tz_convert("UTC")
    local_utc.index.name = "timestamp_utc"
    logger.info(f"Converted to UTC: {local_utc.index.min()} → {local_utc.index.max()}")

    # ── Resample 5-min → hourly ───────────────────────────────────────────────
    # Measurements: mean with minimum-sample enforcement.
    # Note: min_count is not supported on .mean(), so we use a separate .count()
    # call and mask with .where().
    hourly_mean  = local_utc[meas_cols].resample("1h").mean()
    hourly_count = local_utc[meas_cols].resample("1h").count()
    hourly_meas  = hourly_mean.where(hourly_count >= _MIN_SAMPLES_PER_HOUR)

    # Status flags: last known operational state within each hour window.
    hourly_status = local_utc[status_cols].resample("1h").last()

    local_hourly = pd.concat([hourly_meas, hourly_status], axis=1)

    # Report masking against the primary PV power column if present; otherwise any column.
    if _PV_POWER_COL in hourly_count.columns:
        masked_hours = (hourly_count[_PV_POWER_COL] < _MIN_SAMPLES_PER_HOUR).sum()
        mask_label   = "PV Power col"
    else:
        masked_hours = (hourly_count < _MIN_SAMPLES_PER_HOUR).all(axis=1).sum()
        mask_label   = "all cols"
    logger.info(
        f"Resampled to {len(local_hourly):,} hourly rows "
        f"({masked_hours} hours masked to NaN [{mask_label}] — fewer than {_MIN_SAMPLES_PER_HOUR} samples)"
    )

    # ── Clip to overlap window ────────────────────────────────────────────────
    t_start = max(local_hourly.index.min(), nasa_df.index.min())
    t_end   = min(local_hourly.index.max(), nasa_df.index.max())

    if t_end < t_start:
        raise ValueError(
            f"No temporal overlap between local data "
            f"({local_hourly.index.min()} – {local_hourly.index.max()}) "
            f"and NASA data ({nasa_df.index.min()} – {nasa_df.index.max()})."
        )

    local_clipped = local_hourly.loc[t_start:t_end]
    nasa_clipped  = nasa_df.loc[t_start:t_end]
    logger.info(f"Overlap window : {t_start} → {t_end}  ({len(local_clipped):,} hours)")

    # ── Snap NASA to hourly grid ──────────────────────────────────────────────
    nasa_aligned = nasa_clipped.resample("1h").asfreq()

    # ── Index alignment check ─────────────────────────────────────────────────
    if not local_clipped.index.equals(nasa_aligned.index):
        mismatches = (~local_clipped.index.isin(nasa_aligned.index)).sum()
        logger.warning(
            f"Index mismatch: {mismatches} local timestamps not found in NASA index. "
            "Downstream calibration may produce NaN rows."
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALIGNMENT COMPLETE")
    logger.info(f"  Local  : {len(local_clipped):,} rows × {local_clipped.shape[1]} cols")
    logger.info(f"  NASA   : {len(nasa_aligned):,} rows × {nasa_aligned.shape[1]} cols")
    logger.info(f"  Range  : {local_clipped.index.min()}  →  {local_clipped.index.max()}")
    logger.info("=" * 60)

    return local_clipped, nasa_aligned


def save_aligned(
    local_hourly: pd.DataFrame,
    nasa_aligned: pd.DataFrame,
    cfg: dict,
) -> None:
    """
    Save aligned DataFrames to data/interim/ as CSV.

    Parameters
    ----------
    local_hourly : pd.DataFrame
        Hourly UTC local data from align_datasets().
    nasa_aligned : pd.DataFrame
        Clipped hourly NASA data from align_datasets().
    cfg : dict
        Config dict from load_config().
    """
    out_dir = resolve_path(cfg["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path = out_dir / "local_hourly_utc.csv"
    nasa_path  = out_dir / "nasa_aligned.csv"

    local_hourly.to_csv(local_path)
    nasa_aligned.to_csv(nasa_path)

    logger.info(f"Saved → {local_path}  ({local_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Saved → {nasa_path}  ({nasa_path.stat().st_size / 1024:.1f} KB)")


def load_aligned(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved aligned DataFrames from data/interim/.

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (local_hourly_utc, nasa_aligned) with UTC-aware DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        If either interim CSV is missing (run scripts/run_align.py first).
    """
    interim    = resolve_path(cfg["paths"]["interim"])
    local_path = interim / "local_hourly_utc.csv"
    nasa_path  = interim / "nasa_aligned.csv"

    for path in (local_path, nasa_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Aligned file not found: {path}\n"
                "Run  python scripts/run_align.py  to generate interim data."
            )

    local_df = pd.read_csv(local_path, index_col="timestamp_utc", parse_dates=True)
    local_df.index = pd.to_datetime(local_df.index, utc=True)

    nasa_df = pd.read_csv(nasa_path, index_col="timestamp_utc", parse_dates=True)
    nasa_df.index = pd.to_datetime(nasa_df.index, utc=True)

    logger.info(f"Loaded local : {len(local_df):,} rows from {local_path.name}")
    logger.info(f"  Range: {local_df.index.min()} → {local_df.index.max()}")
    logger.info(f"Loaded NASA  : {len(nasa_df):,} rows from {nasa_path.name}")
    logger.info(f"  Range: {nasa_df.index.min()} → {nasa_df.index.max()}")

    return local_df, nasa_df


def align_solcast_5min(
    local_raw:   pd.DataFrame,
    solcast_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align 5-minute local PV data with 5-minute Solcast data.

    Unlike align_datasets(), this function does NOT resample — it keeps the
    full 5-minute resolution throughout.  The local data is timezone-converted
    to UTC and both DataFrames are clipped to their common overlap window.

    Parameters
    ----------
    local_raw : pd.DataFrame
        Raw local PV data with a timezone-aware DatetimeIndex (Asia/Colombo).
        Timestamps represent period-end (e.g. 00:05:00 = end of 00:00–00:05).
    solcast_df : pd.DataFrame
        Solcast 5-min data with UTC DatetimeIndex (period-start convention
        after _parse_solcast_local_csv shift).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (local_5min_utc, solcast_clipped) — both UTC, 5-min, same window.
    """
    local = local_raw.copy()

    # Deduplicate
    n_before = len(local)
    local = local[~local.index.duplicated(keep="first")]
    if len(local) < n_before:
        logger.warning(f"Removed {n_before - len(local):,} duplicate timestamps (5-min)")

    # Convert local Asia/Colombo → UTC
    if local.index.tz is None:
        raise ValueError("local_raw has a timezone-naive index. Localize before calling.")
    local_utc           = local.tz_convert("UTC")
    local_utc.index.name = "timestamp_utc"

    # Convert local period-end timestamps to period-start (subtract 5 min)
    # so they align with Solcast's period-start index convention.
    local_utc.index = local_utc.index - pd.Timedelta(minutes=5)

    logger.info(
        f"Local 5-min UTC range: {local_utc.index.min()} → {local_utc.index.max()}  "
        f"({len(local_utc):,} rows)"
    )

    # Clip to overlap
    t_start = max(local_utc.index.min(), solcast_df.index.min())
    t_end   = min(local_utc.index.max(), solcast_df.index.max())

    if t_end < t_start:
        raise ValueError(
            f"No temporal overlap between local data "
            f"({local_utc.index.min()} – {local_utc.index.max()}) "
            f"and Solcast data ({solcast_df.index.min()} – {solcast_df.index.max()})."
        )

    local_clipped   = local_utc.loc[t_start:t_end]
    solcast_clipped = solcast_df.loc[t_start:t_end]

    logger.info("=" * 60)
    logger.info("5-MIN ALIGNMENT COMPLETE")
    logger.info(f"  Local   : {len(local_clipped):,} rows × {local_clipped.shape[1]} cols")
    logger.info(f"  Solcast : {len(solcast_clipped):,} rows × {solcast_clipped.shape[1]} cols")
    logger.info(f"  Range   : {t_start}  →  {t_end}")
    logger.info("=" * 60)

    return local_clipped, solcast_clipped


def save_aligned_5min(
    local_5min:   pd.DataFrame,
    solcast_5min: pd.DataFrame,
    cfg:          dict,
) -> None:
    """Save 5-min aligned DataFrames to data/interim/."""
    out_dir = resolve_path(cfg["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path   = out_dir / "local_5min_utc.csv"
    solcast_path = out_dir / "solcast_5min_aligned.csv"

    local_5min.to_csv(local_path)
    solcast_5min.to_csv(solcast_path)

    logger.info(f"Saved → {local_path}  ({local_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Saved → {solcast_path}  ({solcast_path.stat().st_size / 1024:.1f} KB)")


def load_aligned_5min(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved 5-min aligned DataFrames from data/interim/."""
    interim      = resolve_path(cfg["paths"]["interim"])
    local_path   = interim / "local_5min_utc.csv"
    solcast_path = interim / "solcast_5min_aligned.csv"

    for path in (local_path, solcast_path):
        if not path.exists():
            raise FileNotFoundError(
                f"5-min aligned file not found: {path}\n"
                "Run  python scripts/run_pv_model.py --source solcast  to generate."
            )

    local_df = pd.read_csv(local_path, index_col="timestamp_utc", parse_dates=True)
    local_df.index = pd.to_datetime(local_df.index, utc=True)

    solcast_df = pd.read_csv(solcast_path, index_col="timestamp_utc", parse_dates=True)
    solcast_df.index = pd.to_datetime(solcast_df.index, utc=True)

    logger.info(f"Loaded local 5-min  : {len(local_df):,} rows")
    logger.info(f"Loaded Solcast 5-min: {len(solcast_df):,} rows")
    return local_df, solcast_df


def plot_alignment(
    local_raw: pd.DataFrame,
    local_hourly: pd.DataFrame,
    nasa_aligned: pd.DataFrame,
    cfg: dict,
) -> None:
    """
    Generate and save two diagnostic plots after alignment.

    Figure 1 — Before vs After resampling (first 7 days of overlap):
      Top    : raw 5-min PV power in UTC
      Bottom : hourly PV power after resampling

    Figure 2 — Local vs NASA full overlap (2×2):
      Top-left    : temperature (local tempC vs NASA T2M)
      Top-right   : humidity (local humidity vs NASA RH2M)
      Bottom-left : NASA GHI (ALLSKY_SFC_SW_DWN)
      Bottom-right: local PV power vs NASA GHI (dual y-axis)

    Parameters
    ----------
    local_raw : pd.DataFrame
        Original 5-min local DataFrame (before resampling).
    local_hourly : pd.DataFrame
        Hourly UTC local data from align_datasets().
    nasa_aligned : pd.DataFrame
        Clipped hourly NASA data from align_datasets().
    cfg : dict
        Config dict from load_config().
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = resolve_path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Convert raw local to UTC for consistent x-axis comparison
    raw_utc = (
        local_raw.tz_convert("UTC")
        if local_raw.index.tz is not None
        else local_raw
    )
    raw_utc.index.name = "timestamp_utc"

    t_start = local_hourly.index.min()
    t_7d    = t_start + pd.Timedelta(days=7)
    pv_col  = _PV_POWER_COL

    # ── Figure 1: Before vs After resampling ─────────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig1.suptitle(
        "Before vs After Resampling — PV Power (First 7 Days of Overlap)",
        fontsize=12, fontweight="bold"
    )

    if pv_col in raw_utc.columns and pv_col in local_hourly.columns:
        raw_win    = raw_utc.loc[t_start:t_7d, pv_col]
        hourly_win = local_hourly.loc[t_start:t_7d, pv_col]

        axes[0].plot(raw_win.index, raw_win.values,
                     linewidth=0.5, color="steelblue", alpha=0.85)
        axes[0].set_title("Raw 5-min data (UTC)")
        axes[0].set_ylabel("Power (W)")

        axes[1].plot(hourly_win.index, hourly_win.values,
                     linewidth=1.2, color="darkorange",
                     marker="o", markersize=2.5)
        axes[1].set_title(f"Hourly mean (UTC) — NaN gaps where < {_MIN_SAMPLES_PER_HOUR} samples")
        axes[1].set_ylabel("Power (W)")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, f"Column not found:\n{pv_col}",
                    ha="center", va="center", transform=ax.transAxes, color="grey")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig1.tight_layout()
    out1 = fig_dir / "data_pre_align_before_after_resample.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info(f"Saved → {out1}")

    # ── Figure 2: Local vs NASA after alignment ───────────────────────────────
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle(
        "Post-Alignment: Local vs NASA POWER (Full Overlap Period)",
        fontsize=12, fontweight="bold"
    )

    # Top-left: temperature
    ax = axes[0, 0]
    if "tempC" in local_hourly.columns:
        ax.plot(local_hourly.index, local_hourly["tempC"],
                linewidth=0.5, color="tomato", alpha=0.8, label="Local tempC")
    if "T2M" in nasa_aligned.columns:
        ax.plot(nasa_aligned.index, nasa_aligned["T2M"],
                linewidth=0.5, color="navy", alpha=0.7, label="NASA T2M")
    ax.set_title("Temperature")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8)

    # Top-right: humidity
    ax = axes[0, 1]
    if "humidity" in local_hourly.columns:
        ax.plot(local_hourly.index, local_hourly["humidity"],
                linewidth=0.5, color="seagreen", alpha=0.8, label="Local humidity")
    if "RH2M" in nasa_aligned.columns:
        ax.plot(nasa_aligned.index, nasa_aligned["RH2M"],
                linewidth=0.5, color="purple", alpha=0.7, label="NASA RH2M")
    ax.set_title("Relative Humidity")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    # Bottom-left: NASA GHI
    ax = axes[1, 0]
    if "ALLSKY_SFC_SW_DWN" in nasa_aligned.columns:
        ax.plot(nasa_aligned.index, nasa_aligned["ALLSKY_SFC_SW_DWN"],
                linewidth=0.5, color="goldenrod", alpha=0.85)
    ax.set_title("NASA GHI (ALLSKY_SFC_SW_DWN)")
    ax.set_ylabel("W/m²")

    # Bottom-right: PV power vs NASA GHI (dual y-axis)
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    if pv_col in local_hourly.columns:
        ax.plot(local_hourly.index, local_hourly[pv_col],
                linewidth=0.5, color="darkorange", alpha=0.8, label="Local PV Power")
        ax.set_ylabel("PV Power (W)", color="darkorange")
    if "ALLSKY_SFC_SW_DWN" in nasa_aligned.columns:
        ax2.plot(nasa_aligned.index, nasa_aligned["ALLSKY_SFC_SW_DWN"],
                 linewidth=0.5, color="goldenrod", alpha=0.65, label="NASA GHI")
        ax2.set_ylabel("GHI (W/m²)", color="goldenrod")
    lines  = ax.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, fontsize=8)
    ax.set_title("Local PV Power vs NASA GHI")

    for row in axes:
        for a in row:
            a.tick_params(axis="x", rotation=25)

    fig2.tight_layout()
    out2 = fig_dir / "data_pre_align_local_vs_nasa.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved → {out2}")
