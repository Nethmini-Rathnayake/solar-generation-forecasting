"""
src/data/local_loader.py
-------------------------
Loads and performs minimal normalisation on the locally measured PV / weather
CSV data.  The goal here is only to get the data into a clean DataFrame with
a proper DatetimeIndex — heavy cleaning and alignment happen in
src/preprocessing/.

Expected CSV format
-------------------
The loader is intentionally flexible. It accepts any CSV that has:
  - One column containing timestamps (name is configurable)
  - One or more numeric measurement columns

If your file has a different timestamp format or column names, update
configs/site.yaml under the `local_data` section.

Usage
-----
    from src.utils.config import load_config
    from src.data.local_loader import load_local_data

    cfg = load_config()
    df = load_local_data("data/raw/pv_data.csv", cfg)
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_local_data(
    filepath: str | Path,
    cfg: dict | None = None,
    timestamp_col: str | None = None,
    timestamp_format: str | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """
    Load a local measurement CSV into a timezone-aware pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    cfg : dict, optional
        Loaded config dict. Used to read default timezone if not overridden.
    timestamp_col : str, optional
        Name of the column containing timestamps. If None, tries common
        names ('timestamp', 'datetime', 'time', 'date', 'Timestamp').
    timestamp_format : str, optional
        strftime format string for parsing timestamps, e.g. '%Y-%m-%d %H:%M:%S'.
        If None, pandas will infer the format automatically.
    timezone : str, optional
        Timezone name (e.g. 'Asia/Colombo'). If None, reads from cfg.
        If the timestamps are already UTC, pass 'UTC'.

    Returns
    -------
    pd.DataFrame
        DataFrame with a timezone-aware DatetimeIndex named 'timestamp_local'.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If no timestamp column can be found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Local data file not found: {filepath}")

    logger.info(f"Loading local data from: {filepath.name}")

    # ── Read CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    logger.info(f"  Raw shape: {df.shape} | Columns: {list(df.columns)}")

    # ── Identify timestamp column ──────────────────────────────────────────────
    ts_col = timestamp_col or _find_timestamp_column(df)
    logger.info(f"  Using timestamp column: '{ts_col}'")

    # ── Parse timestamps ───────────────────────────────────────────────────────
    if timestamp_format:
        df[ts_col] = pd.to_datetime(df[ts_col], format=timestamp_format)
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)

    df.set_index(ts_col, inplace=True)
    df.index.name = "timestamp_local"

    # ── Localise timezone ──────────────────────────────────────────────────────
    tz = timezone or (cfg["site"]["timezone"] if cfg else None)
    if tz:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
        logger.info(f"  Timezone set to: {tz}")

    df.sort_index(inplace=True)

    # ── Basic numeric coercion ─────────────────────────────────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"  Loaded {len(df):,} rows | "
        f"{df.index.min()} → {df.index.max()}"
    )
    missing_pct = df.isna().mean().mul(100).round(1)
    if missing_pct.any():
        logger.info(f"  Missing data (%):\n{missing_pct[missing_pct > 0].to_string()}")

    return df


def _find_timestamp_column(df: pd.DataFrame) -> str:
    """
    Heuristically find the timestamp column from common naming patterns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        Column name.

    Raises
    ------
    ValueError
        If no candidate column is found.
    """
    candidates = [
        "timestamp", "Timestamp", "TIMESTAMP",
        "datetime", "Datetime", "DateTime", "DATETIME",
        "time", "Time", "TIME",
        "date", "Date", "DATE",
        "date_time", "Date_Time",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Last resort: look for any column whose name contains 'time' or 'date'
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            logger.warning(f"  No standard timestamp column found; using '{col}'")
            return col

    raise ValueError(
        f"Cannot identify a timestamp column. Columns found: {list(df.columns)}. "
        "Pass timestamp_col='your_column_name' explicitly."
    )


def describe_local_data(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of the local dataset.
    Useful as the first cell in an exploration notebook.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_local_data().
    """
    print("=" * 60)
    print("LOCAL DATASET SUMMARY")
    print("=" * 60)
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Start      : {df.index.min()}")
    print(f"  End        : {df.index.max()}")
    print(f"  Timezone   : {df.index.tz}")
    freq = pd.infer_freq(df.index)
    print(f"  Inferred frequency: {freq}")
    print()
    print("Missing values:")
    print(df.isna().sum().to_frame("missing").assign(pct=lambda x: (x["missing"] / len(df) * 100).round(2)))
    print()
    print("Numeric statistics:")
    print(df.describe().round(3))
    print("=" * 60)
