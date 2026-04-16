"""
src/data/solcast_loader.py
---------------------------
Downloads and preprocesses Solcast historical radiation data for the
University of Moratuwa site.

Why Solcast?
-------------
Solcast uses a proprietary satellite-based model with ~1–2 km spatial
resolution and 5-minute temporal resolution (resampled here to hourly).
Its cloud-motion forecasting engine resolves sub-mesoscale convective cells
that NASA POWER (0.5°) and even ERA5 (0.25°) cannot detect.  For tropical
coastal sites with frequent isolated convective showers (Sri Lanka monsoon),
Solcast typically achieves 15–30% lower hourly RMSE vs NASA POWER.

Solcast free researcher tier
------------------------------
1. Register at  https://toolkit.solcast.com.au/register/hobbyist
2. You receive a free API key (10 API calls / day).
3. Set environment variable:  export SOLCAST_API_KEY=<your-key>
   OR pass --api-key to scripts/fetch_solcast.py.

Rate-limit note
---------------
Each API call returns up to 31 days.  The full 2020–2025 dataset needs ~72
calls (6 years × 12 months).  At 10 calls/day the download takes ~8 days.
The fetch script is designed to cache monthly chunks and resume interrupted
downloads — run  python scripts/fetch_solcast.py  daily until complete.

Alternatively, request TMY (Typical Meteorological Year) data in a single
call using  --tmy  flag.  TMY gives a synthetic representative year, not the
actual year-by-year record — useful for long-run simulation but not for
evaluating model accuracy against 2022–2023 observations.

Variables downloaded
---------------------
  ghi               Global Horizontal Irradiance       W/m²
  dhi               Diffuse Horizontal Irradiance       W/m²
  dni               Direct Normal Irradiance            W/m²
  air_temp_10m      Air temperature at 10 m             °C
  wind_speed_10m    Wind speed at 10 m                  m/s
  cloud_opacity     Cloud opacity (0–100%)              %

Column mapping
--------------
  Solcast column    NASA POWER equiv         Unit
  ghi               ALLSKY_SFC_SW_DWN        W/m²
  dhi               ALLSKY_SFC_SW_DIFF       W/m²
  dni               ALLSKY_SFC_SW_DNI        W/m²
  air_temp_10m      T2M                      °C
  wind_speed_10m    WS10M                    m/s
  (pvlib clear-sky) CLRSKY_SFC_SW_DWN        W/m²

Usage
-----
    from src.data.solcast_loader import download_solcast_monthly, load_solcast_processed

    download_solcast_monthly(cfg, api_key="your_key", year=2020, month=1)
    df = load_solcast_processed(cfg)
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LAT = 6.7912
_LON = 79.9005

_SOLCAST_HIST_URL = "https://api.solcast.com.au/data/historic/radiation_and_weather"
_SOLCAST_TMY_URL  = "https://api.solcast.com.au/tmy/radiation_and_weather"

_OUTPUT_PARAMS = "ghi,dhi,dni,air_temp_10m,wind_speed_10m,cloud_opacity"


# ─────────────────────────────────────────────────────────────────────────────
# Download — monthly chunks (rate-limit friendly)
# ─────────────────────────────────────────────────────────────────────────────

def download_solcast_monthly(
    cfg:     dict,
    api_key: str | None = None,
    year:    int = 2020,
    month:   int = 1,
) -> Path | None:
    """
    Download one month of Solcast historical data and cache to disk.

    Saves  data/external/solcast/solcast_{year}_{month:02d}.csv
    Returns None if the file already exists (already downloaded).

    Call this function once per month, respecting the 10 calls/day limit.
    """
    key = api_key or os.environ.get("SOLCAST_API_KEY", "")
    if not key:
        raise ValueError(
            "Solcast API key required.\n"
            "1. Register at https://toolkit.solcast.com.au/register/hobbyist\n"
            "2. Set:  export SOLCAST_API_KEY=<your-key>\n"
            "   OR pass  --api-key <key>  to scripts/fetch_solcast.py"
        )

    out_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "solcast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"solcast_{year}_{month:02d}.csv"

    if out_path.exists():
        logger.info(f"  {year}-{month:02d}: already cached — skipping")
        return None

    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00.000Z"
    end   = f"{year}-{month:02d}-{last_day:02d}T23:59:59.000Z"

    logger.info(f"  Requesting Solcast  {year}-{month:02d}  ({start[:10]} → {end[:10]}) …")

    params = {
        "latitude":          _LAT,
        "longitude":         _LON,
        "start":             start,
        "end":               end,
        "period":            "PT60M",
        "output_parameters": _OUTPUT_PARAMS,
        "format":            "csv",
        "api_key":           key,
    }
    resp = requests.get(_SOLCAST_HIST_URL, params=params, timeout=120)

    if resp.status_code == 429:
        logger.warning("  Rate limit hit (10 calls/day). Run again tomorrow.")
        return None
    if resp.status_code == 402:
        raise RuntimeError(
            "Solcast 402: API plan does not support this request.\n"
            "Ensure your account has access to historical data."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Solcast API error {resp.status_code}:\n{resp.text[:500]}"
        )

    out_path.write_bytes(resp.content)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"  Saved → {out_path.name}  ({size_kb:.0f} KB)")
    return out_path


def download_solcast_tmy(
    cfg:     dict,
    api_key: str | None = None,
) -> Path:
    """
    Download Solcast TMY (Typical Meteorological Year) in a single API call.

    TMY is a synthetic representative year, NOT the historical year-by-year
    record.  Use --tmy flag in fetch_solcast.py when you just need a single
    representative dataset and don't want to wait 8+ days for full history.

    Saves  data/external/solcast/solcast_tmy.csv
    """
    key = api_key or os.environ.get("SOLCAST_API_KEY", "")
    if not key:
        raise ValueError(
            "Solcast API key required.\n"
            "Register at https://toolkit.solcast.com.au/register/hobbyist"
        )

    out_dir  = resolve_path(cfg["paths"]["external_nasa"]).parent / "solcast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solcast_tmy.csv"

    if out_path.exists():
        logger.info(f"TMY already cached: {out_path}")
        return out_path

    logger.info("Requesting Solcast TMY …")
    params = {
        "latitude":          _LAT,
        "longitude":         _LON,
        "period":            "PT60M",
        "output_parameters": _OUTPUT_PARAMS,
        "format":            "csv",
        "api_key":           key,
    }
    resp = requests.get(_SOLCAST_TMY_URL, params=params, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Solcast TMY API error {resp.status_code}:\n{resp.text[:500]}"
        )

    out_path.write_bytes(resp.content)
    logger.info(f"Saved TMY → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Download status helper
# ─────────────────────────────────────────────────────────────────────────────

def download_status(cfg: dict, start_year: int = 2020, end_year: int = 2025) -> None:
    """
    Print which months have been downloaded and which are still pending.
    Useful for tracking progress when downloading in daily batches.
    """
    out_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "solcast"
    total = pending = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            total += 1
            path = out_dir / f"solcast_{year}_{month:02d}.csv"
            status = "✓" if path.exists() else "·"
            pending += 0 if path.exists() else 1
        print(f"  {year}: " + "".join(
            "✓" if (out_dir / f"solcast_{year}_{m:02d}.csv").exists() else "·"
            for m in range(1, 13)
        ) + f"  ({12 - pending}/{12})")
    print(f"\n  {total - pending}/{total} months downloaded  "
          f"({pending} remaining, ~{-(-pending // 10)} more days at 10 calls/day)")


# ─────────────────────────────────────────────────────────────────────────────
# Load and process
# ─────────────────────────────────────────────────────────────────────────────

def load_solcast(cfg: dict, tmy: bool = False) -> pd.DataFrame:
    """
    Load all cached Solcast CSV files and combine into a single DataFrame.

    Parameters
    ----------
    tmy : bool
        If True, load the TMY file instead of monthly historical files.
    """
    out_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "solcast"

    if tmy:
        tmy_path = out_dir / "solcast_tmy.csv"
        if not tmy_path.exists():
            raise FileNotFoundError(
                f"{tmy_path} not found.\n"
                "Run  python scripts/fetch_solcast.py --tmy"
            )
        logger.info("Loading Solcast TMY …")
        return _parse_solcast_csv(tmy_path, tmy=True)

    csv_files = sorted(out_dir.glob("solcast_20*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No Solcast monthly files in {out_dir}.\n"
            "Run  python scripts/fetch_solcast.py  (may take multiple days).\n"
            "Check progress with:  python scripts/fetch_solcast.py --status"
        )

    logger.info(f"Loading {len(csv_files)} Solcast monthly file(s) …")
    frames = [_parse_solcast_csv(p) for p in csv_files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info(f"  Rows: {len(df):,}  ({df.index.min()} → {df.index.max()})")
    return df


def load_solcast_local_files(cfg: dict) -> pd.DataFrame:
    """
    Load 5-minute Solcast CSV files placed in data/external/.

    Expects files named  solcast_weather_data_YYYY.csv  or
    solcast_weather_data_YYYY_*.csv  (e.g. solcast_weather_data_2023_end.csv).
    Each file must contain a 'period_end' timestamp column (UTC) and a
    'period' column with value PT5M.

    Returns a single DataFrame at 5-minute resolution with period-start index.
    """
    ext_dir = resolve_path(cfg["paths"]["external_nasa"])
    files   = sorted(ext_dir.glob("solcast_weather_data_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Solcast local files found in {ext_dir}.\n"
            "Expected files named:  solcast_weather_data_YYYY.csv"
        )

    logger.info(f"Loading {len(files)} local Solcast file(s) from {ext_dir} …")
    frames = [_parse_solcast_local_csv(p) for p in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info(
        f"  Total rows : {len(df):,}  "
        f"({df.index.min().date()} → {df.index.max().date()})"
    )
    return df


def _parse_solcast_local_csv(path: Path) -> pd.DataFrame:
    """
    Parse one locally-stored Solcast 5-minute CSV file.

    Column format (from Solcast batch download):
        period_end   — UTC timestamp (end of 5-min period)
        period       — "PT5M"
        ghi, dhi, dni, clearsky_ghi, clearsky_dhi, clearsky_dni
        air_temp, relative_humidity, surface_pressure, cloud_opacity, ...

    Index is shifted to period_start = period_end − 5 min.
    """
    df = pd.read_csv(path, parse_dates=["period_end"])
    logger.debug(f"  Parsing {path.name}  ({len(df):,} rows)")

    # Normalise to UTC
    if df["period_end"].dt.tz is None:
        df["period_end"] = df["period_end"].dt.tz_localize("UTC")
    else:
        df["period_end"] = df["period_end"].dt.tz_convert("UTC")

    # Shift period_end → period_start (subtract 5 min)
    df.index      = df["period_end"] - pd.Timedelta(minutes=5)
    df.index.name = "timestamp_utc"
    drop_cols     = [c for c in ["period_end", "period"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Clip negative irradiance values
    for col in ["ghi", "dhi", "dni", "clearsky_ghi", "clearsky_dhi", "clearsky_dni"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def _parse_solcast_csv(path: Path, tmy: bool = False) -> pd.DataFrame:
    """
    Parse a Solcast CSV file (historical monthly or TMY) from API download.

    Solcast CSVs use ISO 8601 timestamps in the 'period_end' column (UTC).
    For hourly data, period_end = start + 1 hour; we use period_end - 1h
    to get the hour-start timestamp matching our pipeline convention.
    """
    df = pd.read_csv(path, parse_dates=["period_end"])

    # Normalise to UTC
    if df["period_end"].dt.tz is None:
        df["period_end"] = df["period_end"].dt.tz_localize("UTC")
    else:
        df["period_end"] = df["period_end"].dt.tz_convert("UTC")

    # Shift period_end → hour-start  (period_end - 1h for hourly)
    df.index = df["period_end"] - pd.Timedelta(hours=1)
    df.index.name = "timestamp_utc"
    df = df.drop(columns=[c for c in ["period_end","period_mid","period"] if c in df.columns])

    # For TMY: assign a concrete year (2020) so the index is a real DatetimeIndex
    if tmy and df.index.year.nunique() == 1 and df.index.year[0] in [1900, 2001]:
        year_offset = pd.DateOffset(years=2020 - df.index.year[0])
        df.index = df.index + year_offset

    # Clip negatives
    for col in ["ghi", "dhi", "dni"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def solcast_to_nasa_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Solcast columns to the NASA POWER *_cal naming convention.

    Handles both API-downloaded files (air_temp_10m, wind_speed_10m) and
    locally-placed batch files (air_temp, no wind_speed).

    When wind speed is absent a default of 1.5 m/s is used — the Faiman
    cell-temperature model is only weakly sensitive to wind at typical
    Sri Lanka conditions (<1°C error at GHI = 600 W/m²).

    When clearsky_ghi is present in the source file it is used directly;
    otherwise pvlib Ineichen is computed.
    """
    # Column name variants across Solcast API vs batch download
    rename = {
        "ghi":            "ALLSKY_SFC_SW_DWN_cal",
        "dni":            "ALLSKY_SFC_SW_DNI_cal",
        "dhi":            "ALLSKY_SFC_SW_DIFF_cal",
        "clearsky_ghi":   "CLRSKY_SFC_SW_DWN_cal",
        # API names
        "air_temp_10m":   "T2M_cal",
        "wind_speed_10m": "WS10M_cal",
        # Local batch names
        "air_temp":       "T2M_cal",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Default wind speed when absent (no WS column in local batch files)
    if "WS10M_cal" not in out.columns:
        out["WS10M_cal"] = 1.5   # m/s  (typical Sri Lanka coastal average)

    # Compute clear-sky GHI with pvlib when not provided by the source
    if "CLRSKY_SFC_SW_DWN_cal" not in out.columns:
        _loc     = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)
        clearsky = _loc.get_clearsky(out.index, model="ineichen")
        out["CLRSKY_SFC_SW_DWN_cal"] = clearsky["ghi"].clip(lower=0).values

    return out


def save_solcast_processed(df: pd.DataFrame, cfg: dict, tmy: bool = False) -> Path:
    """Save processed Solcast DataFrame to data/processed/."""
    out_dir  = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname    = "solcast_tmy_processed.csv" if tmy else "solcast_processed.csv"
    out_path = out_dir / fname
    df.to_csv(out_path)
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def load_solcast_processed(cfg: dict, tmy: bool = False) -> pd.DataFrame:
    """Load a previously saved solcast_processed.csv."""
    fname = "solcast_tmy_processed.csv" if tmy else "solcast_processed.csv"
    path  = resolve_path(cfg["paths"]["processed"]) / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python scripts/fetch_solcast.py  to download and process Solcast data."
        )
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"Loaded Solcast: {len(df):,} rows  ({df.index.min()} → {df.index.max()})")
    return df
