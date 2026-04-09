"""
src/data/nasa_power.py
-----------------------
Fetches hourly meteorological and irradiance data from the NASA POWER API
for the University of Moratuwa site (6.7912° N, 79.9005° E).

NASA POWER provides satellite-derived reanalysis data at ~0.5° spatial
resolution going back to 1981. We use it to build a 2020–2026 pseudo-
historical dataset for PV generation modelling.

API docs: https://power.larc.nasa.gov/docs/services/api/

Variables fetched
-----------------
  ALLSKY_SFC_SW_DWN   Global Horizontal Irradiance (GHI)   W/m²
  ALLSKY_SFC_SW_DNI   Direct Normal Irradiance (DNI)        W/m²
  ALLSKY_SFC_SW_DIFF  Diffuse Horizontal Irradiance (DHI)   W/m²
  CLRSKY_SFC_SW_DWN   Clear-sky GHI                         W/m²
  T2M                 Air temperature at 2 m                °C
  RH2M                Relative humidity                     %
  WS10M               Wind speed at 10 m                    m/s
  WD10M               Wind direction at 10 m                °

Usage
-----
    from src.utils.config import load_config
    from src.data.nasa_power import fetch_nasa_power, save_raw

    cfg = load_config()
    df  = fetch_nasa_power(cfg)
    path = save_raw(df, cfg)
"""

import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# The NASA POWER hourly endpoint caps each request at roughly 2 years.
# We split the full range into chunks and concatenate.
_MAX_YEARS_PER_CHUNK = 2


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_date_range(start: str, end: str, max_years: int = _MAX_YEARS_PER_CHUNK):
    """
    Split a YYYYMMDD date range into chunks of at most `max_years` years.

    Yields
    ------
    tuple[str, str]  — (chunk_start, chunk_end) in YYYYMMDD format
    """
    cur = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    while cur <= end_dt:
        # Advance by max_years years (approximate with 365 * max_years days)
        chunk_end = datetime(cur.year + max_years, cur.month, cur.day) - timedelta(days=1)
        chunk_end = min(chunk_end, end_dt)
        yield cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        cur = chunk_end + timedelta(days=1)


def _request_chunk(lat: float, lon: float, start: str, end: str,
                   parameters: list, community: str,
                   retry: int, delay: float) -> pd.DataFrame:
    """
    Make a single API request for one date chunk with retry logic.

    Returns
    -------
    pd.DataFrame  — hourly rows for this chunk
    """
    params = {
        "parameters": ",".join(parameters),
        "community":  community,
        "longitude":  lon,
        "latitude":   lat,
        "start":      start,
        "end":        end,
        "format":     "JSON",
        "time-standard": "UTC",
    }

    for attempt in range(1, retry + 1):
        try:
            logger.info(f"    Requesting {start} → {end}  (attempt {attempt}/{retry})")
            r = requests.get(_BASE_URL, params=params, timeout=180)
            r.raise_for_status()
            return _parse_json(r.json())

        except requests.exceptions.RequestException as exc:
            logger.warning(f"    Request failed: {exc}")
            if attempt < retry:
                logger.info(f"    Retrying in {delay}s…")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"All {retry} attempts failed for chunk {start}–{end}"
                ) from exc


def _parse_json(data: dict) -> pd.DataFrame:
    """
    Convert the NASA POWER JSON response into a tidy DataFrame.

    The API nests data as:
        data["properties"]["parameter"][PARAM_NAME][YYYYMMDDHH] = value

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with UTC DatetimeIndex.
    """
    try:
        props = data["properties"]["parameter"]
    except KeyError as exc:
        raise KeyError(f"Unexpected NASA POWER JSON structure: {exc}") from exc

    series = {}
    for param, hourly in props.items():
        idx = pd.to_datetime(list(hourly.keys()), format="%Y%m%d%H", utc=True)
        series[param] = pd.Series(list(hourly.values()), index=idx, dtype=float)

    df = pd.DataFrame(series)
    df.index.name = "timestamp_utc"
    df.sort_index(inplace=True)

    # NASA POWER uses -999 as the missing-data sentinel
    df.replace(-999.0, float("nan"), inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nasa_power(
    cfg: dict,
    start_date: str | None = None,
    end_date:   str | None = None,
    retry_attempts: int = 3,
    retry_delay_s:  float = 15.0,
) -> pd.DataFrame:
    """
    Fetch hourly NASA POWER data for the site defined in configs/site.yaml.

    The function automatically splits 2020–2026 into manageable chunks
    (API limit ~2 years per request), fetches each chunk, and returns
    one combined DataFrame.

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().
    start_date : str, optional
        Override start in YYYYMMDD format. Default: from config.
    end_date : str, optional
        Override end in YYYYMMDD format. Default: from config.
    retry_attempts : int
        Retry count per chunk on failure.
    retry_delay_s : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Hourly rows with UTC DatetimeIndex and one column per parameter.
        Missing values are NaN (NASA sentinel -999 already replaced).
    """
    nasa_cfg = cfg["nasa_power"]
    site_cfg = cfg["site"]

    lat  = site_cfg["latitude"]
    lon  = site_cfg["longitude"]
    pars = nasa_cfg["parameters"]
    comm = nasa_cfg.get("community", "RE")
    start = start_date or nasa_cfg["start_date"]
    end   = end_date   or nasa_cfg["end_date"]

    logger.info("=" * 60)
    logger.info("NASA POWER DATA FETCH")
    logger.info(f"  Site      : {site_cfg['name']}")
    logger.info(f"  Location  : {lat}° N, {lon}° E")
    logger.info(f"  Date range: {start} → {end}")
    logger.info(f"  Parameters: {pars}")
    logger.info("=" * 60)

    chunks = list(_split_date_range(start, end))
    logger.info(f"Split into {len(chunks)} chunk(s) of ≤{_MAX_YEARS_PER_CHUNK} years each")

    frames = []
    for i, (c_start, c_end) in enumerate(chunks, 1):
        logger.info(f"  Chunk {i}/{len(chunks)}")
        df_chunk = _request_chunk(lat, lon, c_start, c_end, pars, comm,
                                  retry_attempts, retry_delay_s)
        frames.append(df_chunk)
        logger.info(f"    ✓ {len(df_chunk):,} rows")
        if i < len(chunks):
            time.sleep(3)   # polite pause between requests

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    logger.info("-" * 60)
    logger.info(f"Total rows : {len(df):,}")
    logger.info(f"Date range : {df.index.min()}  →  {df.index.max()}")
    logger.info("Missing values per column:")
    missing = df.isna().sum()
    for col, n in missing.items():
        pct = 100 * n / len(df)
        logger.info(f"    {col:<25} {n:>6} ({pct:.2f}%)")
    logger.info("=" * 60)

    return df


def save_raw(df: pd.DataFrame, cfg: dict, filename: str | None = None) -> Path:
    """
    Save the raw NASA POWER DataFrame to data/external/ as CSV.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : dict
    filename : str, optional
        Defaults to  nasa_power_<lat>_<lon>_<start>_<end>.csv

    Returns
    -------
    Path  — absolute path of the saved file
    """
    site  = cfg["site"]
    nasa  = cfg["nasa_power"]

    if filename is None:
        filename = (
            f"nasa_power_{site['latitude']}_{site['longitude']}"
            f"_{nasa['start_date']}_{nasa['end_date']}.csv"
        )

    out_dir = resolve_path(cfg["paths"]["external_nasa"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    df.to_csv(out_path)
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def load_raw(cfg: dict, filename: str | None = None) -> pd.DataFrame:
    """
    Load a previously saved raw NASA POWER CSV.

    If filename is None, loads the most recently modified CSV in data/external/.

    Returns
    -------
    pd.DataFrame  with UTC DatetimeIndex
    """
    ext_dir = resolve_path(cfg["paths"]["external_nasa"])

    if filename:
        path = ext_dir / filename
    else:
        candidates = sorted(ext_dir.glob("nasa_power_*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No NASA POWER CSV found in {ext_dir}")
        path = candidates[-1]
        logger.info(f"Auto-loading: {path.name}")

    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df
