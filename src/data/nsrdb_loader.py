"""
src/data/nsrdb_loader.py
------------------------
Downloads and preprocesses NSRDB PSM v3 (Physical Solar Model) data from
the NREL National Solar Radiation Database for the University of Moratuwa site.

Why NSRDB over NASA POWER?
---------------------------
NSRDB PSM v3 uses GOES/Meteosat satellite imagery at ~4 km spatial resolution
(vs NASA POWER's 0.5° ≈ 55 km), dramatically improving cloud-cover accuracy
for convective tropical climates like Sri Lanka.  The PSM v3 physical model
also includes aerosol optical depth (MERRA-2 driven), reducing clear-sky bias.

Variables downloaded
---------------------
  GHI        Global Horizontal Irradiance       W/m²
  DHI        Diffuse Horizontal Irradiance       W/m²
  DNI        Direct Normal Irradiance            W/m²
  Clearsky GHI  Clear-sky GHI (REST2 model)     W/m²
  Temperature   Air temperature at 2 m          °C
  Wind Speed    Wind speed at 10 m              m/s
  Relative Humidity  at 2 m                     %
  Dew Point     Dew-point temperature           °C

Time convention
---------------
  NSRDB hourly data is labelled at the centre of each hour (HH:30 UTC when
  interval=60 and utc=true).  We shift labels to hour-start (HH:00) so the
  index aligns with NASA POWER and ERA5 pipelines.

API setup (one-time)
---------------------
1. Register for a free API key at https://developer.nrel.gov/signup/
2. Set environment variable:  export NREL_API_KEY=<your-key>
   OR pass --api-key flag to scripts/fetch_nsrdb.py

Column mapping to NASA POWER equivalents
-----------------------------------------
  NSRDB column      NASA POWER equiv        Unit
  GHI               ALLSKY_SFC_SW_DWN       W/m²
  DNI               ALLSKY_SFC_SW_DNI       W/m²
  DHI               ALLSKY_SFC_SW_DIFF      W/m²
  Clearsky GHI      CLRSKY_SFC_SW_DWN       W/m²
  Temperature       T2M                     °C
  Relative Humidity RH2M                    %
  Wind Speed        WS10M                   m/s

Usage
-----
    from src.data.nsrdb_loader import download_nsrdb, load_nsrdb_processed

    download_nsrdb(cfg, api_key="your_key", start_year=2020, end_year=2025)
    df = load_nsrdb_processed(cfg)   # drop-in replacement for NASA POWER df
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

# PSM v3 covers the Americas only.
# For South/South-East Asia (incl. Sri Lanka) the correct dataset is
# MSG-IODC (Meteosat Second Generation — Indian Ocean Data Coverage).
# Direct CSV streaming endpoint (single location, single year at a time).
# Available years: 2017, 2018, 2019 only.
_NSRDB_URL       = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
_NSRDB_JSON_URL  = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.json"
_AVAILABLE_YEARS = (2017, 2018, 2019)
_ATTRIBUTES = (
    "ghi,dhi,dni,clearsky_ghi,"
    "air_temperature,relative_humidity,dew_point,wind_speed,wind_direction"
)


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_nsrdb(
    cfg:        dict,
    api_key:    str | None = None,
    start_year: int = 2017,
    end_year:   int = 2019,
    full_name:  str = "Researcher",
    email:      str = "researcher@university.edu",
    affiliation:str = "University of Moratuwa",
    reason:     str = "Solar PV forecasting research",
) -> Path:
    """
    Download NSRDB MSG-IODC hourly data from NREL for the site location.

    For Sri Lanka, the applicable NSRDB dataset is MSG-IODC (Meteosat
    Indian Ocean Data Coverage), which covers 2017–2019 at 4 km resolution.
    Downloads one CSV per year and saves to data/external/nsrdb/.
    Skips years where the file already exists.

    Parameters
    ----------
    cfg      : site config dict
    api_key  : NREL developer API key (falls back to NREL_API_KEY env var)
    start_year, end_year : inclusive year range (valid: 2017–2019)
    full_name, email, affiliation, reason : required by NREL API

    Returns
    -------
    Path — directory containing downloaded CSV files.
    """
    key = api_key or os.environ.get("NREL_API_KEY", "")
    if not key:
        raise ValueError(
            "NREL API key required.\n"
            "1. Register at https://developer.nrel.gov/signup/\n"
            "2. Set:  export NREL_API_KEY=<your-key>\n"
            "   OR pass  --api-key <key>  to scripts/fetch_nsrdb.py"
        )

    # Clamp to available years
    years = [y for y in range(start_year, end_year + 1) if y in _AVAILABLE_YEARS]
    if not years:
        raise ValueError(
            f"MSG-IODC only has data for years {_AVAILABLE_YEARS}.\n"
            f"Requested {start_year}–{end_year} has no overlap."
        )

    out_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "nsrdb"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        out_path = out_dir / f"nsrdb_msg_iodc_{year}_{_LAT}_{_LON}.csv"
        if out_path.exists():
            logger.info(f"  {year}: already exists — skipping  ({out_path.name})")
            continue

        logger.info(f"  Requesting NSRDB MSG-IODC  {year}  from NREL …")
        params = {
            "wkt":          f"POINT({_LON} {_LAT})",
            "names":        str(year),
            "interval":     "60",
            "utc":          "true",
            "leap_day":     "true",
            "full_name":    full_name,
            "email":        email,
            "affiliation":  affiliation,
            "mailing_list": "false",
            "reason":       reason,
            "api_key":      key,
            "attributes":   _ATTRIBUTES,
        }
        resp = requests.get(_NSRDB_URL, params=params, allow_redirects=True, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"NSRDB API error {resp.status_code} for year {year}:\n{resp.text[:500]}"
            )

        out_path.write_bytes(resp.content)
        size_kb = out_path.stat().st_size / 1024
        logger.info(f"  Saved → {out_path.name}  ({size_kb:.0f} KB)")
        time.sleep(1)

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Load and process
# ─────────────────────────────────────────────────────────────────────────────

def load_nsrdb(cfg: dict) -> pd.DataFrame:
    """
    Load all NSRDB CSV files from data/external/nsrdb/, parse, and combine.

    NSRDB CSV format:
      Row 0 : site metadata (Source, Latitude, Longitude, …)
      Row 1 : column names (Year, Month, Day, Hour, Minute, GHI, …)
      Row 2+: data (hourly, UTC, centre-of-hour at :30)

    Returns UTC-indexed DataFrame with columns:
        ghi_Wm2, dhi_Wm2, dni_Wm2, clrsky_ghi_Wm2,
        t2m_C, rh_pct, ws10m_ms, wd10m_deg
    """
    nsrdb_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "nsrdb"
    csv_files = sorted(nsrdb_dir.glob("nsrdb_msg_iodc_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No NSRDB CSV files in {nsrdb_dir}.\n"
            "Run  python scripts/fetch_nsrdb.py  to download them."
        )

    logger.info(f"Loading {len(csv_files)} NSRDB file(s) …")
    frames = []
    for path in csv_files:
        frames.append(_parse_nsrdb_csv(path))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info(f"  Rows: {len(df):,}  ({df.index.min()} → {df.index.max()})")
    return df


def _parse_nsrdb_csv(path: Path) -> pd.DataFrame:
    """
    Parse a single NSRDB PSM v3 CSV file.

    The file has:
      Line 0: site metadata key-value pairs
      Line 1: column names (Year, Month, Day, Hour, Minute, GHI, …)
      Line 2+: data rows

    Time is centre-of-hour (Minute=30 for hourly).  We normalise to
    hour-start (Minute=0) for alignment with the NASA/ERA5 pipeline.
    """
    # Read metadata line to extract site info
    meta_line = path.read_text().splitlines()[0]
    logger.info(f"  {path.name}: {meta_line[:80]} …")

    # Data starts at row 2 (row 0 = metadata, row 1 = column headers)
    df = pd.read_csv(path, skiprows=2)

    # Build UTC timestamp from Year/Month/Day/Hour/Minute
    # NSRDB labels at :30 → shift to :00 (hour-start convention)
    df["timestamp_utc"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].astype(str).agg(
            lambda r: f"{r['Year']}-{r['Month']:>02}-{r['Day']:>02} {r['Hour']:>02}:00",
            axis=1,
        ),
        utc=True,
    )
    df = df.set_index("timestamp_utc")

    # Drop raw time columns
    df = df.drop(columns=[c for c in ["Year","Month","Day","Hour","Minute"] if c in df.columns])

    # Rename to internal schema
    col_map = {
        "GHI":                "ghi_Wm2",
        "DHI":                "dhi_Wm2",
        "DNI":                "dni_Wm2",
        "Clearsky GHI":       "clrsky_ghi_Wm2",
        "Clearsky DHI":       "clrsky_dhi_Wm2",
        "Clearsky DNI":       "clrsky_dni_Wm2",
        "Temperature":        "t2m_C",
        "Relative Humidity":  "rh_pct",
        "Dew Point":          "dew_point_C",
        "Wind Speed":         "ws10m_ms",
        "Wind Direction":     "wd10m_deg",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Replace -999 (NSRDB fill value) with NaN, clip negatives
    df = df.replace(-999, np.nan)
    for col in ["ghi_Wm2", "dhi_Wm2", "dni_Wm2", "clrsky_ghi_Wm2"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    keep = ["ghi_Wm2","dhi_Wm2","dni_Wm2","clrsky_ghi_Wm2",
            "t2m_C","rh_pct","dew_point_C","ws10m_ms","wd10m_deg"]
    return df[[c for c in keep if c in df.columns]]


def nsrdb_to_nasa_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename NSRDB columns to the NASA POWER *_cal names used downstream.

    NSRDB provides clear-sky GHI natively (REST2 model), so no pvlib
    fallback is needed for CLRSKY_SFC_SW_DWN_cal.
    """
    rename = {
        "ghi_Wm2":       "ALLSKY_SFC_SW_DWN_cal",
        "dni_Wm2":       "ALLSKY_SFC_SW_DNI_cal",
        "dhi_Wm2":       "ALLSKY_SFC_SW_DIFF_cal",
        "clrsky_ghi_Wm2":"CLRSKY_SFC_SW_DWN_cal",
        "t2m_C":         "T2M_cal",
        "rh_pct":        "RH2M_cal",
        "ws10m_ms":      "WS10M_cal",
        "wd10m_deg":     "WD10M",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # If clear-sky column still missing (older files), compute with pvlib
    if "CLRSKY_SFC_SW_DWN_cal" not in out.columns:
        logger.warning("Clearsky GHI not found in NSRDB data — computing with pvlib ineichen.")
        _loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)
        clearsky = _loc.get_clearsky(out.index, model="ineichen")
        out["CLRSKY_SFC_SW_DWN_cal"] = clearsky["ghi"].clip(lower=0).values

    return out


def save_nsrdb_processed(df: pd.DataFrame, cfg: dict) -> Path:
    """Save processed NSRDB DataFrame to data/processed/nsrdb_processed.csv."""
    out_dir  = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nsrdb_processed.csv"
    df.to_csv(out_path)
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def load_nsrdb_processed(cfg: dict) -> pd.DataFrame:
    """Load a previously saved nsrdb_processed.csv."""
    path = resolve_path(cfg["paths"]["processed"]) / "nsrdb_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python scripts/fetch_nsrdb.py  to download and process NSRDB data."
        )
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"Loaded NSRDB: {len(df):,} rows  ({df.index.min()} → {df.index.max()})")
    return df
