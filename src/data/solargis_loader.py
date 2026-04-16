"""
src/data/solargis_loader.py
----------------------------
Loader for SolarGIS high-resolution irradiance data.

Why SolarGIS?
--------------
SolarGIS uses a proprietary satellite retrieval algorithm (HelioSat-4) with
~90 m spatial resolution and up to 15-minute temporal resolution.  It provides
the highest-accuracy irradiance dataset commercially available, used by major
solar project developers and lenders.  For Sri Lanka, SolarGIS covers the full
record from 1994 to near-real-time.

Access — commercial (research exceptions available)
----------------------------------------------------
SolarGIS data requires a purchase or academic research agreement.

Options:
  1. SolarGIS API (REST)  — https://solargis.com/docs/api
     - Request a trial or research account at https://solargis.com/contact/
     - Set environment variable:  export SOLARGIS_API_KEY=<your-key>

  2. SolarGIS Prospect    — https://solargis.com/maps-and-gis-data/overview/
     - Web interface for manual CSV download
     - Select: Site → Time series → 2020-01-01 to 2025-12-31, Hourly, UTC

  3. Academic request     — https://solargis.com/company/about-us/contact/
     - Email requesting research data access; many universities have agreements

Once you have a CSV file from SolarGIS, place it at:
    data/external/solargis/solargis_<year>.csv   (one file per year)
OR  data/external/solargis/solargis_full.csv     (single combined file)

Then run:
    python scripts/fetch_solargis.py --local  # skip download, process saved file

SolarGIS CSV format
--------------------
SolarGIS exports vary by product but typically include:
  - Metadata rows (site name, lat, lon, elevation, timezone)
  - Header row with column names
  - Data rows: Date, Time, GHI, DNI, DHI, TEMP, WS, ...

Column mapping
--------------
  SolarGIS column   NASA POWER equiv         Unit
  GHI               ALLSKY_SFC_SW_DWN        W/m²
  DIF/DIFH/DHI      ALLSKY_SFC_SW_DIFF       W/m²
  BNI/DNI           ALLSKY_SFC_SW_DNI        W/m²
  PVOUT (optional)  (direct AC power)        kWh/kWp
  TEMP              T2M                      °C
  WS                WS10M                    m/s
  Clear-sky GHI     CLRSKY_SFC_SW_DWN        W/m²
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LAT = 6.7912
_LON = 79.9005


# ─────────────────────────────────────────────────────────────────────────────
# No automated download — SolarGIS is commercial
# ─────────────────────────────────────────────────────────────────────────────

def check_solargis_files(cfg: dict) -> list[Path]:
    """
    Check for SolarGIS CSV files in data/external/solargis/.
    Raises FileNotFoundError with setup instructions if none found.
    """
    sg_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "solargis"

    candidates = list(sg_dir.glob("solargis*.csv")) if sg_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No SolarGIS files found in {sg_dir}.\n\n"
            "SolarGIS data requires a commercial subscription or research agreement.\n\n"
            "How to obtain data:\n"
            "  1. Request research access:  https://solargis.com/contact/\n"
            "  2. Or download manually from SolarGIS Prospect (web UI):\n"
            "       https://solargis.com/maps-and-gis-data/overview/\n"
            "       Site:  Lat=6.7912  Lon=79.9005  (University of Moratuwa)\n"
            "       Parameters: GHI, DNI, DIF, TEMP, WS, Clear-sky GHI\n"
            "       Period: 2020-01-01 to 2025-12-31, Hourly, UTC\n\n"
            "  3. Place the downloaded file(s) in:\n"
            f"       {sg_dir}/\n\n"
            "  4. Re-run:  python scripts/fetch_solargis.py --local"
        )
    return sorted(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Load and process
# ─────────────────────────────────────────────────────────────────────────────

def load_solargis(cfg: dict) -> pd.DataFrame:
    """
    Load and parse SolarGIS CSV file(s) from data/external/solargis/.

    Handles the two most common SolarGIS export formats:
      A. Prospect export: metadata block at top, then Date/Time + variables
      B. API export: clean CSV with ISO datetime column
    """
    files = check_solargis_files(cfg)
    logger.info(f"Loading {len(files)} SolarGIS file(s) …")
    frames = [_parse_solargis_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info(f"  Rows: {len(df):,}  ({df.index.min()} → {df.index.max()})")
    return df


def _parse_solargis_csv(path: Path) -> pd.DataFrame:
    """
    Parse a SolarGIS CSV export, handling the metadata header block.

    SolarGIS Prospect exports contain a metadata block (lines starting with
    '#' or containing '::') before the actual data.  We skip those rows and
    parse from the first row that looks like a column header.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    # Find the first line that looks like a data header (contains Date or year digits)
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Header line typically starts with "Date" or a 4-digit year
        if re.match(r"^(Date|20\d{2})", stripped, re.IGNORECASE):
            data_start = i
            break

    logger.info(f"  {path.name}: data starts at row {data_start}")
    df = pd.read_csv(path, skiprows=data_start, sep=None, engine="python")

    # Build timestamp index
    dt_col = _detect_datetime_column(df)
    if dt_col:
        df.index = pd.to_datetime(df[dt_col], utc=True)
        df = df.drop(columns=[dt_col])
    elif "Date" in df.columns and "Time" in df.columns:
        df.index = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), utc=True
        )
        df = df.drop(columns=["Date", "Time"])
    else:
        raise ValueError(
            f"Cannot detect timestamp column in {path.name}.\n"
            f"Columns found: {list(df.columns)}"
        )
    df.index.name = "timestamp_utc"

    # Standardise column names (SolarGIS uses different names across versions)
    col_map = _build_solargis_col_map(df.columns)
    df = df.rename(columns=col_map)

    # Replace fill values (-999, -9999) with NaN
    df = df.replace([-999, -9999, "-999", "-9999"], np.nan)
    for col in ["ghi_Wm2", "dhi_Wm2", "dni_Wm2", "clrsky_ghi_Wm2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    keep = ["ghi_Wm2","dhi_Wm2","dni_Wm2","clrsky_ghi_Wm2","t2m_C","ws10m_ms"]
    return df[[c for c in keep if c in df.columns]]


def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Return the name of the ISO datetime column if present."""
    for col in df.columns:
        if re.search(r"datetime|timestamp|time_utc", col, re.IGNORECASE):
            return col
    return None


def _build_solargis_col_map(columns) -> dict:
    """
    Map SolarGIS column names (which vary by product version) to our schema.
    """
    mapping = {}
    for col in columns:
        c = col.strip().upper()
        if c in ("GHI", "GHICS"):
            mapping[col] = "ghi_Wm2"
        elif c in ("DIF", "DIFH", "DHI"):
            mapping[col] = "dhi_Wm2"
        elif c in ("BNI", "DNI"):
            mapping[col] = "dni_Wm2"
        elif c in ("GHICS_CLEAR", "CLRSKY_GHI", "GHI_CLEAR"):
            mapping[col] = "clrsky_ghi_Wm2"
        elif c in ("TEMP", "T2M", "TAIR", "AIR_TEMPERATURE"):
            mapping[col] = "t2m_C"
        elif c in ("WS", "WS10", "WINDSPEED", "WIND_SPEED"):
            mapping[col] = "ws10m_ms"
    return mapping


def solargis_to_nasa_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename SolarGIS columns to the NASA POWER *_cal naming convention.
    Computes clear-sky GHI via pvlib if not provided by SolarGIS.
    """
    rename = {
        "ghi_Wm2":       "ALLSKY_SFC_SW_DWN_cal",
        "dni_Wm2":       "ALLSKY_SFC_SW_DNI_cal",
        "dhi_Wm2":       "ALLSKY_SFC_SW_DIFF_cal",
        "clrsky_ghi_Wm2":"CLRSKY_SFC_SW_DWN_cal",
        "t2m_C":         "T2M_cal",
        "ws10m_ms":      "WS10M_cal",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "CLRSKY_SFC_SW_DWN_cal" not in out.columns:
        logger.info("Clear-sky GHI not in SolarGIS file — computing with pvlib ineichen.")
        _loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)
        clearsky = _loc.get_clearsky(out.index, model="ineichen")
        out["CLRSKY_SFC_SW_DWN_cal"] = clearsky["ghi"].clip(lower=0).values

    return out


def save_solargis_processed(df: pd.DataFrame, cfg: dict) -> Path:
    """Save processed SolarGIS DataFrame to data/processed/solargis_processed.csv."""
    out_dir  = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solargis_processed.csv"
    df.to_csv(out_path)
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def load_solargis_processed(cfg: dict) -> pd.DataFrame:
    """Load a previously saved solargis_processed.csv."""
    path = resolve_path(cfg["paths"]["processed"]) / "solargis_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python scripts/fetch_solargis.py --local  after placing SolarGIS "
            "CSV files in  data/external/solargis/"
        )
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"Loaded SolarGIS: {len(df):,} rows  ({df.index.min()} → {df.index.max()})")
    return df
