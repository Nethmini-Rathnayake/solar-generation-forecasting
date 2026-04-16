"""
src/data/era5_loader.py
------------------------
Downloads and preprocesses ERA5 hourly reanalysis data from the
Copernicus Climate Data Store (CDS) for the University of Moratuwa site.

Why ERA5 over NASA POWER?
--------------------------
NASA POWER delivers data at 0.5° (~55 km) spatial resolution. Tropical
convective cloud systems are 5–20 km wide — invisible to 55 km averaging.
ERA5 is at 0.25° (~28 km), reducing cloud mismatch error by ~25–35%.
ERA5 also has a richer atmospheric model (ERA5 uses 137 vertical levels vs
POWER's simplified MERRA-2 base), giving more accurate surface radiation.

Variables requested
--------------------
  ssrd    Surface solar radiation downwards         J/m²  (accumulated)
  fdir    Total sky direct solar radiation at surface J/m² (accumulated)
  t2m     2m air temperature                         K
  d2m     2m dewpoint temperature                    K   → used for RH
  u10     10m U-component of wind                    m/s
  v10     10m V-component of wind                    m/s

All ERA5 radiation variables are accumulated (J/m² per timestep).
We de-accumulate to hourly means → divide by 3600 → W/m².

CDS setup (one-time, done by user)
------------------------------------
1. Register at https://cds.climate.copernicus.eu
2. Accept the ERA5 dataset licence
3. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: <your-api-key>
   (the key is shown on your CDS profile page)

Column mapping to NASA POWER equivalents
-----------------------------------------
  ERA5 column      NASA POWER equiv      Unit
  ghi_Wm2          ALLSKY_SFC_SW_DWN     W/m²
  dni_Wm2          ALLSKY_SFC_SW_DNI     W/m²   (approximated)
  dhi_Wm2          ALLSKY_SFC_SW_DIFF    W/m²   (GHI − DNI·cos_z)
  t2m_C            T2M                   °C
  rh_pct           RH2M                  %
  ws10m_ms         WS10M                 m/s
  wd10m_deg        WD10M                 °

Usage
-----
    from src.data.era5_loader import download_era5, load_era5, era5_to_nasa_schema

    path = download_era5(cfg, start_year=2020, end_year=2025)
    df   = load_era5(path, cfg)
    df_nasa = era5_to_nasa_schema(df)   # drop-in replacement for NASA POWER df
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LAT = 6.7912
_LON = 79.9005

# CDS variables to request
_CDS_VARIABLES = [
    "surface_solar_radiation_downwards",   # ssrd  (GHI, accumulated J/m²)
    "total_sky_direct_solar_radiation_at_surface",  # fdir (DNI beam, acc. J/m²)
    "2m_temperature",                      # t2m   K
    "2m_dewpoint_temperature",             # d2m   K
    "10m_u_component_of_wind",             # u10   m/s
    "10m_v_component_of_wind",             # v10   m/s
]


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_era5(
    cfg:        dict,
    start_year: int = 2020,
    end_year:   int = 2025,
) -> Path:
    """
    Download ERA5 hourly data from CDS for the site location.

    Downloads one NetCDF file per year and saves to data/external/.
    Skips years where the file already exists.

    Requires ~/.cdsapirc to be configured (see module docstring).

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().
    start_year, end_year : int
        Inclusive year range to download.

    Returns
    -------
    Path  — directory containing downloaded NetCDF files.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "cdsapi not installed. Run: pip install cdsapi\n"
            "Then configure ~/.cdsapirc with your CDS API key."
        )

    out_dir = resolve_path(cfg["paths"]["external_nasa"]).parent / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Round to nearest 0.25° grid point
    lat_r = round(_LAT * 4) / 4
    lon_r = round(_LON * 4) / 4
    # CDS area: [N, W, S, E] — request a small box around site
    area = [lat_r + 0.25, lon_r - 0.25, lat_r - 0.25, lon_r + 0.25]

    client = cdsapi.Client(quiet=True)

    for year in range(start_year, end_year + 1):
        out_path = out_dir / f"era5_{year}_{_LAT}_{_LON}.nc"
        if out_path.exists():
            logger.info(f"  {year}: already exists — skipping  ({out_path.name})")
            continue

        logger.info(f"  Requesting ERA5 {year} from CDS …")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     _CDS_VARIABLES,
                "year":         str(year),
                "month":        [f"{m:02d}" for m in range(1, 13)],
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time":         [f"{h:02d}:00" for h in range(24)],
                "area":         area,
                "format":       "netcdf",
            },
            str(out_path),
        )
        logger.info(f"  Saved → {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Load and process
# ─────────────────────────────────────────────────────────────────────────────

def load_era5(
    era5_dir: Path,
    cfg:      dict,
) -> pd.DataFrame:
    """
    Load all ERA5 NetCDF files from era5_dir, process to hourly UTC DataFrame.

    De-accumulates radiation (J/m² → W/m²), derives DHI, RH, wind speed/dir.
    Returns UTC-indexed DataFrame ready for pvlib simulation.

    Parameters
    ----------
    era5_dir : Path
        Directory containing era5_*.nc files (output of download_era5).
    cfg : dict

    Returns
    -------
    pd.DataFrame  with UTC DatetimeIndex and columns:
        ghi_Wm2, dni_Wm2, dhi_Wm2, t2m_C, rh_pct, ws10m_ms, wd10m_deg
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray not installed. Run: pip install xarray netcdf4")

    nc_files = sorted(era5_dir.glob("era5_*.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No ERA5 NetCDF files found in {era5_dir}.\n"
            "Run  python scripts/fetch_era5.py  to download them."
        )

    logger.info(f"Loading {len(nc_files)} ERA5 NetCDF file(s) from {era5_dir} …")
    frames = []
    for nc in nc_files:
        ds = xr.open_dataset(nc)
        # Nearest grid point to site
        ds = ds.sel(
            latitude=_LAT,  longitude=_LON,
            method="nearest",
        )
        frames.append(_process_era5_ds(ds))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info(f"  Total rows: {len(df):,}  ({df.index.min()} → {df.index.max()})")
    return df


def _process_era5_ds(ds) -> pd.DataFrame:
    """
    Convert a single ERA5 xarray Dataset to a tidy hourly DataFrame.

    ERA5 radiation is accumulated since 01:00 UTC on the first day of the file
    (or since previous retrieval reset at 07:00 UTC for hourly ERA5). The safe
    de-accumulation is: value(t) = raw(t) - raw(t-1), then clip to ≥ 0.
    Divide by 3600 to convert J/m² → W/m² (hourly mean).
    """
    import xarray as xr

    # Convert to DataFrame (drop spatial coords — we already selected nearest)
    df = ds.to_dataframe().reset_index()

    # Keep only the time column and variables
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp_utc"

    # ── De-accumulate radiation (J/m² accumulated → W/m² hourly mean) ────────
    # ERA5 single-levels hourly: each value is cumulative from forecast start.
    # The difference between consecutive hours gives the hourly flux.
    for var in ["ssrd", "fdir"]:
        if var in df.columns:
            df[var] = df[var].diff(1).clip(lower=0) / 3600.0

    # Rename radiation → meaningful names
    rename = {}
    if "ssrd" in df.columns: rename["ssrd"] = "ghi_Wm2"
    if "fdir" in df.columns: rename["fdir"] = "dir_Wm2"  # beam on horizontal
    df = df.rename(columns=rename)

    # ── Derive DNI from beam-on-horizontal ───────────────────────────────────
    # fdir = DNI × cos(solar_zenith)  →  DNI = fdir / cos(zenith)
    loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)
    solar_pos = loc.get_solarposition(df.index)
    cos_z = np.cos(np.deg2rad(solar_pos["apparent_zenith"].clip(upper=89)))
    cos_z[solar_pos["elevation"] <= 0] = np.nan

    if "dir_Wm2" in df.columns:
        df["dni_Wm2"] = (df["dir_Wm2"] / cos_z).clip(lower=0, upper=1400)
        # DHI = GHI − DNI × cos(zenith) = GHI − fdir
        df["dhi_Wm2"] = (df["ghi_Wm2"] - df["dir_Wm2"]).clip(lower=0)
    else:
        df["dni_Wm2"] = np.nan
        df["dhi_Wm2"] = np.nan

    # ── Temperature (K → °C) ─────────────────────────────────────────────────
    if "t2m" in df.columns:
        df["t2m_C"] = df["t2m"] - 273.15

    # ── Relative humidity from dewpoint ──────────────────────────────────────
    # August-Roche-Magnus approximation
    if "t2m" in df.columns and "d2m" in df.columns:
        T   = df["t2m"] - 273.15
        Td  = df["d2m"] - 273.15
        df["rh_pct"] = (
            100 * np.exp((17.625 * Td) / (243.04 + Td))
                / np.exp((17.625 * T)  / (243.04 + T))
        ).clip(0, 100)

    # ── Wind speed and direction ──────────────────────────────────────────────
    if "u10" in df.columns and "v10" in df.columns:
        df["ws10m_ms"]  = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wd10m_deg"] = (
            np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360
        )

    # Keep only derived columns
    keep = ["ghi_Wm2","dni_Wm2","dhi_Wm2","t2m_C","rh_pct","ws10m_ms","wd10m_deg"]
    return df[[c for c in keep if c in df.columns]]


def era5_to_nasa_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename ERA5 columns to the NASA POWER *_cal column names used downstream.

    This makes ERA5 a drop-in replacement for the calibrated NASA dataset
    in pv_model.py (which reads ALLSKY_SFC_SW_DWN_cal, T2M_cal, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_era5().

    Returns
    -------
    pd.DataFrame  with columns matching the *_cal naming convention.
    """
    rename = {
        "ghi_Wm2":   "ALLSKY_SFC_SW_DWN_cal",
        "dni_Wm2":   "ALLSKY_SFC_SW_DNI_cal",
        "dhi_Wm2":   "ALLSKY_SFC_SW_DIFF_cal",
        "t2m_C":     "T2M_cal",
        "rh_pct":    "RH2M_cal",
        "ws10m_ms":  "WS10M_cal",
        "wd10m_deg": "WD10M",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # ERA5 does not include a clear-sky GHI product — compute it with pvlib.
    # Using the placeholder CLRSKY = ALLSKY would make kt = 1.0 always,
    # breaking the clearness_index feature in the ML pipeline.
    if "ALLSKY_SFC_SW_DWN_cal" in out.columns and "CLRSKY_SFC_SW_DWN_cal" not in out.columns:
        _loc = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)
        clearsky = _loc.get_clearsky(out.index, model="ineichen")
        out["CLRSKY_SFC_SW_DWN_cal"] = clearsky["ghi"].clip(lower=0).values
    return out


def save_era5_processed(df: pd.DataFrame, cfg: dict) -> Path:
    """Save the processed ERA5 DataFrame to data/processed/era5_processed.csv."""
    out_dir  = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "era5_processed.csv"
    df.to_csv(out_path)
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def load_era5_processed(cfg: dict) -> pd.DataFrame:
    """Load a previously saved era5_processed.csv."""
    path = resolve_path(cfg["paths"]["processed"]) / "era5_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python scripts/fetch_era5.py  to download and process ERA5 data."
        )
    df = pd.read_csv(path, index_col="timestamp_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    logger.info(f"Loaded ERA5: {len(df):,} rows  ({df.index.min()} → {df.index.max()})")
    return df
