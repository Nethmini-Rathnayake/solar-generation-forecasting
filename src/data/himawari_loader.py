"""
src/data/himawari_loader.py
──────────────────────────────────────────────────────────────────────────────
Himawari-8 AHI Band-3 (visible 0.64 µm) data loader.

Reads HSD (Himawari Standard Data) binary files from the NOAA public S3
bucket.  No boto3 or satpy required — uses only requests + numpy.

Coordinate math
───────────────
Himawari-8 uses a Geostationary (GEOS) projection centred on 140.7 °E.
The projection parameters (CFAC, LFAC, COFF, LOFF, sub_lon) are read
directly from Block 3 of each HSD file, so the code works for any band
or resolution without hard-coded values.

Segment selection
─────────────────
The full disk is divided into 10 segments (S01–S10, north→south).
For site latitude 6.79 °N the relevant segment is S05 (segment 5 of 10).
For upstream cloud tracking we also download S04 (more northerly clouds
carried south by the monsoon circulation).

Usage
─────
    from src.data.himawari_loader import (
        s3_url,
        download_segment,
        read_hsd_segment,
        geos_to_latlon,
        latlon_to_pixel,
        extract_roi,
    )
"""

from __future__ import annotations

import bz2
import io
import struct
import time
from pathlib import Path

import numpy as np
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Himawari-8 constants (JMA Himawari-8/9 AHI User's Guide values) ──────────
_SUB_LON_DEFAULT = 140.7       # degrees (sub-satellite longitude, Himawari-8)
_H_KM            = 42164.0    # km — distance Earth centre → satellite
_R_EQ_KM         = 6378.169   # km — equatorial radius (JMA/CGMS spec)
_R_POL_KM        = 6356.5838  # km — polar radius  (JMA/CGMS spec)
_E2              = 1 - (_R_POL_KM / _R_EQ_KM) ** 2   # eccentricity²
_GEOCEN_FACTOR   = (_R_POL_KM / _R_EQ_KM) ** 2       # geocentric latitude factor ≈ 0.993252

# AWS S3 public buckets (no credentials required)
# Himawari-8 operated until 2022-12-13; Himawari-9 took over from 2022-12-13
_S3_BASE_H8 = "https://noaa-himawari8.s3.amazonaws.com"
_S3_BASE_H9 = "https://noaa-himawari9.s3.amazonaws.com"
_H9_START   = "2022-12-13"   # date H-9 became primary (H-8 retired)

# Default data directory
_HIM_DIR = Path("data/himawari")

# Site location
_SITE_LAT =  6.7912
_SITE_LON = 79.9005


# ─────────────────────────────────────────────────────────────────────────────
# S3 URL builder
# ─────────────────────────────────────────────────────────────────────────────

def s3_url(dt: "pd.Timestamp", band: int = 3, segment: int = 5) -> str:
    """
    Build the public S3 URL for a Himawari-8 AHI full-disk segment file.

    Parameters
    ----------
    dt      : timestamp (UTC)
    band    : AHI band number 1–16.  Band 3 = visible 0.64 µm, 0.5 km.
    segment : segment number 1–10 (S01=north, S10=south).

    Returns
    -------
    str  Full HTTPS URL to the bz2-compressed HSD file.

    Example
    -------
    >>> s3_url(pd.Timestamp("2022-04-01 00:00"), band=3, segment=5)
    'https://noaa-himawari8.s3.amazonaws.com/AHI-L1b-FLDK/2022/04/01/0000/
     HS_H08_20220401_0000_B03_FLDK_R05_S0510.DAT.bz2'
    """
    # Band resolution codes: B01-B02=1km, B03=0.5km, B04-B06=1km, B07-B16=2km
    res_map = {3: "R05"}
    res = res_map.get(band, "R10" if band <= 6 else "R20")

    # Select satellite: H-9 from 2022-12-13 onward
    import pandas as _pd
    dt_naive = _pd.Timestamp(dt).tz_localize(None) if _pd.Timestamp(dt).tzinfo is None \
               else _pd.Timestamp(dt).tz_convert(None)
    if dt_naive.normalize() >= _pd.Timestamp(_H9_START):
        sat_id  = "H09"
        s3_base = _S3_BASE_H9
    else:
        sat_id  = "H08"
        s3_base = _S3_BASE_H8

    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M")
    fname = (
        f"HS_{sat_id}_{date_str}_{time_str}_B{band:02d}_FLDK"
        f"_{res}_S{segment:02d}10.DAT.bz2"
    )
    path = f"AHI-L1b-FLDK/{dt.year}/{dt.month:02d}/{dt.day:02d}/{time_str}/{fname}"
    return f"{s3_base}/{path}"


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_segment(
    dt:        "pd.Timestamp",
    out_dir:   Path = _HIM_DIR,
    band:      int  = 3,
    segment:   int  = 5,
    timeout:   int  = 300,
    max_retry: int  = 5,
) -> Path | None:
    """
    Download a Himawari-8 HSD segment file from AWS S3 if not already cached.

    Parameters
    ----------
    dt      : UTC timestamp (will be rounded down to nearest 10 min).
    out_dir : local directory to save files.
    band    : AHI band (default 3 = visible).
    segment : segment number (default 5 = covers ~0°–12°N).
    timeout : HTTP timeout in seconds (connect, read).
    max_retry : number of retry attempts on failure.

    Returns
    -------
    Path to the local file, or None if download failed.
    """
    # Round to 10-min boundary
    import pandas as pd
    dt = pd.Timestamp(dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    dt = dt.floor("10min")

    url  = s3_url(dt, band=band, segment=segment)
    fname = Path(url.split("/")[-1])
    local = out_dir / str(dt.year) / f"{dt.month:02d}" / fname

    _MIN_VALID_BYTES = 5 * 1024 * 1024   # 5 MB — real HSD files are 25–35 MB
    if local.exists() and local.stat().st_size >= _MIN_VALID_BYTES:
        return local
    if local.exists():
        local.unlink()   # remove stale partial from a prior failed run

    local.parent.mkdir(parents=True, exist_ok=True)

    # Use a session with a larger read buffer; separate connect vs read timeout
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)

    for attempt in range(1, max_retry + 1):
        tmp = local.with_suffix(".tmp")
        try:
            r = session.get(url, timeout=(15, timeout), stream=True)
            if r.status_code == 404:
                logger.warning(f"  [Himawari] 404 — file not found: {fname.name}")
                return None
            r.raise_for_status()

            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)

            tmp.rename(local)   # atomic move — only committed if complete
            logger.info(
                f"  [Himawari] Downloaded {fname.name}  "
                f"({downloaded/1e6:.1f} MB)"
            )
            return local

        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            logger.warning(
                f"  [Himawari] Attempt {attempt}/{max_retry} failed: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < max_retry:
                time.sleep(min(30, 5 * attempt))
            else:
                return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HSD binary reader (pure numpy — no satpy required)
# ─────────────────────────────────────────────────────────────────────────────

def _read_block_header(buf: bytes, offset: int) -> tuple[int, int]:
    """Read 3-byte block header: (block_number, block_length)."""
    block_num = struct.unpack_from(">B", buf, offset)[0]
    block_len = struct.unpack_from(">H", buf, offset + 1)[0]
    return block_num, block_len


def read_hsd_segment(filepath: Path) -> dict:
    """
    Read a Himawari-8 HSD segment file and return image data + metadata.

    The HSD format (Himawari Standard Data) is a binary format published
    by JMA.  This reader decodes:
      - Block 1  : observation timestamp
      - Block 3  : projection parameters (CFAC, LFAC, COFF, LOFF, sub_lon)
      - Block 5  : calibration coefficients (slope, offset → radiance)
      - Block 7  : segment information (segment number, first line)
      - Data blocks (one per scan line) → raw 10-bit counts

    Parameters
    ----------
    filepath : path to .DAT or .DAT.bz2 file.

    Returns
    -------
    dict with keys:
        "counts"     : np.ndarray[uint16]  shape (nlines, npixels)
        "timestamp"  : pd.Timestamp  (UTC observation time)
        "segment"    : int           segment sequence number (1–10)
        "first_line" : int           first scan line in this segment
        "proj"       : dict          CFAC, LFAC, COFF, LOFF, sub_lon
        "calib"      : dict          slope, offset for radiance conversion
    """
    import pandas as pd

    # Decompress if needed
    raw = filepath.read_bytes()
    if filepath.suffix == ".bz2":
        raw = bz2.decompress(raw)

    buf = raw
    offset = 0

    proj  = {}
    calib = {}
    seg   = {}
    ts    = None
    lines_data: list[np.ndarray] = []

    while offset < len(buf):
        if offset + 3 > len(buf):
            break

        try:
            block_num, block_len = _read_block_header(buf, offset)
        except struct.error:
            break

        if block_len == 0 or offset + block_len > len(buf):
            break

        block = buf[offset: offset + block_len]

        # ── Block 1: Basic information ────────────────────────────────────
        if block_num == 1:
            # Observation start time: offset 51, float64 (days since 1858-11-17)
            try:
                obs_jd = struct.unpack_from(">d", block, 51)[0]
                # JMA epoch: Modified Julian Date (MJD), epoch = 1858-11-17
                ts = pd.Timestamp("1858-11-17") + pd.Timedelta(days=obs_jd)
                ts = ts.tz_localize("UTC")
            except (struct.error, ValueError):
                pass

        # ── Block 3: Projection information ──────────────────────────────
        elif block_num == 3:
            try:
                sub_lon = struct.unpack_from(">d", block, 3)[0]
                CFAC    = struct.unpack_from(">I", block, 11)[0]
                LFAC    = struct.unpack_from(">I", block, 15)[0]
                COFF    = struct.unpack_from(">f", block, 19)[0]
                LOFF    = struct.unpack_from(">f", block, 23)[0]
                proj    = {"sub_lon": sub_lon, "CFAC": CFAC, "LFAC": LFAC,
                           "COFF": COFF, "LOFF": LOFF}
            except struct.error:
                pass

        # ── Block 5: Calibration ──────────────────────────────────────────
        elif block_num == 5:
            try:
                slope  = struct.unpack_from(">d", block, 3)[0]
                offset_ = struct.unpack_from(">d", block, 11)[0]
                calib  = {"slope": slope, "offset": offset_}
            except struct.error:
                pass

        # ── Block 7: Segment information ─────────────────────────────────
        elif block_num == 7:
            try:
                seg = {
                    "total_segments"   : struct.unpack_from(">B", block, 3)[0],
                    "segment_number"   : struct.unpack_from(">B", block, 4)[0],
                    "first_line_number": struct.unpack_from(">H", block, 5)[0],
                    "last_line_number" : struct.unpack_from(">H", block, 7)[0],
                }
            except struct.error:
                pass

        # ── Block 11: Data (scan lines) ───────────────────────────────────
        elif block_num == 11:
            try:
                n_pixels = struct.unpack_from(">H", block, 5)[0]
                # Raw pixel data starts at byte 7 (uint16 each)
                raw_counts = np.frombuffer(
                    block, dtype=">u2", count=n_pixels, offset=7
                ).astype(np.uint16)
                lines_data.append(raw_counts)
            except (struct.error, ValueError):
                pass

        offset += block_len

    if not lines_data:
        raise ValueError(f"No data blocks found in {filepath.name}")

    # Stack scan lines → 2D array
    counts = np.vstack(lines_data) if all(
        l.shape == lines_data[0].shape for l in lines_data
    ) else np.array(lines_data, dtype=object)

    return {
        "counts"     : counts,
        "timestamp"  : ts,
        "segment"    : seg.get("segment_number", -1),
        "first_line" : seg.get("first_line_number", -1),
        "proj"       : proj,
        "calib"      : calib,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GEOS projection math (no pyproj required)
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_pixel(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    proj: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to Himawari pixel (col, row).

    Implements the CGMS LRIT/HRIT Global Specification GEOS projection.

    Parameters
    ----------
    lat, lon : geographic coordinates (degrees).
    proj     : dict with keys CFAC, LFAC, COFF, LOFF, sub_lon.

    Returns
    -------
    col, row : pixel coordinates (fractional, 1-based indexing).
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    sub_lon = np.radians(proj["sub_lon"])
    lat_r   = np.radians(lat)
    lon_r   = np.radians(lon)

    # Geocentric latitude (using JMA factor, consistent with inverse)
    c_lat    = np.arctan(_GEOCEN_FACTOR * np.tan(lat_r))
    cos_clat = np.cos(c_lat)
    sin_clat = np.sin(c_lat)
    dlon     = lon_r - sub_lon
    cos_dlon = np.cos(dlon)

    # Earth radius at geocentric latitude
    r_l = _R_POL_KM / np.sqrt(1.0 - _E2 * cos_clat ** 2)

    # Satellite-to-point vector
    r1 = _H_KM - r_l * cos_clat * cos_dlon
    r2 = -r_l * cos_clat * np.sin(dlon)
    r3 = r_l * sin_clat
    rn = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)

    # Scan angles (degrees)
    x = np.degrees(np.arctan2(-r2, r1))
    y = np.degrees(np.arcsin(-r3 / rn))

    # Convert to pixel coordinates
    col = proj["COFF"] + x * 2 ** -16 * proj["CFAC"]
    row = proj["LOFF"] + y * 2 ** -16 * proj["LFAC"]

    return col, row


def geos_to_latlon(
    col: "float | np.ndarray",
    row: "float | np.ndarray",
    proj: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Himawari pixel (col, row) to geographic (lat, lon).

    Inverse of latlon_to_pixel().

    Returns
    -------
    lat, lon : degrees.  NaN where the pixel is outside Earth disk.
    """
    scalar = np.ndim(col) == 0
    col = np.atleast_1d(np.asarray(col, dtype=np.float64))
    row = np.atleast_1d(np.asarray(row, dtype=np.float64))

    sub_lon = np.radians(proj["sub_lon"])

    # Scan angles (radians)
    x = np.radians((col - proj["COFF"]) / proj["CFAC"] * 2 ** 16)
    y = np.radians((row - proj["LOFF"]) / proj["LFAC"] * 2 ** 16)

    cos_x = np.cos(x);  sin_x = np.sin(x)
    cos_y = np.cos(y);  sin_y = np.sin(y)

    a = sin_x ** 2 + cos_x ** 2 * (cos_y ** 2 + (_R_EQ_KM / _R_POL_KM) ** 2 * sin_y ** 2)
    b = -2.0 * _H_KM * cos_x * cos_y
    c = _H_KM ** 2 - _R_EQ_KM ** 2

    disc = b ** 2 - 4 * a * c
    valid = disc >= 0

    rs = np.full_like(col, np.nan)
    rs[valid] = (-b[valid] - np.sqrt(disc[valid])) / (2 * a[valid])

    # Reconstruct Earth-centered satellite-frame coordinates of surface point.
    # Forward defines:
    #   y = arcsin(-r3/rn)  where r3 = Z_sat = r_l × sin(c_lat) > 0 for north
    # ⟹ Z_sat = -rs × sin_y  (positive for north since y < 0 for north)
    #   x = arctan(-r2/r1)  where r1 = H - X_sat, r2 = -Y_sat × cos_y ... (approx)
    # ⟹ X_sat = H - rs × cos_x × cos_y
    # ⟹ Y_sat = rs × sin_x × cos_y
    X_sat = _H_KM - rs * cos_x * cos_y   # earth-centered x (in satellite frame)
    Y_sat = rs * sin_x * cos_y            # earth-centered y
    Z_sat = -rs * sin_y                   # earth-centered z (positive = north)

    # Geographic latitude: geocentric → geographic via (R_eq/R_pol)² factor
    lat = np.degrees(np.arctan2(
        (_R_EQ_KM / _R_POL_KM) ** 2 * Z_sat,
        np.sqrt(X_sat ** 2 + Y_sat ** 2)
    ))

    # Longitude: angle in the equatorial plane relative to sub-satellite point
    lon = np.degrees(np.arctan2(Y_sat, X_sat)) + proj["sub_lon"]
    # Normalise to (-180, 180)
    lon = (lon + 180.0) % 360.0 - 180.0

    lat[~valid] = np.nan
    lon[~valid] = np.nan

    if scalar:
        return float(lat[0]), float(lon[0])
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# ROI extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi(
    hsd: dict,
    lat_min: float = 3.0,
    lat_max: float = 11.0,
    lon_min: float = 76.0,
    lon_max: float = 84.0,
) -> dict:
    """
    Crop HSD counts to a geographic bounding box.

    Default box: 8° × 8° centred on site (6.79°N, 79.90°E), ~880 km ×
    890 km.  At 0.5 km resolution this is ~1760 × 1780 pixels but the
    actual pixel size in the GEOS projection varies with distance from
    the sub-satellite point (140.7°E) — pixels are elongated east-west
    for Sri Lanka (~60° from sub-satellite).  Effective ground resolution
    at the site ≈ 0.9 km (column) × 0.6 km (row).

    Parameters
    ----------
    hsd       : output of read_hsd_segment().
    lat_min/max, lon_min/max : bounding box (degrees).

    Returns
    -------
    dict with keys:
        "counts"  : np.ndarray[uint16]  cropped counts
        "row_off" : int   row offset within the full-disk image
        "col_off" : int   column offset
        "proj"    : dict  same projection dict (unchanged)
        "timestamp", "segment", "calib" : forwarded unchanged
    """
    proj    = hsd["proj"]
    counts  = hsd["counts"]          # (nlines_seg, npixels_total)
    first_l = hsd["first_line"]      # 1-based line number of first row in segment

    n_rows, n_cols = counts.shape

    # Compute lat/lon for all pixel corners
    rows_idx = np.arange(n_rows) + first_l   # 1-based global row
    cols_idx = np.arange(n_cols)

    # Convert bounding box corners to pixel coordinates
    corners_lat = np.array([lat_min, lat_min, lat_max, lat_max])
    corners_lon = np.array([lon_min, lon_max, lon_min, lon_max])
    cs, rs = latlon_to_pixel(corners_lat, corners_lon, proj)

    # Clip to valid array bounds
    c0 = max(0, int(np.nanmin(cs)) - 5)
    c1 = min(n_cols - 1, int(np.nanmax(cs)) + 5)
    r0 = max(0, int(np.nanmin(rs)) - first_l - 5)
    r1 = min(n_rows - 1, int(np.nanmax(rs)) - first_l + 5)

    if r0 >= r1 or c0 >= c1:
        logger.warning(
            f"  [extract_roi] ROI out of segment bounds — "
            f"r=[{r0},{r1}] c=[{c0},{c1}]"
        )
        return hsd  # return full segment as fallback

    roi_counts = counts[r0:r1+1, c0:c1+1]

    return {
        "counts"    : roi_counts,
        "row_off"   : r0 + first_l,
        "col_off"   : c0,
        "proj"      : proj,
        "timestamp" : hsd["timestamp"],
        "segment"   : hsd["segment"],
        "calib"     : hsd["calib"],
    }


def site_pixel(proj: dict) -> tuple[float, float]:
    """
    Return the (col, row) pixel coordinates of the site in the full-disk image.

    Returns
    -------
    col, row : float  (fractional, 1-based Himawari indexing)
    """
    return latlon_to_pixel(_SITE_LAT, _SITE_LON, proj)


def km_to_pixels(km: float, proj: dict, axis: str = "col") -> float:
    """
    Rough conversion: km on the ground → number of pixels.

    Accounts for the foreshortening of the GEOS projection at the site
    (~60° from sub-satellite point), which elongates pixels east-west
    by a factor of ~1.8 relative to the sub-satellite nadir pixel size.

    Parameters
    ----------
    km   : distance in km.
    proj : projection dict.
    axis : "col" (east-west) or "row" (north-south).

    Returns
    -------
    float  approximate number of pixels.
    """
    # Pixel size at nadir (sub-satellite point) for Band 3: 0.5 km
    pix_nadir = 0.5

    # Foreshortening factor for the site (empirically derived)
    if axis == "col":
        factor = 1.8   # east-west elongated due to oblique view
    else:
        factor = 1.1   # north-south less affected

    return km / (pix_nadir * factor)
