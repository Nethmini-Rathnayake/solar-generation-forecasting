"""
src/cmv/optical_flow.py
──────────────────────────────────────────────────────────────────────────────
Cloud Motion Vector extraction using dense optical flow.

Uses scikit-image's Iterative Lucas-Kanade (ILK) optical flow, which
requires no GPU and no OpenCV.  For two consecutive Himawari Band-3 frames
10 minutes apart, ILK gives sub-pixel accuracy on cloud features.

Physics interpretation
──────────────────────
The optical flow vector (u, v) at each pixel represents how many pixels the
image pattern moved between the two frames.  Converting to physical units:

    cloud_speed = sqrt(u² + v²) × pixel_size_km / Δt_hours   [km/h]
    cloud_dir   = atan2(u, -v)                                  [° from N, CW]

where pixel_size_km is the effective ground resolution at the site's oblique
view angle (~0.9 km column, ~0.55 km row for Sri Lanka viewed from Himawari).

Upstream cloud state
────────────────────
The "upstream" pixel at distance d km is found by extrapolating the motion
vector backwards from the site pixel.  Its reflectance (proxy for cloud
opacity) predicts what will arrive at the site in (d / speed) minutes.

Usage
─────
    from src.cmv.optical_flow import compute_dense_flow, extract_cmv, upstream_state

    flow_uv  = compute_dense_flow(frame_t0, frame_t1)
    cmv      = extract_cmv(flow_uv, site_col, site_row, dt_min=10)
    upstream = upstream_state(frame_t0, cmv, site_col, site_row,
                              distances_km=[10, 20, 40])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from skimage.registration import optical_flow_ilk
from scipy.ndimage import gaussian_filter, zoom

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Effective pixel size at site (GEOS projection foreshortening at ~60° off-nadir)
_PIX_KM_COL = 0.90   # km per pixel, east-west
_PIX_KM_ROW = 0.55   # km per pixel, north-south


@dataclass
class CMV:
    """Cloud Motion Vector result for one site at one time step."""
    timestamp:        "pd.Timestamp"    # observation time (UTC, frame t0)
    u_pix_per_frame:  float             # east-west displacement (pixels/10 min)
    v_pix_per_frame:  float             # north-south displacement (pixels/10 min)
    speed_kmh:        float             # cloud speed (km/h)
    direction_deg:    float             # direction from North, clockwise (deg)
    confidence:       float             # 0–1 (spatial coherence of flow field)
    site_reflectance: float             # reflectance at site pixel (0–1 proxy)
    dt_min:           float             # time between frames (minutes)


def _normalise_frame(counts: np.ndarray) -> np.ndarray:
    """
    Convert raw 10-bit counts to normalised reflectance proxy [0, 1].

    Uses robust percentile normalisation so clouds (high counts) → 1
    and clear sky / ocean (low counts) → ~0.  Gaussian smoothing reduces
    sensor noise without blurring cloud edges significantly.
    """
    arr = counts.astype(np.float32)

    p2, p98 = np.percentile(arr, [2, 98])
    if p98 - p2 < 1e-6:   # catches truly blank frames regardless of scale
        return np.zeros_like(arr)

    arr = (arr - p2) / (p98 - p2)
    arr = np.clip(arr, 0.0, 1.0)
    arr = gaussian_filter(arr, sigma=1.5)   # mild smoothing
    return arr


def _downsample(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample a frame by integer factor using area-average."""
    if factor == 1:
        return frame
    h, w = frame.shape
    h2, w2 = h // factor, w // factor
    return frame[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def compute_dense_flow(
    frame_t0:       np.ndarray,
    frame_t1:       np.ndarray,
    downsample:     int   = 2,
    num_warp:       int   = 5,
    gaussian_sigma: float = 1.0,
    prefilter:      bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow between two consecutive satellite frames.

    Uses scikit-image ILK (Iterative Lucas-Kanade) with multi-scale
    warping.  Returns the (row, col) displacement field at the resolution
    of the downsampled frames.

    Parameters
    ----------
    frame_t0, frame_t1 : np.ndarray
        Normalised reflectance frames [0, 1], same shape.
    downsample : int
        Downsample factor before flow computation (speeds up ILK; flow is
        later scaled back).  Default 2 → process at 1 km instead of 0.5 km.
    num_warp : int
        Number of ILK warping iterations.
    gaussian_sigma : float
        Pre-smoothing applied before ILK to reduce noise sensitivity.
    prefilter : bool
        Apply Gaussian prefilter.

    Returns
    -------
    row_flow, col_flow : np.ndarray
        Displacement in row/col directions (pixels at original resolution).
        Positive row_flow = southward motion.
        Positive col_flow = eastward motion.
    """
    if frame_t0.shape != frame_t1.shape:
        raise ValueError(
            f"Frame shape mismatch: {frame_t0.shape} vs {frame_t1.shape}"
        )

    # Normalise both frames
    f0 = _normalise_frame(frame_t0)
    f1 = _normalise_frame(frame_t1)

    # Downsample for speed
    if downsample > 1:
        f0_ds = _downsample(f0, downsample)
        f1_ds = _downsample(f1, downsample)
    else:
        f0_ds, f1_ds = f0, f1

    # Pre-smoothing
    if prefilter:
        f0_ds = gaussian_filter(f0_ds, sigma=gaussian_sigma)
        f1_ds = gaussian_filter(f1_ds, sigma=gaussian_sigma)

    # ILK optical flow (row_flow, col_flow in downsampled pixels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row_flow_ds, col_flow_ds = optical_flow_ilk(
            f0_ds, f1_ds,
            radius=15,
            num_warp=num_warp,
            gaussian=True,
            prefilter=False,
        )

    # Scale back to original resolution
    if downsample > 1:
        scale = float(downsample)
        row_flow = zoom(row_flow_ds, scale, order=1) * scale
        col_flow = zoom(col_flow_ds, scale, order=1) * scale

        # Trim to original size (zoom may add 1 pixel)
        row_flow = row_flow[:f0.shape[0], :f0.shape[1]]
        col_flow = col_flow[:f0.shape[0], :f0.shape[1]]
    else:
        row_flow, col_flow = row_flow_ds, col_flow_ds

    return row_flow, col_flow


def extract_cmv(
    row_flow:  np.ndarray,
    col_flow:  np.ndarray,
    site_col:  float,
    site_row:  float,
    col_off:   int,
    row_off:   int,
    dt_min:    float = 10.0,
    radius_pix: int  = 20,
    timestamp: "pd.Timestamp | None" = None,
    frame_t0:  np.ndarray | None = None,
) -> CMV:
    """
    Extract the cloud motion vector at the site from the dense flow field.

    Rather than using a single pixel (noisy), the site CMV is the
    median flow vector within a radius_pix neighbourhood around the site
    pixel.  Confidence is the spatial coherence (1 - normalised IQR).

    Parameters
    ----------
    row_flow, col_flow : dense flow arrays (pixels/frame).
    site_col, site_row : site pixel coordinates in full-disk image.
    col_off, row_off   : ROI offset in full-disk image.
    dt_min             : time between frames (minutes, default 10).
    radius_pix         : neighbourhood radius for median pooling.
    timestamp          : UTC timestamp of frame t0.
    frame_t0           : normalised reflectance at frame t0 (for site value).

    Returns
    -------
    CMV dataclass.
    """
    # Convert full-disk pixel to ROI pixel
    sc = site_col - col_off
    sr = site_row - row_off

    H, W = row_flow.shape
    sc, sr = int(round(sc)), int(round(sr))
    sc = np.clip(sc, radius_pix, W - radius_pix - 1)
    sr = np.clip(sr, radius_pix, H - radius_pix - 1)

    # Neighbourhood
    r0, r1 = sr - radius_pix, sr + radius_pix
    c0, c1 = sc - radius_pix, sc + radius_pix
    u_patch = col_flow[r0:r1, c0:c1]   # east-west
    v_patch = row_flow[r0:r1, c0:c1]   # north-south (positive = south)

    u_med = float(np.median(u_patch))
    v_med = float(np.median(v_patch))

    # Confidence: 1 - IQR/median_magnitude (spatial coherence)
    mag_patch = np.sqrt(u_patch**2 + v_patch**2)
    med_mag = float(np.median(mag_patch))
    iqr_mag = float(np.percentile(mag_patch, 75) - np.percentile(mag_patch, 25))
    confidence = float(np.clip(1.0 - (iqr_mag / (med_mag + 1e-6)), 0.0, 1.0))

    # Convert to physical units
    # u (east-west): positive = eastward; pixel size in col direction
    # v (north-south): positive = southward (row index increases downward)
    dt_h = dt_min / 60.0
    speed_e = u_med * _PIX_KM_COL / dt_h   # km/h eastward
    speed_n = -v_med * _PIX_KM_ROW / dt_h  # km/h northward (flip sign: row↑=N)

    speed_kmh = float(np.sqrt(speed_e**2 + speed_n**2))

    # Direction: meteorological convention (direction FROM which wind comes)
    # Here we use direction TO which clouds move (FROM_direction + 180)
    direction_to = float(np.degrees(np.arctan2(speed_e, speed_n)) % 360)

    # Site reflectance (cloud opacity proxy)
    site_ref = 0.0
    if frame_t0 is not None:
        norm = _normalise_frame(frame_t0)
        site_ref = float(norm[sr, sc]) if 0 <= sr < norm.shape[0] and 0 <= sc < norm.shape[1] else 0.0

    return CMV(
        timestamp        = timestamp,
        u_pix_per_frame  = u_med,
        v_pix_per_frame  = v_med,
        speed_kmh        = speed_kmh,
        direction_deg    = direction_to,
        confidence       = confidence,
        site_reflectance = site_ref,
        dt_min           = dt_min,
    )


def upstream_state(
    frame:          np.ndarray,
    cmv:            CMV,
    site_col:       float,
    site_row:       float,
    col_off:        int,
    row_off:        int,
    distances_km:   list[float] = [10, 20, 40],
) -> dict[float, float]:
    """
    Read cloud reflectance at upstream pixels along the motion vector.

    "Upstream" = opposite direction of cloud motion → these clouds will
    arrive at the site in (distance / speed) minutes.

    Parameters
    ----------
    frame          : normalised reflectance image at time t0.
    cmv            : CMV result from extract_cmv().
    site_col/row   : full-disk pixel of site.
    col_off/row_off: ROI offsets.
    distances_km   : list of distances to sample upstream (km).

    Returns
    -------
    dict {distance_km: reflectance}  (reflectance ≈ cloud opacity proxy 0–1)
    """
    norm = _normalise_frame(frame)
    H, W = norm.shape

    # Unit vector pointing upstream (opposite to motion)
    mag = np.sqrt(cmv.u_pix_per_frame**2 + cmv.v_pix_per_frame**2)
    if mag < 0.05:
        return {d: float(norm[int(site_row - row_off), int(site_col - col_off)])
                for d in distances_km}

    ux = -cmv.u_pix_per_frame / mag   # upstream east-west
    uy = -cmv.v_pix_per_frame / mag   # upstream north-south

    site_c = site_col - col_off
    site_r = site_row - row_off

    result: dict[float, float] = {}
    for d_km in distances_km:
        dc = d_km / _PIX_KM_COL * ux
        dr = d_km / _PIX_KM_ROW * uy
        c = int(round(site_c + dc))
        r = int(round(site_r + dr))

        if 0 <= r < H and 0 <= c < W:
            result[d_km] = float(norm[r, c])
        else:
            result[d_km] = np.nan

    return result


def flow_quality_check(
    row_flow: np.ndarray,
    col_flow: np.ndarray,
    max_speed_kmh: float = 200.0,
    dt_min: float = 10.0,
) -> bool:
    """
    Basic sanity check on the computed flow field.

    Returns True if the flow is physically plausible (cloud speed < 200 km/h).
    """
    max_u_pix = max_speed_kmh * (dt_min / 60.0) / _PIX_KM_COL
    max_v_pix = max_speed_kmh * (dt_min / 60.0) / _PIX_KM_ROW

    u99 = float(np.percentile(np.abs(col_flow), 99))
    v99 = float(np.percentile(np.abs(row_flow), 99))

    if u99 > max_u_pix or v99 > max_v_pix:
        logger.warning(
            f"  [flow_quality] Suspicious flow: u99={u99:.1f}px, v99={v99:.1f}px "
            f"(limits: {max_u_pix:.1f}, {max_v_pix:.1f})"
        )
        return False
    return True
