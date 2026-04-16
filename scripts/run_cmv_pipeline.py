"""
scripts/run_cmv_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Cloud Motion Vector pipeline: Himawari-8 Band-3 → CMV features → CSV.

Stages
──────
  1. Scan local Himawari cache for available frame pairs (10-min apart)
  2. For each consecutive pair:
       a. Read both HSD files (decompress + parse binary)
       b. Extract ROI (8°×8° box around site)
       c. Compute dense optical flow (ILK)
       d. Extract CMV at site pixel (median neighbourhood)
       e. Sample upstream cloud reflectance at 5/10/20/40 km
       f. Compute shadow arrival times (solar geometry corrected)
  3. Write CMV feature CSV: data/interim/himawari_cmv_features.csv
  4. Merge into existing 5-min feature matrix
  5. Plot CMV diagnostics

Output columns
──────────────
    timestamp_utc          : 10-min UTC timestamp
    cloud_speed_kmh        : cloud speed [km/h]
    cloud_direction_deg    : direction cloud moves toward [°N CW]
    shadow_offset_km       : solar geometry shadow displacement [km]
    solar_zenith_deg       : solar zenith angle [°]
    shadow_arrival_5km     : minutes until 5-km-upstream shadow arrives
    shadow_arrival_10km    : minutes until 10-km-upstream shadow arrives
    shadow_arrival_20km    : minutes until 20-km-upstream shadow arrives
    shadow_arrival_40km    : minutes until 40-km-upstream shadow arrives
    upstream_ref_5km       : cloud reflectance 5 km upstream [0–1]
    upstream_ref_10km      : cloud reflectance 10 km upstream [0–1]
    upstream_ref_20km      : cloud reflectance 20 km upstream [0–1]
    upstream_ref_40km      : cloud reflectance 40 km upstream [0–1]
    cmv_confidence         : flow field coherence [0–1]
    site_reflectance       : cloud reflectance at site pixel [0–1]

Run
───
    # Process all available frames in cache
    python scripts/run_cmv_pipeline.py

    # Process specific date range
    python scripts/run_cmv_pipeline.py --start 2022-06-01 --end 2022-06-30

    # Visualise CMV for one specific day
    python scripts/run_cmv_pipeline.py --start 2022-06-15 --days 1 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger
from src.data.himawari_loader import (
    read_hsd_segment,
    extract_roi,
    site_pixel,
    km_to_pixels,
)
from src.cmv.optical_flow import (
    compute_dense_flow,
    extract_cmv,
    upstream_state,
    flow_quality_check,
)
from src.cmv.shadow_predictor import (
    compute_shadow_features,
    build_null_features,
    annotate_pv_with_cmv,
)

logger = get_logger("run_cmv_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HIM_DIR  = Path("data/himawari")
_OUT_CSV  = Path("data/interim/himawari_cmv_features.csv")
_FIG_DIR  = Path("results/figures/cmv")
_PV_5MIN  = Path("data/interim/local_5min_utc.csv")
_FEAT_OUT = Path("data/processed/feature_matrix_cmv.parquet")

# ── ROI definition ─────────────────────────────────────────────────────────────
# 8°×8° box centred on site — enough to capture 40 km upstream in any direction
_ROI = dict(lat_min=2.5, lat_max=11.0, lon_min=75.5, lon_max=84.5)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Discover available frame pairs
# ─────────────────────────────────────────────────────────────────────────────

def discover_frame_pairs(
    him_dir: Path,
    start:   pd.Timestamp | None = None,
    end:     pd.Timestamp | None = None,
    segment: int = 5,
) -> list[tuple[Path, Path, pd.Timestamp]]:
    """
    Find all consecutive 10-min frame pairs in the local Himawari cache.

    Returns
    -------
    list of (path_t0, path_t1, timestamp_t0)
    """
    # Collect all .bz2 files for the requested segment
    pattern = f"**/*_S{segment:02d}10.DAT.bz2"
    all_files = sorted(him_dir.glob(pattern))

    if not all_files:
        logger.warning(f"  No Himawari files found in {him_dir} for segment {segment}")
        return []

    # Build timestamp → path mapping
    ts_map: dict[pd.Timestamp, Path] = {}
    for fpath in all_files:
        # Filename: HS_H08_{YYYYMMDD}_{HHMM}_B03_FLDK_R05_S0510.DAT.bz2
        parts = fpath.stem.split("_")
        try:
            date_str = parts[2]   # YYYYMMDD
            time_str = parts[3]   # HHMM
            ts = pd.Timestamp(
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                f"T{time_str[:2]}:{time_str[2:4]}:00",
                tz="UTC"
            )
            ts_map[ts] = fpath
        except (IndexError, ValueError):
            continue

    # Filter by date range
    timestamps = sorted(ts_map.keys())
    if start:
        timestamps = [t for t in timestamps if t >= start]
    if end:
        timestamps = [t for t in timestamps if t <= end]

    # Find consecutive pairs (exactly 10 min apart)
    pairs = []
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if (t1 - t0) == pd.Timedelta(minutes=10):
            pairs.append((ts_map[t0], ts_map[t1], t0))

    logger.info(
        f"  Found {len(all_files)} files → {len(pairs)} consecutive pairs"
        f"  (segment {segment})"
    )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Process one frame pair → CMV features
# ─────────────────────────────────────────────────────────────────────────────

def process_pair(
    path_t0: Path,
    path_t1: Path,
    timestamp: pd.Timestamp,
) -> dict:
    """
    Full pipeline for a single 10-min frame pair.

    Returns a feature dict (all floats).  Returns null features on error.
    """
    try:
        # Read both frames
        hsd0 = read_hsd_segment(path_t0)
        hsd1 = read_hsd_segment(path_t1)

        # Use timestamp from file header if available
        if hsd0.get("timestamp") is not None:
            timestamp = hsd0["timestamp"]

        # Use projection from first frame
        proj = hsd0["proj"]
        if not proj:
            logger.warning(f"  No projection info in {path_t0.name}")
            return build_null_features(timestamp)

        # Extract ROI for both frames
        roi0 = extract_roi(hsd0, **_ROI)
        roi1 = extract_roi(hsd1, **_ROI)

        if roi0["counts"].shape != roi1["counts"].shape:
            logger.warning(
                f"  ROI shape mismatch: {roi0['counts'].shape} vs "
                f"{roi1['counts'].shape}"
            )
            return build_null_features(timestamp)

        # Site pixel in full-disk coordinates
        site_col, site_row = site_pixel(proj)

        # Compute optical flow
        row_flow, col_flow = compute_dense_flow(
            roi0["counts"].astype(np.float32),
            roi1["counts"].astype(np.float32),
            downsample=2,
        )

        # Quality check
        if not flow_quality_check(row_flow, col_flow, dt_min=10.0):
            logger.warning(f"  [CMV] Flow quality check failed at {timestamp}")
            return build_null_features(timestamp)

        # Extract CMV at site
        cmv = extract_cmv(
            row_flow, col_flow,
            site_col=site_col,
            site_row=site_row,
            col_off=roi0["col_off"],
            row_off=roi0["row_off"],
            dt_min=10.0,
            radius_pix=25,
            timestamp=timestamp,
            frame_t0=roi0["counts"].astype(np.float32),
        )

        # Upstream cloud state
        upstream = upstream_state(
            roi0["counts"].astype(np.float32),
            cmv,
            site_col=site_col,
            site_row=site_row,
            col_off=roi0["col_off"],
            row_off=roi0["row_off"],
            distances_km=[5, 10, 20, 40],
        )

        # Shadow features
        features = compute_shadow_features(cmv, upstream, timestamp)

        logger.debug(
            f"  {timestamp}  speed={cmv.speed_kmh:.1f} km/h  "
            f"dir={cmv.direction_deg:.0f}°  conf={cmv.confidence:.2f}"
        )
        return features

    except Exception as e:
        logger.error(f"  Error processing {path_t0.name}: {e}")
        return build_null_features(timestamp)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_cmv_diagnostics(cmv_df: pd.DataFrame, fig_dir: Path) -> None:
    """
    5-panel CMV diagnostics plot.

    Panel 1: Cloud speed time series
    Panel 2: Cloud direction rose (polar)
    Panel 3: Upstream reflectance (5/10/20/40 km) — daily mean
    Panel 4: Shadow arrival time at 10 km vs actual PV ramp events
    Panel 5: CMV confidence distribution by month
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.gridspec as gridspec
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.32)

    df = cmv_df.copy()
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["cloud_speed_kmh"].notna()]

    # ── Panel 1: Speed time series ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    daily_speed = df["cloud_speed_kmh"].resample("D").median()
    ax1.fill_between(daily_speed.index, daily_speed.values,
                     alpha=0.35, color="#2a9d8f")
    ax1.plot(daily_speed.index, daily_speed.values,
             color="#2a9d8f", lw=1.5, label="Daily median")
    ax1.set_ylabel("Cloud Speed (km/h)", fontsize=10)
    ax1.set_title("Daily Median Cloud Speed — Himawari-8 Band-3 CMV", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.legend(fontsize=9)

    # ── Panel 2: Wind rose (polar) ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0], projection="polar")
    dirs_rad = np.radians(df["cloud_direction_deg"].dropna().values)
    counts, bins = np.histogram(dirs_rad, bins=36, range=(0, 2*np.pi))
    width = 2 * np.pi / 36
    bars = ax2.bar(bins[:-1], counts, width=width, bottom=0, align="edge",
                   color=plt.cm.viridis(counts / counts.max()), alpha=0.8)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)   # clockwise
    ax2.set_title("Cloud Motion Direction\n(direction cloud moves toward)",
                  fontsize=10, pad=20)

    # ── Panel 3: Upstream reflectance ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    colors = {"upstream_ref_5km":  "#e63946",
              "upstream_ref_10km": "#f4a261",
              "upstream_ref_20km": "#2a9d8f",
              "upstream_ref_40km": "#457b9d"}
    labels = {"upstream_ref_5km":  "5 km",
              "upstream_ref_10km": "10 km",
              "upstream_ref_20km": "20 km",
              "upstream_ref_40km": "40 km"}
    for col, clr in colors.items():
        if col in df.columns:
            daily = df[col].resample("D").mean()
            ax3.plot(daily.index, daily.values, color=clr,
                     lw=1.5, alpha=0.85, label=labels[col])
    ax3.set_ylabel("Cloud Reflectance (opacity proxy)", fontsize=10)
    ax3.set_title("Daily Mean Upstream Cloud Reflectance", fontsize=10)
    ax3.legend(title="Distance upstream", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax3.set_ylim(0, 1)

    # ── Panel 4: Shadow arrival time ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    for col, lbl, clr in [
        ("shadow_arrival_5km",  "5 km",  "#e63946"),
        ("shadow_arrival_10km", "10 km", "#f4a261"),
        ("shadow_arrival_20km", "20 km", "#2a9d8f"),
    ]:
        if col in df.columns:
            monthly = df[col].resample("ME").median()
            ax4.plot(monthly.index, monthly.values, "o-",
                     color=clr, lw=1.5, ms=5, label=lbl)
    ax4.set_ylabel("Shadow Arrival Time (minutes)", fontsize=10)
    ax4.set_title("Median Shadow Arrival Time by Month", fontsize=10)
    ax4.legend(title="Upstream distance", fontsize=9)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.set_ylim(0)

    # ── Panel 5: Confidence by month ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    monthly_conf = [
        df[df.index.month == m]["cmv_confidence"].dropna().values
        for m in range(1, 13)
    ]
    months = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    bp = ax5.boxplot(
        [c for c in monthly_conf if len(c) > 0],
        labels=[months[i] for i, c in enumerate(monthly_conf) if len(c) > 0],
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    wet = {4, 5, 10, 11, 12}
    for i, (patch, m) in enumerate(
        [(p, i+1) for i, p in enumerate(bp["boxes"])]
    ):
        patch.set_facecolor("#577590" if m in wet else "#90be6d")
        patch.set_alpha(0.7)
    ax5.set_ylabel("CMV Confidence [0–1]", fontsize=10)
    ax5.set_title("CMV Confidence by Month\n(green=dry, blue=wet)", fontsize=10)
    ax5.set_ylim(0, 1)

    fig.suptitle(
        "Himawari-8 Cloud Motion Vector — University of Moratuwa Site\n"
        "6.79°N 79.90°E  |  Band 3 (0.64 µm)  |  10-min resolution",
        fontsize=13, fontweight="bold",
    )
    out = fig_dir / "cmv_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_sample_flow_field(
    frame0:   np.ndarray,
    row_flow: np.ndarray,
    col_flow: np.ndarray,
    site_col: float,
    site_row: float,
    col_off:  int,
    row_off:  int,
    timestamp: pd.Timestamp,
    fig_dir:  Path,
) -> None:
    """
    Visualise the optical flow field overlaid on the Himawari visible image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Optical Flow — Himawari-8 Band-3  |  {timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=12
    )

    # Left: visible image with flow vectors
    ax = axes[0]
    ax.imshow(frame0, cmap="gray", vmin=0, vmax=0.8, origin="upper")

    # Quiver plot (every 20th pixel)
    step = 20
    H, W = row_flow.shape
    ys, xs = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing="ij")
    u_ds = col_flow[::step, ::step]
    v_ds = row_flow[::step, ::step]
    ax.quiver(xs, ys, u_ds, -v_ds, color="lime", scale=50, width=0.002,
              headwidth=4, alpha=0.7)

    # Mark site
    sc = site_col - col_off
    sr = site_row - row_off
    ax.plot(sc, sr, "r*", ms=15, label="Site")
    ax.legend(fontsize=10)
    ax.set_title("Visible Image + Flow Vectors", fontsize=10)

    # Right: flow magnitude
    mag = np.sqrt(col_flow**2 + row_flow**2)
    im = axes[1].imshow(mag, cmap="hot", origin="upper")
    plt.colorbar(im, ax=axes[1], label="Flow magnitude (pixels/10min)")
    axes[1].plot(sc, sr, "b*", ms=15, label="Site")
    axes[1].legend(fontsize=10)
    axes[1].set_title("Flow Magnitude", fontsize=10)

    out = fig_dir / f"flow_{timestamp.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved flow plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Himawari-8 CMV pipeline")
    parser.add_argument("--start",   default=None,
                        help="Start date (YYYY-MM-DD). Default: use all cached data.")
    parser.add_argument("--end",     default=None,
                        help="End date (YYYY-MM-DD).")
    parser.add_argument("--days",    type=int, default=None,
                        help="Number of days (overrides --end).")
    parser.add_argument("--segment", type=int, default=5,
                        help="Himawari segment to process (default 5).")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate diagnostic plots.")
    parser.add_argument("--plot-flow", action="store_true",
                        help="Save optical flow field plots (one per pair; many files).")
    parser.add_argument("--merge-pv", action="store_true",
                        help="Merge CMV features into the 5-min PV feature matrix.")
    parser.add_argument("--him-dir", default=str(_HIM_DIR))
    args = parser.parse_args()

    him_dir = Path(args.him_dir)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Date range
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    if args.days and start:
        end = start + pd.Timedelta(days=args.days) - pd.Timedelta(seconds=1)
    elif args.end:
        end = pd.Timestamp(args.end, tz="UTC").replace(hour=23, minute=50)
    else:
        end = None

    print("\n" + "═" * 68)
    print("  HIMAWARI-8 CLOUD MOTION VECTOR PIPELINE")
    print(f"  Site: 6.7912°N 79.9005°E  |  Segment: {args.segment}")
    if start:
        print(f"  Period: {start.date()} → {end.date() if end else 'end of cache'}")
    print("═" * 68)

    # Stage 1: discover pairs
    logger.info("\nStage 1 — Discovering frame pairs …")
    pairs = discover_frame_pairs(him_dir, start=start, end=end,
                                  segment=args.segment)

    if not pairs:
        logger.error(
            "No frame pairs found. Run fetch_himawari.py first to download data."
        )
        print(
            "\n  No Himawari data found locally.\n"
            "  Download data first:\n"
            "    python scripts/fetch_himawari.py --start 2022-06-01 --days 3\n"
        )
        return

    # Stage 2: process pairs
    logger.info(f"\nStage 2 — Processing {len(pairs)} frame pairs …")
    features_list = []
    first_pair_processed = False

    for i, (p0, p1, ts) in enumerate(pairs, 1):
        feat = process_pair(p0, p1, ts)
        features_list.append(feat)

        # Save flow field plot for first pair (diagnostic)
        if args.plot_flow and not first_pair_processed:
            try:
                hsd0  = read_hsd_segment(p0)
                proj  = hsd0["proj"]
                roi0  = extract_roi(hsd0, **_ROI)
                hsd1  = read_hsd_segment(p1)
                roi1  = extract_roi(hsd1, **_ROI)
                rf, cf = compute_dense_flow(
                    roi0["counts"].astype(np.float32),
                    roi1["counts"].astype(np.float32),
                    downsample=2,
                )
                from src.cmv.optical_flow import _normalise_frame
                sc, sr = site_pixel(proj)
                plot_sample_flow_field(
                    _normalise_frame(roi0["counts"].astype(np.float32)),
                    rf, cf, sc, sr,
                    roi0["col_off"], roi0["row_off"],
                    ts, _FIG_DIR,
                )
                first_pair_processed = True
            except Exception as e:
                logger.warning(f"  Could not plot flow: {e}")

        if i % 100 == 0 or i == len(pairs):
            ok_count = sum(
                1 for f in features_list
                if not np.isnan(f.get("cloud_speed_kmh", np.nan))
            )
            logger.info(
                f"  [{i}/{len(pairs)}]  valid CMV: {ok_count}/{i} "
                f"({100*ok_count/i:.0f}%)"
            )

    # Stage 3: save CMV features
    logger.info(f"\nStage 3 — Saving CMV features → {_OUT_CSV} …")
    cmv_df = pd.DataFrame(features_list)
    if "timestamp" in cmv_df.columns:
        cmv_df = cmv_df.sort_values("timestamp")
    cmv_df.to_csv(_OUT_CSV, index=False)

    valid = cmv_df["cloud_speed_kmh"].notna().sum()
    logger.info(
        f"  Saved {len(cmv_df)} rows  |  valid CMV: {valid} "
        f"({100*valid/max(len(cmv_df),1):.0f}%)"
    )

    # Summary statistics
    print("\n" + "═" * 68)
    print("  CMV SUMMARY STATISTICS")
    print("═" * 68)
    cols = ["cloud_speed_kmh", "cloud_direction_deg", "cmv_confidence",
            "shadow_arrival_10km", "upstream_ref_10km"]
    print(cmv_df[[c for c in cols if c in cmv_df.columns]].describe().round(2).to_string())

    # Stage 4: plots
    if args.plot:
        logger.info("\nStage 4 — Generating CMV diagnostics …")
        try:
            plot_cmv_diagnostics(cmv_df, _FIG_DIR)
        except Exception as e:
            logger.error(f"  Plot failed: {e}")

    # Stage 5: merge into PV feature matrix
    if args.merge_pv:
        logger.info(f"\nStage 5 — Merging CMV into PV feature matrix …")
        if not _PV_5MIN.exists():
            logger.warning(f"  PV data not found: {_PV_5MIN} — skipping merge")
        else:
            pv = pd.read_csv(_PV_5MIN, index_col="timestamp_utc", parse_dates=True)
            pv.index = pd.to_datetime(pv.index, utc=True)
            pv_cmv = annotate_pv_with_cmv(pv, cmv_df)
            pv_cmv.to_parquet(_FEAT_OUT)
            logger.info(
                f"  Merged feature matrix → {_FEAT_OUT}  "
                f"({len(pv_cmv)} rows, {len(pv_cmv.columns)} columns)"
            )

    print("\n" + "═" * 68)
    print("  PIPELINE COMPLETE")
    print(f"  CMV features : {_OUT_CSV}")
    if args.plot:
        print(f"  Figures      : {_FIG_DIR}/")
    if args.merge_pv:
        print(f"  Feature matrix (with CMV): {_FEAT_OUT}")
    print("═" * 68)


if __name__ == "__main__":
    main()
