"""
scripts/fetch_himawari.py
──────────────────────────────────────────────────────────────────────────────
Download Himawari-8 AHI Band-3 (visible 0.64 µm) segment files from the
NOAA public AWS S3 bucket (no credentials required).

Strategy
────────
• Downloads only daytime frames (solar elevation > 0° at the site).
  For Sri Lanka (6.79°N, 79.90°E, UTC): daytime ≈ 01:00–13:00 UTC.
  → 7 frames/hour × 12 hours × 365 days ≈ 30,660 files per year.

• Two segments per frame:
    S05 — covers ~0°–12°N  (site is here)
    S04 — covers ~12°–24°N (upstream during NE monsoon; clouds from north)
  → 61,320 downloads per year.

• Each file: ~25–35 MB bz2.  Total for 1 year: ~1.8 TB.
  This is substantial — see --start / --end flags to limit the range.
  For testing, use --days 3 to download just a few days.

• Resume-safe: skips files that already exist locally.
• Parallel downloads: --workers N (default 4).

Output
──────
    data/himawari/{year}/{month:02d}/
        HS_H08_{date}_{time}_B03_FLDK_R05_S{seg}10.DAT.bz2

Run
───
    # Full overlap period (Apr 2022 – Mar 2023), segment 5 only
    python scripts/fetch_himawari.py --start 2022-04-01 --end 2023-03-31

    # Quick test: 3 days, both segments, 4 parallel workers
    python scripts/fetch_himawari.py --start 2022-06-01 --days 3 --segments 4 5 --workers 4

    # Only segment 5, 1 specific day
    python scripts/fetch_himawari.py --start 2022-11-15 --days 1 --segments 5
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.himawari_loader import download_segment, s3_url
from src.utils.logger import get_logger

logger = get_logger("fetch_himawari")

# Site
_LAT = 6.7912
_LON = 79.9005
_LOCATION = pvlib.location.Location(_LAT, _LON, tz="UTC", altitude=20)

_HIM_DIR = Path("data/himawari")


def daytime_timestamps(
    start: pd.Timestamp,
    end:   pd.Timestamp,
    freq:  str = "10min",
) -> list[pd.Timestamp]:
    """
    Generate UTC timestamps between start and end at freq intervals,
    keeping only daytime ones (solar elevation > 0° at site).
    """
    idx = pd.date_range(start, end, freq=freq, tz="UTC")

    # Batch solar position computation (fast)
    sol = _LOCATION.get_solarposition(idx)
    day_mask = sol["elevation"] > 0.0

    return list(idx[day_mask])


def _download_one(args: tuple) -> tuple[bool, str]:
    """Worker function for parallel download."""
    dt, segment, out_dir = args
    try:
        path = download_segment(dt, out_dir=out_dir, band=3, segment=segment)
        return (path is not None), str(dt)
    except Exception as e:
        logger.error(f"  Error downloading {dt} seg{segment}: {e}")
        return False, str(dt)


def main():
    parser = argparse.ArgumentParser(
        description="Download Himawari-8 Band-3 segments from NOAA S3"
    )
    parser.add_argument("--start",    default="2022-04-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",      default=None,
                        help="End date inclusive (YYYY-MM-DD). If omitted, use --days.")
    parser.add_argument("--days",     type=int, default=None,
                        help="Number of days to download (overrides --end).")
    parser.add_argument("--segments", type=int, nargs="+", default=[5],
                        choices=[1,2,3,4,5,6,7,8,9,10],
                        help="Himawari segment numbers to download (default: 5). "
                             "Segment 5 covers ~0–12°N (site). "
                             "Segment 4 covers ~12–24°N (upstream NE monsoon).")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel download threads (default 4).")
    parser.add_argument("--out-dir",  default=str(_HIM_DIR),
                        help=f"Output directory (default: {_HIM_DIR})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print URLs without downloading.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    if args.days:
        end = start + pd.Timedelta(days=args.days) - pd.Timedelta(seconds=1)
    elif args.end:
        end = pd.Timestamp(args.end, tz="UTC").replace(hour=23, minute=50)
    else:
        parser.error("Provide --end or --days")

    logger.info(f"Fetching Himawari-8 Band-3")
    logger.info(f"  Period   : {start.date()} → {end.date()}")
    logger.info(f"  Segments : {args.segments}")
    logger.info(f"  Workers  : {args.workers}")
    logger.info(f"  Out dir  : {out_dir}")

    # Generate daytime timestamps
    logger.info("  Computing daytime timestamps …")
    timestamps = daytime_timestamps(start, end)
    logger.info(f"  Daytime frames : {len(timestamps):,}")

    # Build download task list
    tasks = [(dt, seg, out_dir) for dt in timestamps for seg in args.segments]

    # Count already-cached files
    already = sum(
        1 for dt, seg, od in tasks
        if (od / str(dt.year) / f"{dt.month:02d}" /
            s3_url(dt, 3, seg).split("/")[-1]).exists()
    )
    logger.info(
        f"  Total tasks : {len(tasks):,}  |  "
        f"Already cached: {already:,}  |  "
        f"To download: {len(tasks)-already:,}"
    )

    if args.dry_run:
        print("\nDRY RUN — URLs (first 10):")
        for dt, seg, _ in tasks[:10]:
            print(f"  {s3_url(dt, 3, seg)}")
        print(f"  ... and {len(tasks)-10} more")
        return

    # Storage estimate
    avg_mb = 30
    total_gb = (len(tasks) - already) * avg_mb / 1024
    logger.info(
        f"  Estimated download : ~{total_gb:.1f} GB "
        f"(~{avg_mb} MB per file)"
    )
    if total_gb > 10:
        ans = input(
            f"\n  This will download ~{total_gb:.0f} GB. Continue? [y/N] "
        ).strip().lower()
        if ans != "y":
            print("  Aborted.")
            return

    # Download
    t0 = time.time()
    success = 0
    failed  = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            ok, ts_str = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = success / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"  [{i}/{len(tasks)}]  "
                    f"ok={success}  failed={failed}  "
                    f"rate={rate:.1f}/s  ETA={eta:.0f} min"
                )

    elapsed = time.time() - t0
    logger.info(
        f"\nDone.  {success} downloaded  {failed} failed  "
        f"in {elapsed/60:.1f} min"
    )

    # Write manifest
    manifest = out_dir / "download_manifest.csv"
    rows = []
    for dt, seg, od in tasks:
        fname = s3_url(dt, 3, seg).split("/")[-1]
        fpath = od / str(dt.year) / f"{dt.month:02d}" / fname
        rows.append({"timestamp_utc": dt, "segment": seg,
                     "filename": fname, "exists": fpath.exists()})

    pd.DataFrame(rows).to_csv(manifest, index=False)
    logger.info(f"  Manifest → {manifest}  ({len(rows)} entries)")


if __name__ == "__main__":
    main()
