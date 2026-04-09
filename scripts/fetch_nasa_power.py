"""
scripts/fetch_nasa_power.py
----------------------------
Run this script once to pull all NASA POWER satellite data (2020–2026)
for the University of Moratuwa site and save it to data/external/.

Run from the project root:
    python scripts/fetch_nasa_power.py

Optional flags:
    --start 20200101     override start date
    --end   20261231     override end date
    --config configs/site.yaml
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.nasa_power import fetch_nasa_power, save_raw

logger = get_logger("fetch_nasa_power")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NASA POWER hourly data for the configured site."
    )
    parser.add_argument("--config", default="configs/site.yaml")
    parser.add_argument("--start",  default=None, help="Override start YYYYMMDD")
    parser.add_argument("--end",    default=None, help="Override end YYYYMMDD")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    site = cfg["site"]
    nasa = cfg["nasa_power"]

    logger.info(f"Site      : {site['name']}")
    logger.info(f"Location  : {site['latitude']}° N, {site['longitude']}° E")
    logger.info(f"Period    : {args.start or nasa['start_date']} → {args.end or nasa['end_date']}")

    # ── Fetch ──────────────────────────────────────────────────────────────
    df = fetch_nasa_power(cfg, start_date=args.start, end_date=args.end)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = save_raw(df, cfg)
    logger.info(f"Done. Data saved to: {out_path}")

    # ── Quick preview ──────────────────────────────────────────────────────
    print("\n── First 3 rows ──────────────────────────────────────")
    print(df.head(3).to_string())
    print(f"\nShape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Range  : {df.index.min()}  →  {df.index.max()}")


if __name__ == "__main__":
    main()
