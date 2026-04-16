"""
scripts/run_align.py
--------------------
Aligns local PV/weather measurements with NASA POWER satellite data,
resampling the 5-minute local data to hourly UTC and clipping both
DataFrames to their common overlap window.

Output files written to data/interim/:
  local_hourly_utc.csv  — local data resampled to hourly UTC
  nasa_aligned.csv      — NASA POWER clipped to the same window

Diagnostic plots saved to results/figures/:
  data_pre_align_before_after_resample.png
  data_pre_align_local_vs_nasa.png

Run from project root:
    python scripts/run_align.py

Optional flags:
    --config configs/site.yaml         override config path
    --local  data/raw/my_data.csv      path to local CSV (auto-detects newest otherwise)
    --nasa   my_nasa_file.csv          filename within data/external/ (auto-detects otherwise)
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.data.local_loader import load_local_data
from src.data import nasa_power
from src.preproccesing.align import align_datasets, save_aligned, plot_alignment

logger = get_logger("run_align")


def _auto_detect_local(cfg: dict) -> Path:
    """Return the most recently modified CSV in data/raw/."""
    raw_dir    = resolve_path(cfg["paths"]["raw_local"])
    candidates = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Place your local PV data there or pass --local <path>."
        )
    path = candidates[-1]
    logger.info(f"Auto-detected local CSV: {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Align local PV measurements with NASA POWER satellite data."
    )
    parser.add_argument("--config", default="configs/site.yaml")
    parser.add_argument("--local",  default=None,
                        help="Path to local PV CSV (auto-detects newest in data/raw/ if omitted)")
    parser.add_argument("--nasa",   default=None,
                        help="NASA POWER CSV filename inside data/external/ (auto-detects if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Site      : {cfg['site']['name']}")
    logger.info(f"Location  : {cfg['site']['latitude']}° N, {cfg['site']['longitude']}° E")

    # ── Load local data ───────────────────────────────────────────────────────
    local_path = Path(args.local) if args.local else _auto_detect_local(cfg)
    logger.info(f"Local CSV : {local_path}")
    local_df = load_local_data(local_path, cfg)

    # ── Load NASA data ────────────────────────────────────────────────────────
    logger.info("Loading NASA POWER data …")
    nasa_df = nasa_power.load_raw(cfg, filename=args.nasa)

    # ── Align ─────────────────────────────────────────────────────────────────
    local_hourly, nasa_aligned = align_datasets(local_df, nasa_df, cfg)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_aligned(local_hourly, nasa_aligned, cfg)

    # ── Plot ──────────────────────────────────────────────────────────────────
    logger.info("Generating alignment plots …")
    plot_alignment(local_df, local_hourly, nasa_aligned, cfg)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n── Local hourly (first 3 rows) ───────────────────────────────────────")
    print(local_hourly.head(3).to_string())
    print(f"\nShape  : {local_hourly.shape}")
    print(f"Columns: {list(local_hourly.columns)}")
    print(f"Range  : {local_hourly.index.min()}  →  {local_hourly.index.max()}")

    print("\n── NASA aligned (first 3 rows) ───────────────────────────────────────")
    print(nasa_aligned.head(3).to_string())
    print(f"\nShape  : {nasa_aligned.shape}")
    print(f"Columns: {list(nasa_aligned.columns)}")
    print(f"Range  : {nasa_aligned.index.min()}  →  {nasa_aligned.index.max()}")


if __name__ == "__main__":
    main()
