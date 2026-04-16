"""
src/calibration/apply.py
-------------------------
Top-level calibration orchestrator: loads the full 6-year NASA POWER
dataset, applies meteorological and GHI corrections, and saves the
result to data/processed/nasa_calibrated.csv.

Output columns
--------------
  Original columns retained (raw), plus calibrated counterparts:
    T2M_cal, RH2M_cal, WS10M_cal
    ALLSKY_SFC_SW_DWN_cal, ALLSKY_SFC_SW_DNI_cal,
    ALLSKY_SFC_SW_DIFF_cal, CLRSKY_SFC_SW_DWN_cal

Usage
-----
    from src.calibration.apply import calibrate_nasa

    nasa_cal = calibrate_nasa(cfg, corrections, ghi_cal)
"""

import pandas as pd

from src.data import nasa_power
from src.calibration.bias_correction import LinearCorrection, apply_met_corrections
from src.calibration.regression      import GHICalibration, apply_ghi_calibration
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calibrate_nasa(
    cfg:              dict,
    corrections:      dict[str, LinearCorrection],
    ghi_cal:          GHICalibration,
    nasa_filename:    str | None = None,
) -> pd.DataFrame:
    """
    Load the full NASA POWER dataset, apply all calibrations, and save.

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().
    corrections : dict[str, LinearCorrection]
        Meteorological corrections from fit_met_corrections().
    ghi_cal : GHICalibration
        GHI correction from fit_ghi_calibration().
    nasa_filename : str, optional
        Filename within data/external/ (auto-detects newest if None).

    Returns
    -------
    pd.DataFrame
        Calibrated NASA data, full 6-year period.
    """
    logger.info("=" * 60)
    logger.info("APPLYING CALIBRATIONS TO FULL NASA DATASET")

    # ── Load full NASA data ───────────────────────────────────────────────────
    logger.info("Loading full NASA POWER data …")
    nasa_full = nasa_power.load_raw(cfg, filename=nasa_filename)
    logger.info(f"  Input : {len(nasa_full):,} rows × {nasa_full.shape[1]} cols")
    logger.info(f"  Range : {nasa_full.index.min()}  →  {nasa_full.index.max()}")

    # ── Meteorological corrections ────────────────────────────────────────────
    logger.info("Applying met corrections (T2M, RH2M, WS10M) …")
    nasa_cal = apply_met_corrections(nasa_full, corrections)

    # ── GHI corrections ───────────────────────────────────────────────────────
    logger.info("Applying GHI calibration (monthly factors) …")
    nasa_cal = apply_ghi_calibration(nasa_cal, ghi_cal)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = resolve_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nasa_calibrated.csv"
    nasa_cal.to_csv(out_path)

    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved → {out_path}  ({size_kb:.1f} KB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    cal_cols = [c for c in nasa_cal.columns if c.endswith("_cal")]
    logger.info("-" * 60)
    logger.info(f"Output : {len(nasa_cal):,} rows × {nasa_cal.shape[1]} cols")
    logger.info(f"New _cal columns : {cal_cols}")
    logger.info("=" * 60)

    return nasa_cal
