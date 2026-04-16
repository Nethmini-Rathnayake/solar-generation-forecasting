"""
src/calibration/bias_correction.py
-----------------------------------
Fits and applies linear bias-correction models for meteorological variables
by comparing 1-year local measurements against NASA POWER during the
alignment overlap window.

Correction equation fitted for each pair
-----------------------------------------
    local_value = slope × (nasa_value × unit_scale) + intercept

When R² < R2_SLOPE_THRESHOLD, the slope is forced to 1.0 (bias-only).
A fitted slope << 1 at low R² would incorrectly collapse the signal
variance toward the mean — an additive offset is more defensible.

Variables fitted
----------------
  T2M   → local tempC          (both °C,  scale = 1.0)
  RH2M  → local humidity       (both %,   scale = 1.0)
  WS10M → local windspeedKmph  (scale = 3.6: m/s → km/h before fitting)

When applying corrections, results are returned in original NASA units:
  T2M_cal  = slope × T2M  + intercept         [°C]
  RH2M_cal = slope × RH2M + intercept         [%]
  WS10M_cal = slope × WS10M + intercept/3.6   [m/s]

Usage
-----
    from src.calibration.bias_correction import fit_met_corrections, apply_met_corrections

    corrections = fit_met_corrections(local_hourly, nasa_aligned)
    nasa_cal    = apply_met_corrections(nasa_full, corrections)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum R² for a slope correction to be trusted.
# Below this, slope is forced to 1.0 (bias-only offset).
_R2_SLOPE_THRESHOLD: float = 0.30

# (local_col, nasa_col, unit_scale, unit_label)
# unit_scale converts nasa_col → local_col units before regression.
_MET_PAIRS: list[tuple[str, str, float, str]] = [
    ("tempC",         "T2M",   1.0, "°C"),
    ("humidity",      "RH2M",  1.0, "%"),
    ("windspeedKmph", "WS10M", 3.6, "km/h"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinearCorrection:
    """
    Linear bias-correction model for one variable pair.

    The regression is fitted in local_col units:
        local = slope × (nasa × unit_scale) + intercept

    Applying returns the corrected value in the original NASA units:
        corrected_nasa = slope × nasa + intercept / unit_scale
    """
    local_col:   str
    nasa_col:    str
    slope:       float        # fitted on local units
    intercept:   float        # fitted on local units
    r2:          float
    mae:         float        # MAE in local units
    n:           int
    unit_scale:  float        # nasa × unit_scale = local units
    unit_label:  str
    bias_only:   bool = field(default=False)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def apply(self, nasa_series: pd.Series) -> pd.Series:
        """
        Correct a NASA series; returns result in original NASA units.

        For WS10M (unit_scale=3.6):
            step 1: m/s → km/h  (× 3.6)
            step 2: linear correction in km/h
            step 3: km/h → m/s  (÷ 3.6)
        """
        x_local    = nasa_series * self.unit_scale
        y_local    = self.slope * x_local + self.intercept
        return y_local / self.unit_scale

    def correction_in_nasa_units(self) -> tuple[float, float]:
        """
        Return (slope, intercept) of the correction in original NASA units.

        i.e., corrected_nasa = eff_slope × nasa + eff_intercept
        """
        eff_slope     = self.slope
        eff_intercept = self.intercept / self.unit_scale
        return eff_slope, eff_intercept

    def equation_str(self, nasa_units: bool = False) -> str:
        """Human-readable correction equation."""
        if nasa_units:
            s, b = self.correction_in_nasa_units()
            sign = "+" if b >= 0 else "−"
            tag  = " [bias-only]" if self.bias_only else ""
            return (
                f"{self.nasa_col}_cal = {s:.3f}·{self.nasa_col} {sign} {abs(b):.3f}"
                f"  [{self.unit_label}]  R²={self.r2:.3f}{tag}"
            )
        else:
            sign = "+" if self.intercept >= 0 else "−"
            tag  = " [bias-only]" if self.bias_only else ""
            return (
                f"{self.local_col} = {self.slope:.3f}·x {sign} {abs(self.intercept):.3f}"
                f"  R²={self.r2:.3f}  MAE={self.mae:.2f} {self.unit_label}  n={self.n:,}{tag}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fit_met_corrections(
    local_hourly: pd.DataFrame,
    nasa_aligned: pd.DataFrame,
) -> dict[str, LinearCorrection]:
    """
    Fit T2M, RH2M, and WS10M bias corrections from the 1-year overlap data.

    Parameters
    ----------
    local_hourly : pd.DataFrame
        Hourly UTC local data (output of align_datasets).
    nasa_aligned : pd.DataFrame
        Hourly UTC NASA POWER data, same time window.

    Returns
    -------
    dict[str, LinearCorrection]
        Keyed by nasa_col (e.g. "T2M", "RH2M", "WS10M").
    """
    logger.info("=" * 60)
    logger.info("FITTING METEOROLOGICAL BIAS CORRECTIONS")

    corrections: dict[str, LinearCorrection] = {}

    for local_col, nasa_col, scale, unit_label in _MET_PAIRS:
        if local_col not in local_hourly.columns:
            logger.warning(f"  '{local_col}' not in local data — skipping {nasa_col}")
            continue
        if nasa_col not in nasa_aligned.columns:
            logger.warning(f"  '{nasa_col}' not in NASA data — skipping")
            continue

        # ── Pair and convert to local units ───────────────────────────────────
        x_local = nasa_aligned[nasa_col] * scale
        y_local = local_hourly[local_col]
        df_pair = pd.concat(
            [x_local.rename("x"), y_local.rename("y")], axis=1
        ).dropna()

        if len(df_pair) < 10:
            logger.warning(f"  {nasa_col}: too few valid pairs ({len(df_pair)}) — skipping")
            continue

        x, y = df_pair["x"].values, df_pair["y"].values

        # ── Full linear regression ─────────────────────────────────────────────
        slope_full, intercept_full, r, _, _ = stats.linregress(x, y)
        r2 = float(r ** 2)

        # ── Choose: bias-only or full slope ───────────────────────────────────
        if r2 < _R2_SLOPE_THRESHOLD:
            slope     = 1.0
            intercept = float(np.mean(y - x))   # mean(local − nasa_in_local_units)
            bias_only = True
            logger.info(
                f"  {nasa_col}: R²={r2:.3f} < {_R2_SLOPE_THRESHOLD} "
                f"→ bias-only  (offset = {intercept:+.3f} {unit_label})"
            )
        else:
            slope     = float(slope_full)
            intercept = float(intercept_full)
            bias_only = False

        pred = slope * x + intercept
        mae  = float(np.mean(np.abs(pred - y)))

        corr = LinearCorrection(
            local_col=local_col,
            nasa_col=nasa_col,
            slope=slope,
            intercept=intercept,
            r2=r2,
            mae=mae,
            n=len(df_pair),
            unit_scale=scale,
            unit_label=unit_label,
            bias_only=bias_only,
        )
        corrections[nasa_col] = corr
        logger.info(f"  {nasa_col}: {corr.equation_str(nasa_units=True)}")

    logger.info("=" * 60)
    return corrections


def apply_met_corrections(
    nasa_df: pd.DataFrame,
    corrections: dict[str, LinearCorrection],
) -> pd.DataFrame:
    """
    Apply fitted corrections to a NASA POWER DataFrame (any time window).

    Corrected columns are added with suffix ``_cal``; originals are preserved.
    Physically impossible values are clipped (RH ∈ [0,100], WS ≥ 0).

    Parameters
    ----------
    nasa_df : pd.DataFrame
        NASA POWER data (full 6-year range or any subset).
    corrections : dict[str, LinearCorrection]
        From fit_met_corrections().

    Returns
    -------
    pd.DataFrame
        Original columns + ``*_cal`` corrected columns.
    """
    out = nasa_df.copy()

    for nasa_col, corr in corrections.items():
        if nasa_col not in out.columns:
            logger.warning(f"  '{nasa_col}' not in DataFrame — skipping apply")
            continue

        cal_col    = f"{nasa_col}_cal"
        out[cal_col] = corr.apply(out[nasa_col])

        if nasa_col == "RH2M":
            out[cal_col] = out[cal_col].clip(0.0, 100.0)
        elif nasa_col == "WS10M":
            out[cal_col] = out[cal_col].clip(lower=0.0)

        # Report delta in local units (multiply by scale; e.g. WS10M: m/s → km/h)
        delta_local = (out[cal_col] - out[nasa_col]).mean() * corr.unit_scale
        logger.info(
            f"  {nasa_col} → {cal_col}  "
            f"(Δmean = {delta_local:+.3f} {corr.unit_label}, n = {len(out):,})"
        )

    return out
