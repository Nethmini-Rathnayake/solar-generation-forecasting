"""
src/calibration/regression.py
-------------------------------
Derives GHI calibration factors using locally measured PV power as a proxy.

Since no on-site irradiance sensor exists, the PV system acts as a
"natural pyranometer": during daytime, measured PV power scales
approximately linearly with incident GHI.

Method
------
1. Filter to daytime: GHI ≥ GHI_MIN_DAYTIME and PV ≥ PV_MIN_DAYTIME.
2. Fit:  PV_power = slope × GHI + intercept   [W vs W/m²]
   using ordinary least squares (scipy.stats.linregress).
3. Compute the hourly prediction-to-actual ratio:
       ratio(t) = PV_actual(t) / PV_predicted(t)
4. Aggregate per calendar month → monthly correction factors, normalised
   so the annual (all-overlap) median = 1.0.
       factor_M = median(ratio for month M) / median(ratio for all months)
5. GHI columns are scaled: GHI_cal = GHI_nasa × factor_M

Interpretation of monthly factors
-----------------------------------
  factor > 1.0 : NASA underestimates effective irradiance that month
                 (e.g. cloud cover underestimated → GHI should be lower → ?)
                 Actually: ratio > 1 means PV > prediction → GHI_nasa is too
                 LOW to explain observed output → scale GHI UP.
  factor < 1.0 : NASA overestimates effective irradiance → scale GHI DOWN.

Caveats
-------
- Monthly factors absorb both NASA GHI bias AND seasonal PV performance
  variation (soiling, availability). Both are treated as a single factor.
- Only 12 monthly estimates from 1 year of data — approximate.
- Note: site.yaml lists capacity_kw = 10 kW. Empirical slope implies
  ~217 kW actual capacity. The site.yaml value appears incorrect.

Usage
-----
    from src.calibration.regression import fit_ghi_calibration, apply_ghi_calibration

    ghi_cal  = fit_ghi_calibration(local_hourly, nasa_aligned)
    nasa_cal = apply_ghi_calibration(nasa_df, ghi_cal)
"""

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GHI_MIN_DAYTIME: float = 50.0      # W/m² — below this GHI readings are noisy
PV_MIN_DAYTIME:  float = 100.0     # W    — below this treat as night or off
PV_POWER_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"

# All irradiance columns to scale with the same monthly factors.
# GHI, DNI, DHI and clear-sky GHI share the same cloud-cover parameterisation.
GHI_IRRAD_COLS: list[str] = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "CLRSKY_SFC_SW_DWN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GHICalibration:
    """
    GHI calibration results derived from PV-proxy regression.

    Attributes
    ----------
    slope : float
        PV response slope  [W per W/m²]  from  PV = slope × GHI + intercept.
        Empirically implies ~217 kW system at GHI = 1000 W/m².
    intercept : float
        PV response intercept [W].
    r2 : float
        R² of the daytime regression.
    n_daytime : int
        Daytime hours used for regression.
    monthly_factors : dict[int, float]
        {month (1–12): correction_factor}.
        factor = 1.0 → no change.  All values normalised to annual median = 1.0.
    """
    slope:           float
    intercept:       float
    r2:              float
    n_daytime:       int
    monthly_factors: dict[int, float]

    def equation_str(self) -> str:
        sign = "+" if self.intercept >= 0 else "−"
        return (
            f"PV = {self.slope:.2f} × GHI {sign} {abs(self.intercept):.0f} W"
            f"  R²={self.r2:.3f}  n={self.n_daytime:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fit_ghi_calibration(
    local_hourly: pd.DataFrame,
    nasa_aligned: pd.DataFrame,
) -> GHICalibration:
    """
    Fit GHI correction factors using measured PV power as a proxy.

    Parameters
    ----------
    local_hourly : pd.DataFrame
        Hourly UTC local data (output of align_datasets).
    nasa_aligned : pd.DataFrame
        Hourly UTC NASA POWER data, same time window.

    Returns
    -------
    GHICalibration
    """
    logger.info("=" * 60)
    logger.info("FITTING GHI CALIBRATION VIA PV PROXY")

    pv_col  = PV_POWER_COL
    ghi_col = "ALLSKY_SFC_SW_DWN"

    if pv_col not in local_hourly.columns:
        raise ValueError(f"PV column not found: '{pv_col}'")
    if ghi_col not in nasa_aligned.columns:
        raise ValueError(f"GHI column not found: '{ghi_col}'")

    # ── Build daytime DataFrame ───────────────────────────────────────────────
    df = pd.concat(
        [local_hourly[pv_col].rename("pv"),
         nasa_aligned[ghi_col].rename("ghi")],
        axis=1,
    ).dropna()

    mask_day = (df["ghi"] >= GHI_MIN_DAYTIME) & (df["pv"] >= PV_MIN_DAYTIME)
    df_day   = df.loc[mask_day].copy()

    logger.info(f"  Daytime hours (GHI ≥ {GHI_MIN_DAYTIME}, PV ≥ {PV_MIN_DAYTIME}): "
                f"{len(df_day):,} / {len(df):,}")

    if len(df_day) < 50:
        raise ValueError(
            f"Too few daytime hours ({len(df_day)}) for GHI calibration. "
            "Check that PV and GHI data overlap."
        )

    # ── Overall regression: PV = slope × GHI + intercept ─────────────────────
    slope, intercept, r, _, _ = stats.linregress(df_day["ghi"], df_day["pv"])
    r2 = float(r ** 2)

    implied_kw = (slope * 1000 + intercept) / 1000
    logger.info(f"  Regression: PV = {slope:.2f} × GHI + {intercept:.0f} W  (R²={r2:.3f})")
    logger.info(f"  Implied system capacity at GHI=1000: {implied_kw:.1f} kW")
    logger.info(f"  (site.yaml lists 10.0 kW — this appears to be underspecified)")

    # ── Monthly correction factors ────────────────────────────────────────────
    # ratio(t) = PV_actual / PV_predicted  — measures how well NASA GHI
    # explains observed output each hour.  > 1 → GHI under-reported by NASA.
    df_day = df_day.copy()
    df_day["pv_pred"] = slope * df_day["ghi"] + intercept
    df_day["ratio"]   = df_day["pv"] / df_day["pv_pred"]

    monthly_median = df_day.groupby(df_day.index.month)["ratio"].median()
    annual_median  = float(df_day["ratio"].median())
    monthly_factors = (monthly_median / annual_median).to_dict()

    logger.info(f"  Annual median ratio: {annual_median:.3f}")
    logger.info("  Monthly GHI correction factors (normalised to annual median):")
    for m in range(1, 13):
        f = monthly_factors.get(m, 1.0)
        symbol = "▲" if f > 1.05 else ("▼" if f < 0.95 else "≈")
        logger.info(f"    Month {m:2d}: {f:.3f} {symbol}")

    logger.info("=" * 60)

    return GHICalibration(
        slope=float(slope),
        intercept=float(intercept),
        r2=r2,
        n_daytime=len(df_day),
        monthly_factors={int(k): float(v) for k, v in monthly_factors.items()},
    )


def apply_ghi_calibration(
    nasa_df: pd.DataFrame,
    cal: GHICalibration,
) -> pd.DataFrame:
    """
    Apply monthly GHI correction factors to all irradiance columns.

    Months not present in cal.monthly_factors receive factor = 1.0.
    All calibrated values are clipped to ≥ 0 W/m².

    Parameters
    ----------
    nasa_df : pd.DataFrame
        NASA POWER data with a UTC DatetimeIndex (any time window).
    cal : GHICalibration
        From fit_ghi_calibration().

    Returns
    -------
    pd.DataFrame
        Original columns + ``*_cal`` columns for each irradiance variable.
    """
    out = nasa_df.copy()

    # Per-row factor series: map each timestamp's month to its correction factor
    month_factors = (
        pd.Series(out.index.month, index=out.index)
        .map(cal.monthly_factors)
        .fillna(1.0)
    )

    for col in GHI_IRRAD_COLS:
        if col not in out.columns:
            continue
        cal_col       = f"{col}_cal"
        out[cal_col]  = (out[col] * month_factors).clip(lower=0.0)
        delta         = (out[cal_col] - out[col]).mean()
        logger.info(
            f"  {col} → {cal_col}  "
            f"(mean monthly factor = {month_factors.mean():.3f}, Δmean = {delta:+.2f} W/m²)"
        )

    return out
