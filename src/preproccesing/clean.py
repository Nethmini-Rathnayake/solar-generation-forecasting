"""
src/preproccesing/clean.py
---------------------------
Identifies and flags system unavailability hours in the synthetic PV dataset.

Problem
-------
The physics simulation (pvlib PVWatts) produces power output whenever
NASA/ERA5 GHI > 0. But the real system is sometimes unavailable:
  - Grid outages / load-shedding
  - Inverter trips or faults
  - Scheduled maintenance
  - Communication failures

These hours appear as:  sim_ac >> 0  AND  obs_ac ≈ 0
They are NOT physics errors — the model is correct; the plant was just off.

Including them in training teaches the forecasting model:
  "sometimes at high GHI, output is zero"  ← wrong generalisation

Why this matters for the forecasting model
------------------------------------------
The XGBoost model will see hundreds of daytime hours where GHI is high
but target power is zero.  It will learn a "sometimes zero" rule that
makes it systematically under-predict on clear days.  Removing these
hours from training (not from inference) prevents this.

Detection method
----------------
During the 1-year overlap window where both sim and obs are available:
  unavailable(t) = True  iff  sim(t) > SIM_MIN_W  AND  obs(t) < OBS_MAX_W

Outside the overlap, unavailability cannot be detected and is set to False
(assume plant available — conservative, no false positives).

Thresholds chosen:
  SIM_MIN_W  = 10,000 W  (~4% of peak) — clearly a daytime hour by physics
  OBS_MAX_W  =  1,000 W  (~0.4% of peak) — system effectively off

Usage
-----
    from src.preproccesing.clean import flag_unavailable_hours

    unavail_mask = flag_unavailable_hours(pv_sim, local_hourly)
    # returns pd.Series[bool] indexed like pv_sim; True = unavailable
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Detection thresholds
_SIM_MIN_W: float = 10_000.0   # sim must exceed this to count as "daytime"
_OBS_MAX_W: float =  1_000.0   # obs must be below this to count as "off"

_PV_OBS_COL = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"


def flag_unavailable_hours(
    pv_sim:       pd.DataFrame,
    local_hourly: pd.DataFrame,
    sim_min_w:    float = _SIM_MIN_W,
    obs_max_w:    float = _OBS_MAX_W,
) -> pd.Series:
    """
    Return a boolean Series marking system-unavailability hours.

    True  = unavailable  (exclude from model training)
    False = available    (include normally)

    Detection only applies within the overlap window where obs is known.
    All hours outside the overlap default to False (available).

    Parameters
    ----------
    pv_sim : pd.DataFrame
        Full 6-year synthetic PV DataFrame (must contain 'pv_ac_W').
    local_hourly : pd.DataFrame
        1-year local measurements aligned to UTC (must contain _PV_OBS_COL).
    sim_min_w : float
        Minimum simulated power (W) to consider an hour "daytime".
    obs_max_w : float
        Maximum observed power (W) to classify as "system off".

    Returns
    -------
    pd.Series[bool]
        Same index as pv_sim. True = unavailable hour.
    """
    # Start with all False (available)
    mask = pd.Series(False, index=pv_sim.index, name="unavailable")

    if _PV_OBS_COL not in local_hourly.columns:
        logger.warning(
            f"  '{_PV_OBS_COL}' not found in local_hourly — "
            "unavailability masking skipped."
        )
        return mask

    obs = local_hourly[_PV_OBS_COL]
    sim = pv_sim["pv_ac_W"]

    # Align on common timestamps (overlap window only)
    common_idx = sim.index.intersection(obs.index)
    sim_ov     = sim.loc[common_idx]
    obs_ov     = obs.loc[common_idx]

    unavail_idx = common_idx[
        (sim_ov.values > sim_min_w) & (obs_ov.values < obs_max_w)
    ]
    mask.loc[unavail_idx] = True

    n = int(mask.sum())
    logger.info(
        f"  Unavailability flags: {n} hours  "
        f"({100 * n / len(mask):.2f}% of full dataset, "
        f"{100 * n / len(common_idx):.1f}% of overlap window)"
    )
    logger.info(
        f"  Detection: sim > {sim_min_w/1000:.0f} kW AND obs < {obs_max_w/1000:.1f} kW"
    )
    if n > 0:
        logger.info(
            f"  Month breakdown:\n"
            + "\n".join(
                f"    {m:>2}: {cnt} hrs"
                for m, cnt in mask[mask].groupby(mask[mask].index.month).size().items()
            )
        )

    return mask
