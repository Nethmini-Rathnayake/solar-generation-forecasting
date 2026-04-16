"""
src/evaluation/report.py
-------------------------
Generate a concise markdown evaluation report from saved metrics CSVs.

Usage
-----
    from src.evaluation.report import generate_report
    generate_report(metrics_dir, out_path)
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_report(metrics_dir: Path, out_path: Path | None = None) -> str:
    """
    Load metrics CSVs and write a markdown summary report.

    Parameters
    ----------
    metrics_dir : Path  directory containing metrics_xgboost.csv etc.
    out_path    : Path  where to write the .md file (optional)

    Returns
    -------
    str  report text
    """
    metrics_dir = Path(metrics_dir)

    def _load(name: str) -> pd.DataFrame | None:
        p = metrics_dir / f"metrics_{name}.csv"
        if p.exists():
            return pd.read_csv(p, index_col="horizon")
        return None

    xgb  = _load("xgboost")
    pers = _load("persistence")
    clim = _load("climatology")

    lines = [
        "# 24h-Ahead PV Forecasting — Evaluation Report",
        "",
        "## Model",
        "**XGBoost Direct Multi-Step (DMS)** — 24 independent models, one per horizon h=1…24.",
        "",
        "## Test-Set Performance",
        "",
    ]

    if xgb is not None:
        lines += [
            "### RMSE by Horizon (kW)",
            "",
            "| h | XGBoost | Persistence | Climatology | Skill vs Pers |",
            "|---|---------|-------------|-------------|---------------|",
        ]
        for h in [1, 3, 6, 12, 18, 24]:
            if h not in xgb.index:
                continue
            xv = xgb.loc[h, "RMSE_W"] / 1000
            pv = pers.loc[h, "RMSE_W"] / 1000 if pers is not None and h in pers.index else None
            cv = clim.loc[h, "RMSE_W"] / 1000 if clim is not None and h in clim.index else None
            skill = f"{(pv-xv)/pv*100:+.1f}%" if pv else "—"
            pv_s = f"{pv:.2f}" if pv else "—"
            cv_s = f"{cv:.2f}" if cv else "—"
            lines.append(f"| {h} | {xv:.2f} | {pv_s} | {cv_s} | {skill} |")

        lines += [
            "",
            f"**Mean R² (h=1..24):** {xgb['R2'].mean():.4f}",
            f"  \n**Mean nRMSE (h=1..24):** {xgb['nRMSE_pct'].mean():.2f}%",
            "",
        ]

    lines += [
        "## Figures",
        "- `metrics_vs_horizon.png` — RMSE / MAE / nRMSE / R² vs h",
        "- `scatter_selected_horizons.png` — observed vs predicted at h=1,6,12,24",
        "- `feature_importance.png` — top-20 XGBoost features by mean gain",
        "- `error_by_hour.png` — systematic bias by UTC hour",
        "- `forecast_sample_days.png` — 24h forecast trace on 4 sample days",
        "",
        "## Data",
        "- Training data: 70% of feature matrix (chronological split)",
        "- Validation: 15% (used for XGBoost early stopping)",
        "- Test: 15% (never seen during training)",
        "",
    ]

    report = "\n".join(lines)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        logger.info(f"Report saved → {out_path}")

    return report
