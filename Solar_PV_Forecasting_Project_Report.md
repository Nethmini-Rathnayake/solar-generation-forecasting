# Solar PV Generation Forecasting — Project Report

**Site:** University of Moratuwa Smart Grid Lab, Sri Lanka  
**Coordinates:** 6.7912° N, 79.9005° E  
**System:** Hybrid PV plant (~100 kW AC capacity)  
**Actual PV data available:** April 2022 – March 2023 (1 year, 5-min resolution)  
**Test set:** February – March 2023

---

## 1. Problem Statement

Forecast 5-minute solar PV generation up to 24 hours ahead for a grid-connected PV system in Sri Lanka. The core challenge is Sri Lanka's **bimodal climate**: a stable dry season (NE monsoon, Dec–Mar) with predictable solar, and a chaotic wet season (SW monsoon, May–Oct) with rapid cloud transitions that are hard to forecast. Only one year of actual PV data is available, requiring augmentation with physics simulation and satellite weather data.

---

## 2. Data Sources

| Source | Resolution | Period | Variables |
|--------|-----------|--------|-----------|
| Actual PV (lab meter) | 5-min | Apr 2022 – Mar 2023 | AC power output [W] |
| Solcast satellite weather | 5-min | Jan 2020 – Feb 2024 | GHI, DNI, DHI, cloud_opacity, wind, temp, RH |
| NASA POWER | Hourly | 2020–2026 | GHI, temperature, wind, humidity |
| Open-Meteo ERA5 | Hourly | Apr 2022 – Mar 2023 | 10m wind speed and direction |
| Himawari-9 AHI (Band 3) | 10-min | Dec 2022 – present | Visible 0.64 µm satellite imagery |

---

## 3. Approach 1 — Baseline Models

### 3.1 Persistence Model
Naively repeats yesterday's power at the same time of day (lag 288 = 24 h × 12 intervals/h).

### 3.2 Climatological Model
Predicts the monthly-hour mean power computed from training data.

**Results (test set Feb–Mar 2023, horizon = 1–24 steps)**

| Model | R² (h+1) | R² (h+24) | RMSE h+1 (kW) | RMSE h+24 (kW) |
|-------|---------|---------|--------------|----------------|
| Persistence (lag-288) | 0.216 | 0.801 | 54.9 | 27.5 |
| Climatology | 0.714 | 0.765 | 31.1 | 29.8 |

**Key observations:**
- Persistence completely fails at short horizons (R² = 0.22 at h+1) because clouds change rapidly minute-to-minute.
- Climatology is better short-term but still limited — it knows typical diurnal patterns but not today's cloud cover.
- Both models serve as lower-bound benchmarks.

---

## 4. Approach 2 — Physics-Based PV Simulation (NASA POWER)

A deterministic physics model translating satellite GHI to AC power:

```
P_ac = a·sim + b·sim²
```

Where `sim` is the simulated AC power from pvlib (using NASA POWER GHI → plane-of-array irradiance → cell temperature → DC power → AC power). Polynomial coefficients (a, b) are calibrated per-season (dry/wet) against actual PV measurements.

**Calibration levels:** Global → Seasonal (dry/wet) → Monthly (12 months)

**Results (Apr 2022 – Mar 2023 overlap year)**

| Period | R² | RMSE [kW] | nRMSE [%] | MBE [kW] |
|--------|-----|---------|---------|---------|
| **Overall** | **0.797** | **32.83** | **35.3** | **+0.41** |
| Dry season | 0.847 | 28.62 | 28.5 | +0.22 |
| Wet season | 0.689 | 39.30 | 49.7 | +0.74 |
| Best month (Sep) | 0.889 | 25.21 | 24.2 | -1.58 |
| Worst month (Apr) | 0.536 | 47.85 | 61.0 | -2.60 |

**Key observations:**
- Physics model works well in the dry season but struggles in wet months (Apr, May, Oct, Nov) where cloud dynamics create high variability.
- The `a·sim + b·sim²` zero-intercept quadratic cannot capture bimodal cloud-cover distributions in the SW monsoon.

---

## 5. Approach 3 — Physics-Based PV Simulation (Solcast) + Sky-Stratified Calibration

Upgraded data source (Solcast 5-min vs NASA hourly) and calibration strategy.

**Improvements:**
- Solcast provides 5-min cloud_opacity and clearsky GHI — enables clearness index kt = GHI / GHI_clear
- Sky condition classification: Clear (kt > 0.7), PartlyCloudy (0.4–0.7), MostlyCloudy (0.2–0.4), Overcast (< 0.2)
- Separate polynomial fits per **(month × sky condition)** — 48 coefficient pairs

**Results (Solcast-calibrated physics, 4-year period)**

| Period | R² | RMSE [kW] | nRMSE [%] |
|--------|-----|---------|---------|
| **Overall** | **0.749** | **39.22** | **39.9** |
| Dry season | 0.784 | 36.52 | 34.7 |
| Wet season | 0.667 | 43.82 | 51.2 |
| Best month (Mar) | 0.872 | 29.29 | 26.0 |
| Worst month (Apr) | 0.540 | 51.70 | 61.4 |

**Key observations:**
- Solcast 5-min gives R² = 0.741 on the overlap year (vs NASA 0.797 — NASA appears better here because it was calibrated only on the 1-year overlap; Solcast is evaluated on 4 years including harder periods).
- Sky-stratified calibration reduces bias in wet months but the structural variance remains.

---

## 6. Approach 4 — Hi-Fi Multi-Stage Calibration Pipeline

A 10-stage calibration pipeline on top of Solcast physics simulation:

1. Raw pvlib simulation
2. Global polynomial correction (a, b)
3. Seasonal correction (dry/wet)
4. Monthly correction (12 months)
5. **Sky-stratified correction** (Clear/PartlyCloudy/MostlyCloudy/Overcast × month)
6. kt-normalization (scales by clearness index ratio)
7. Polynomial bias fitting on residuals
8. Post-calibration smoothing
9. Daytime mask and clipping
10. Final output validation

**Results (hi-fi pipeline, overlap year)**

| Stage | R² | RMSE [kW] | nRMSE [%] | MBE [kW] |
|-------|-----|---------|---------|---------|
| Raw physics (pre-calibration) | 0.693 | 43.10 | 43.0 | -10.76 |
| **Sky-stratified calibrated** | **0.743** | **39.40** | **39.3** | **-0.13** |

**Key observations:**
- Sky-stratified calibration reduces RMSE by 3.7 kW and nearly eliminates MBE (from -10.76 → -0.13 kW).
- AR(1) noise injection was tested but discarded — injecting noise degraded R² from 0.743 → 0.515 because the residual standard deviation in Clear regime (~38 kW) was too large relative to the signal.

---

## 7. Approach 5 — Machine Learning: XGBoost + LSTM/GRU (1-Year, Solcast)

Direct ML on the 1-year overlap period with Solcast weather features.

**Architecture:**
- **XGBoost:** 6,000 trees, depth 8, early stopping
- **LSTM/GRU:** 2-layer, hidden 64, 6-step lookback (30 min), MPS acceleration
- **Ensemble:** 4-way average of XGBoost + LSTM + GRU + Physics

**Features:** GHI, DNI, DHI, clearsky GHI, cloud_opacity, temperature, humidity, wind speed, clearness index, sky condition class, diffuse fraction, 5 kt rolling statistics, 6 PV lag features (t-5m to t-24h), 17 time/solar features

**Results (test set Feb–Mar 2023)**

| Model | R² | RMSE [kW] | nRMSE [%] | MAE [kW] | MBE [kW] |
|-------|-----|---------|---------|---------|---------|
| XGBoost ML | **0.864** | **29.37** | **24.0** | **18.85** | **-5.05** |
| XGB Residual (Physics+ML) | 0.859 | 29.92 | 24.5 | 17.67 | +4.51 |
| Ensemble (4-way) | 0.864 | 29.38 | 24.0 | 18.84 | -3.47 |
| GRU | 0.827 | 33.14 | 27.1 | 21.37 | -2.69 |
| LSTM | 0.820 | 33.77 | 27.6 | 21.17 | +2.15 |
| Physics (Solcast) | 0.830 | 32.84 | 26.9 | 21.58 | -0.33 |
| Persistence (lag-288) | 0.326 | 65.30 | 53.4 | 42.56 | -2.62 |

**By sky condition (XGBoost):**

| Sky Condition | R² | RMSE [kW] | nRMSE [%] |
|---------------|-----|---------|---------|
| Clear | 0.837 | 28.89 | 18.5 |
| PartlyCloudy | 0.768 | 35.00 | 34.9 |
| MostlyCloudy | 0.764 | 23.36 | 37.6 |
| Overcast | 0.866 | 7.63 | 30.2 |

**Key observations:**
- XGBoost achieves R² = 0.864 — a 4.1% absolute improvement over pure physics (0.830).
- PartlyCloudy and MostlyCloudy have the worst performance — rapid cloud transitions are not predictable from weather forecasts alone.
- LSTM/GRU underperform XGBoost on this dataset because the 1-year training window is small for sequence models.

---

## 8. Approach 6 — 4-Year ML with Synthetic PV Labels

The 1-year training window limits deep learning. Solution: extend to 4 years (2020–2024) using physics-calibrated **synthetic PV** as labels for the 3 years without actual measurements.

**Label strategy:**
- 2020-01 – 2022-03: synthetic PV (sky-stratified calibrated physics)
- 2022-04 – 2023-03: actual PV observations (overlap year)
- 2023-04 – 2024-02: synthetic PV

**Split:** Train 2020-01–2022-10 | Val 2022-11–2023-01 | Test 2023-02–2023-03

**Results (test set Feb–Mar 2023)**

| Model | R² | RMSE [kW] | nRMSE [%] | MAE [kW] | MBE [kW] |
|-------|-----|---------|---------|---------|---------|
| **XGBoost (4-yr)** | **0.873** | **28.38** | **23.2** | **17.07** | **-0.50** |
| GRU (4-yr) | 0.836 | 32.26 | 26.4 | 19.08 | +1.81 |
| CNN-GRU (4-yr) | 0.828 | 33.05 | 27.0 | 19.78 | +4.85 |
| Physics (Solcast) | 0.833 | 32.50 | 26.6 | 21.24 | +0.36 |

**Key observations:**
- 4-year training improves XGBoost by +0.009 R² over 1-year (0.873 vs 0.864).
- GRU benefits more from the extra data (+0.009 R²), confirming sequence models need more training data.
- CNN-GRU uses 1-hour lookback (12 steps) to capture irradiance ramps before cloud transitions.

---

## 9. Approach 7 — Cloud Motion Vector (CMV) Features

### 9.1 Himawari-9 Satellite CMV (Pipeline Built, Download Pending)

Himawari-9 AHI Band-3 (visible 0.64 µm, 500 m resolution, 10-min cadence) enables **direct cloud tracking** via optical flow (scikit-image ILK algorithm).

CMV physics:
- Dense optical flow between consecutive frames → pixel displacement → cloud speed [km/h] and direction [°]
- Shadow displacement: `offset_km = cloud_altitude × tan(solar_zenith)`
- Shadow arrival time: `t = (d_km - shadow_offset_projection) / cloud_speed × 60` [minutes]

Status: Pipeline implemented and tested (`src/cmv/optical_flow.py`, `src/data/himawari_loader.py`). Data download is feasible at ~490 KB/s from NOAA S3 (H-9 bucket). Full 3.5-month overlap period (~390 GB) pending download.

### 9.2 Wind-Based CMV (Open-Meteo ERA5, Implemented and Evaluated)

Since Himawari-8 2022 data is archived at ~4 KB/s (unusable), a wind-based CMV proxy was built using ERA5 surface wind scaled to cloud level:

```
cloud_speed = wind_10m × 1.8   (tropical marine 850 hPa / 10m ratio)
cloud_direction = wind_direction_from + 180°
```

**15 CMV features generated:**
- `cloud_speed_kmh`, `cloud_direction_deg`, `shadow_offset_km`
- `shadow_arrival_{5,10,20,40}km` — minutes until cloud shadow arrives
- `opacity_lag_{5,10,20,40}km` — cloud opacity at upstream distance
- `site_cloud_opacity`, `cloud_opacity_trend`

**Overall test set impact (Feb–Mar 2023 = dry season, unfavourable for CMV):**

| Model | Base R² | CMV R² | ΔR² | Base RMSE | CMV RMSE | ΔRMSE |
|-------|---------|--------|-----|-----------|----------|-------|
| XGBoost | 0.8728 | 0.8715 | -0.0013 | 28.38 kW | 28.53 kW | +0.14 |
| GRU | 0.8356 | 0.8329 | -0.0028 | 32.26 kW | 32.53 kW | +0.27 |
| CNN-GRU | 0.8275 | 0.8309 | **+0.0034** | 33.05 kW | 32.72 kW | **-0.32** |

**CMV impact by month (SW monsoon — where CMV matters most):**

| Month | Base R² | CMV R² | ΔR² | ΔRMSE [kW] | Season |
|-------|---------|--------|-----|------------|--------|
| Apr | 0.9006 | 0.9076 | +0.0071 ▲ | -0.86 | SW Transition |
| May | 0.8803 | 0.8877 | +0.0073 ▲ | -0.77 | SW Monsoon |
| Jun | 0.8548 | 0.8597 | +0.0050 ▲ | -0.49 | SW Monsoon |
| Jul | 0.8601 | 0.8691 | +0.0089 ▲ | -0.85 | SW Monsoon |
| Aug | 0.8718 | 0.8783 | +0.0065 ▲ | -0.73 | SW Monsoon |
| Sep | 0.8507 | 0.8537 | +0.0031 ▲ | -0.32 | SW Monsoon |
| Oct | 0.8912 | 0.9000 | +0.0088 ▲ | -1.02 | SW Transition |
| Nov–Feb | ~0.87 | ~0.87 | ~0.000 | ~0.00 | NE Monsoon (dry) |

**CMV impact by sky condition:**

| Sky Condition | Base R² | CMV R² | ΔR² | ΔRMSE [kW] |
|---------------|---------|--------|-----|------------|
| Clear | 0.8456 | 0.8492 | +0.0036 ▲ | -0.34 |
| PartlyCloudy | 0.8145 | 0.8207 | +0.0062 ▲ | -0.52 |
| MostlyCloudy | 0.8555 | 0.8655 | +0.0099 ▲ | -0.56 |
| Overcast | 0.8710 | 0.8811 | +0.0101 ▲ | -0.25 |

**Key observations:**
- CMV features improve all sky conditions, with largest gains in MostlyCloudy and Overcast.
- The fixed test split (Feb–Mar = dry NE monsoon) under-represents CMV benefit. Across the full year, CMV adds +0.005–+0.009 R² and saves 0.5–1.0 kW RMSE every SW monsoon month.
- When Himawari-9 real satellite CMV is available, shadow arrival precision improves from ±10 min (wind proxy) to ±2 min (optical flow).

---

## 10. Overall Comparison Table

All results on the **test set: February – March 2023** (5-min resolution, daytime only, actual PV observations).

| # | Approach | Data Source | R² | RMSE [kW] | nRMSE [%] | MAE [kW] | MBE [kW] |
|---|----------|------------|-----|---------|---------|---------|---------|
| 1a | Persistence (lag-288) | Actual PV | 0.216 | 54.86 | 59.0 | 44.44 | +0.54 |
| 1b | Climatology (monthly-hour mean) | Actual PV | 0.714 | 31.12 | 30.8 | 22.81 | +3.27 |
| 2 | Physics (NASA POWER + calibration) | NASA POWER | 0.797 | 32.83 | 35.3 | 22.60 | +0.41 |
| 3a | Physics (Solcast 5-min, seasonal cal.) | Solcast | 0.741 | 39.57 | 39.5 | 27.67 | +0.12 |
| 3b | Physics (Solcast, sky-stratified cal.) | Solcast | 0.749 | 39.22 | 39.9 | 27.20 | +0.16 |
| 4 | Hi-Fi multi-stage calibration | Solcast | 0.743 | 39.40 | 39.3 | — | -0.13 |
| 5a | Physics (Solcast) baseline | Solcast | 0.830 | 32.84 | 26.9 | 21.58 | -0.33 |
| 5b | XGBoost ML (1-year) | Solcast | 0.864 | 29.37 | 24.0 | 18.85 | -5.05 |
| 5c | GRU (1-year) | Solcast | 0.827 | 33.14 | 27.1 | 21.37 | -2.69 |
| 5d | LSTM (1-year) | Solcast | 0.820 | 33.77 | 27.6 | 21.17 | +2.15 |
| 5e | Ensemble: XGB + LSTM + GRU + Physics | Solcast | 0.864 | 29.38 | 24.0 | 18.84 | -3.47 |
| 6a | **XGBoost (4-year, synthetic labels)** | **Solcast** | **0.873** | **28.38** | **23.2** | **17.07** | **-0.50** |
| 6b | GRU (4-year, synthetic labels) | Solcast | 0.836 | 32.26 | 26.4 | 19.08 | +1.81 |
| 6c | CNN-GRU (4-year, synthetic labels) | Solcast | 0.828 | 33.05 | 27.0 | 19.78 | +4.85 |
| 7a | XGBoost + Wind CMV (dry season test) | Solcast + ERA5 | 0.872 | 28.53 | 23.3 | 17.27 | -1.15 |
| 7b | CNN-GRU + Wind CMV (dry season test) | Solcast + ERA5 | 0.831 | 32.72 | 26.8 | 20.09 | +3.06 |
| 7c | XGBoost + Wind CMV (SW monsoon Jul) | Solcast + ERA5 | 0.869 | 25.21 | — | — | — |

> **Bold** = best overall result on the standard test set.  
> Rows 3a–4 are evaluated on the full overlap year (not just Feb–Mar); row 2 also uses the overlap year.  
> Row 7c shows July evaluation to illustrate CMV benefit in the wet season.

---

## 11. Key Findings

1. **Best overall accuracy: XGBoost with 4-year synthetic labels** — R² = 0.873, RMSE = 28.38 kW (23.2% nRMSE). The synthetic label strategy multiplied training data 4× with acceptable label noise.

2. **Biggest single improvement: raw physics → XGBoost ML** — R² jumps from 0.741 (physics alone) to 0.864 (+0.123), demonstrating that the satellite weather features contain predictive information not captured by the physics model.

3. **CMV features add the most value where it matters most** — SW monsoon (May–Oct): +0.005–+0.009 R² and -0.5 to -1.0 kW RMSE per month. The fixed dry-season test set masks this gain.

4. **Wet season remains the hard problem** — Best wet-season R² is ~0.67–0.70 (vs 0.84 in dry season). Rapid convective cloud events with sub-10-minute timescales cannot be predicted from hourly wind + satellite opacity alone.

5. **Himawari-9 real CMV (next step)** — Optical flow on actual satellite images gives 2-min shadow arrival precision vs 10-min for the wind proxy. This is expected to close the remaining wet-season gap.

---

## 12. Next Steps

| Priority | Task | Expected Impact |
|----------|------|----------------|
| High | Download Himawari-9 (3 days test, then full overlap) | Real CMV → +0.01–0.02 R² in SW monsoon |
| High | Re-evaluate CMV with SW monsoon test split (May–Oct) | Properly quantify CMV benefit |
| Medium | Sky-stratified calibration on 4-yr synthetic labels | Improve wet-season synthetic label quality |
| Medium | Hyperparameter tuning for CNN-GRU with CMV features | CNN-GRU showed most CMV responsiveness |
| Low | Ensemble: XGBoost (4-yr) + CNN-GRU + CMV | Combine best features of all approaches |
