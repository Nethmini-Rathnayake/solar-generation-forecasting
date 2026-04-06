# ⚡ Cost-Optimized Microgrid Generation Forecasting

---

## 🧭 Introduction

This project focuses on developing a **solar power generation forecasting model** as a core component of a **cost-optimized microgrid controller**.

Accurate generation forecasting is critical for:
- ⚡ Optimal energy dispatch  
- 💰 Cost minimization  
- 🔌 Reliable microgrid operation  

However, a key challenge in this project is **data scarcity**.  
The target microgrid system is still under development and does not yet have a long-term data logging infrastructure.

Currently, only **~1 year of local data** is available, which is insufficient for training robust forecasting models that capture:
- Seasonal variations  
- Weather-driven fluctuations  
- Monsoon effects (specific to Sri Lanka)

To address this limitation, the project adopts a **hybrid approach**:
> 📡 Satellite data + 🧪 calibration + ⚙️ physics-based modeling

---

## 🧪 Data Generation & Calibration Strategy

To overcome the lack of long-term local data, this project constructs a **pseudo-historical dataset** using satellite/reanalysis data, calibration techniques, and physical PV modeling.

---

### 📡 1. Satellite & Reanalysis Data Collection

Multi-year environmental data is collected from:

- **NASA POWER API**
  - Solar irradiance  
  - Temperature  
  - Humidity  
  - Other weather variables  

These datasets provide **long-term, consistent inputs** required for solar generation modeling.

---

### 🧩 2. Site-Specific Calibration

Satellite-derived data does not perfectly match local site conditions.  
Therefore, a **calibration step** is applied using the available real dataset.

**Process:**
- Compare satellite variables with local sensor data  
- Learn correction relationships  
- Apply calibration to all historical satellite data  

**Outcome:**
- Satellite data becomes **site-adjusted**
- Better representation of:
  - Local climate behavior  
  - Seasonal trends  
  - Monsoon variability  

---

### ⚙️ 3. Physics-Based PV Generation Modeling

Calibrated weather data is converted into solar power generation using **physical PV modeling principles**:

- ☀️ Irradiance → Plane-of-array transformation  
- 🌡️ Temperature → Efficiency correction  
- ⚡ Environmental factors → Power output estimation  

This ensures that generated data follows **real-world physical relationships**, not just statistical patterns.

---

### 📊 4. Dataset for Model Training

The final dataset includes:

- 📡 Calibrated multi-year satellite weather data  
- ⚙️ Physics-based PV generation estimates  
- 📍 Real measured data (for validation & fine-tuning)  

This enables:
- Training under **data-scarce conditions**
- Capturing **seasonal and weather patterns**
- Improving **model generalization**

---

### ⚠️ Limitations

> The generated dataset is a **calibrated proxy**, not a perfect replacement for real historical data.

- May not capture all extreme or rare events  
- Performance depends on calibration quality  
- Accuracy improves as more real data becomes available  

---
