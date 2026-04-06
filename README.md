⚡ Cost-Optimized Microgrid Generation Forecasting
🧭 Introduction
This project focuses on developing a solar power generation forecasting model as a core component of a cost-optimized microgrid controller.
Accurate generation forecasting is essential for optimal energy dispatch, cost minimization, and reliable microgrid operation. However, a key challenge in this project is data scarcity, as the target microgrid system is still under development and lacks a long-term data logging infrastructure.
Currently, only limited local data (~1 year) is available, which is insufficient for training robust forecasting models that capture seasonal and weather-driven variability, especially in a tropical region like Sri Lanka with strong monsoon effects.
To address this, the project adopts a hybrid data-driven and physics-informed approach, leveraging satellite-derived data, calibration techniques, and physical modeling to construct a reliable forecasting pipeline under constrained data conditions.
🧪 Data Generation & Calibration Strategy
Due to the absence of multi-year local measurements, this project constructs a pseudo-historical dataset using satellite/reanalysis data combined with site-specific calibration and physical modeling.
🔹 1. Satellite & Reanalysis Data Collection
Multi-year weather and solar irradiance data are retrieved from sources such as:
NASA POWER API (irradiance, temperature, humidity, etc.)
These datasets provide consistent, long-term environmental variables required for modeling solar generation.
🔹 2. Site-Specific Calibration
Since satellite-derived data may not perfectly represent local conditions, a calibration step is performed using the available one-year real dataset.
Satellite variables (e.g., temperature, humidity) are compared with local sensor measurements
Correction relationships are learned to align satellite data with real site behavior
Calibration is applied across all historical satellite data to create site-adjusted inputs
This step ensures that the generated dataset reflects local climatic characteristics, including seasonal and monsoon-driven variations.
🔹 3. Physics-Based PV Generation Modeling
The calibrated weather and irradiance data are then used to generate solar power output using physical PV modeling principles:
Solar irradiance → Plane-of-array conversion
Temperature effects → Module efficiency adjustments
Environmental conditions → Power output estimation
This produces a multi-year synthetic PV generation dataset grounded in physical relationships rather than purely statistical assumptions.
🔹 4. Dataset for Model Training
The final dataset consists of:
Calibrated multi-year satellite-derived weather data
Physics-based PV generation estimates
Real measured data (used for validation and fine-tuning)
This hybrid dataset enables:
Training of forecasting models despite limited local data
Improved generalization across seasonal patterns
Better representation of real-world operating conditions
