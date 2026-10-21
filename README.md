# Turbojet Digital Twin ✈️

This repository contains a full **Physics-Informed Digital Twin** solution for real-time four-stage turbojet health monitoring (HAL x IIT Indore Aerothon 2026, Problem Statement 2).

---

## Project Structure

```
├── README.md                    # This file
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies (pinned versions)
├── requirements-dev.txt         # Dev/test dependencies
├── presentation_slides.md       # Project presentation
├── technical_report.md          # Detailed technical report
├── data/                        # Consolidated dataset directory
│   ├── train.csv                # Raw training sensor data
│   ├── test.csv                 # Raw testing sensor data (no labels)
│   ├── ground_truth.csv         # True labels for the entire dataset
│   └── turbojet_complete_dataset.csv # Pre-merged complete dataset
├── src/                         # Source code scripts
│   ├── physics_features.py      # Feature engineering module (thermodynamics)
│   ├── train_models.py          # ML training and cross-validation pipeline
│   ├── predict.py               # Inference, validation, and fleet RUL script
│   ├── degradation_trend.py     # Adaptive degradation curve fitting (RUL)
│   ├── physics_correlation.py   # Physics-feature correlation analysis + plots
│   ├── cmapss_benchmark.py      # NASA C-MAPSS benchmarking module
│   ├── benchmark_inference.py   # Inference latency benchmarking
│   └── run_tests_and_report.py  # Test runner with report generation
├── tests/                       # pytest test suite
│   ├── test_physics_features.py # Tests for physics feature engineering
│   ├── test_data_integrity.py   # Dataset schema/range integrity tests
│   └── test_degradation_trend.py# Tests for RUL trend fitting
├── models/                      # Saved trained models (.pkl)
├── results/                     # Output directory for predictions and reports
│   ├── predictions.csv          # Inference predictions with uncertainty bounds
│   └── metrics_report.md        # Detailed model performance metrics
└── dashboard/                   # Interactive UI application
    └── app.py                   # Multi-page Streamlit dashboard app
```

---

## How to Run

Run the commands below from the repository root:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   python src/train_models.py
   ```
   This engineers the physics features, trains XGBoost, RandomForest, Ridge, and Uncertainty Ensembles, and saves the artifacts to `models/`.

3. **Run Inference & Validation**
   ```bash
   python src/predict.py
   ```
   This loads the test data, runs inference with 10th-90th percentile bounds, outputs results to `results/predictions.csv`, prints metrics against `ground_truth.csv`, and runs fleet-wide RUL projections.

4. **Launch Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```
   This launches the interactive, dark-themed diagnostic dashboard.

---

## Physics-Informed Feature Engineering

To supplement raw sensor readings with gas turbine domain knowledge, we apply isentropic relations (assuming $\gamma = 1.4$, $C_p = 1004 \text{ J/kg.K}$, $R = 287.05 \text{ J/kg.K}$):

1. **Compressor & Turbine Efficiency**:
   Ideal isentropic relations are used to compute efficiency factors:
   - $PR_c = P_2 / P_{amb}$
   - $T_{ideal\_rise} = T_{amb} \times \left( PR_c^{\frac{\gamma-1}{\gamma}} - 1 \right)$
   - $\eta_{comp} = \frac{T_{ideal\_rise}}{T_2 - T_{amb}}$
   A matching approach calculates turbine expansion ratio $ER_t$ and efficiency $\eta_t$.

2. **Corrected Parameters**:
   Parameters are corrected for ambient temperature and pressure variation:
   - $\theta = T_{amb}/288.15$
   - $\delta = P_{amb}/101325$
   - $RPM_{corrected} = \frac{RPM}{\sqrt{\theta}}$
   - $FuelFlow_{corrected} = \frac{FuelFlow}{\delta \sqrt{\theta}}$

3. **Estimated Thrust Proxy**:
   Constructed from momentum principles:
   - $\dot{m}_{proxy} \propto \frac{P_3}{\sqrt{T_3}}$ (choked nozzle assumption)
   - $V_e \propto \sqrt{2 C_p (T_3 - T_4)}$
   - $V_0 = M \sqrt{\gamma R T_{amb}}$ (flight speed)
   - $Thrust_{proxy} = A_p \cdot \dot{m}_{proxy} (V_e - V_0)$

---

## Validation & Uncertainty Quantification

* **GroupKFold (by EngineID)**: Validation is performed by grouping cycles by `EngineID` (5 splits) to prevent temporal data leakage and ensure models generalize to entirely unseen engines.
* **Bootstrap Ensembling**: We train an ensemble of 15 bootstrapped XGBoost models per target to estimate the 10th and 90th prediction percentiles, providing a robust conformal uncertainty band around health and performance curves.
* **RUL Extrapolation**: RUL forecasts fit linear and exponential curves to health histories, selecting the model with the highest $R^2$, and projecting to find the cycle where health drops below the safety threshold of $0.75$.

---

## References & Inspirations

This implementation adapts and builds upon patterns from the following open-source resources:
1. **[rohanmatt/Predictive-Maintenance-for-Industrial-Equipment](https://github.com/rohanmatt/Predictive-Maintenance-for-Industrial-Equipment)**: Reference design for Streamlit sidebar-driven multi-page layout (Home, Fleet, Engine, Model tabs) and interactive time simulation.
2. **[shankara-93/predictive-maintenance-streamlit-blog](https://github.com/shankara-93/predictive-maintenance-streamlit-blog)**: Reference for lifecycle grouping by engine number and explanatory narrative format.
3. **NASA's C-MAPSS Dataset Literature**: Foundation for RUL degradation thresholds (0.75) and bootrap uncertainty bands.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Testing

Run the test suite with:

```bash
pip install pandas numpy pytest
python -m pytest -q
```

The tests in `tests/` cover the physics-informed feature engineering in `src/physics_features.py` and the RUL trend fitting in `src/degradation_trend.py` (model selection, threshold-crossover math, and an end-to-end fit on the real engine data in `data/`).
