# Turbojet Digital Twin - Model Training Metrics Report

This report summarizes the Out-of-Fold (OOF) cross-validation performance of three models: XGBoost (primary surrogate), RandomForest (baseline), and Ridge (interpretable physics-only baseline) using 5-Fold GroupKFold cross-validation grouped by `EngineID` on the training dataset.

## Target Performance Table

| Target | Model | $R^2$ | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **CompressorHealth** | XGBoost | 0.6067 | 0.0326 | 0.0453 |
| | RandomForest | 0.5957 | 0.0343 | 0.0460 |
| | Ridge | 0.3921 | 0.0448 | 0.0564 |
| **CombustorHealth** | XGBoost | 0.4655 | 0.0147 | 0.0189 |
| | RandomForest | 0.4635 | 0.0146 | 0.0190 |
| | Ridge | 0.3232 | 0.0172 | 0.0213 |
| **TurbineHealth** | XGBoost | 0.4929 | 0.0281 | 0.0373 |
| | RandomForest | 0.4769 | 0.0287 | 0.0379 |
| | Ridge | 0.3322 | 0.0336 | 0.0428 |
| **OverallHealth** | XGBoost | 0.6079 | 0.0237 | 0.0319 |
| | RandomForest | 0.5856 | 0.0245 | 0.0328 |
| | Ridge | 0.3972 | 0.0314 | 0.0395 |
| **Thrust_N** | XGBoost | 0.9823 | 1807.5086 | 2321.5657 |
| | RandomForest | 0.9585 | 2906.9533 | 3560.3950 |
| | Ridge | 0.9010 | 4256.9055 | 5496.8239 |
| **TSFC_g_N_s** | XGBoost | 0.9821 | 0.0007 | 0.0009 |
| | RandomForest | 0.9726 | 0.0008 | 0.0011 |
| | Ridge | 0.9547 | 0.0010 | 0.0014 |

---

## Observations
1. **Performance Predictions**: Performance indicators (`Thrust_N` and `TSFC_g_N_s`) are highly predictable, achieving $R^2 > 0.98$ for the XGBoost model.
2. **Health Metrics**: Component healths are harder to predict, but XGBoost consistently outperforms RandomForest and Ridge.
3. **Physics features (Ridge)**: Although Ridge is a linear model fitted on physics features only, it maintains a respectable baseline (especially for `TSFC` and `Thrust`), confirming that the engineered thermodynamic features carry strong physical signals.
