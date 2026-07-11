# Turbojet Digital Twin - Model Training Metrics Report

This report summarizes the Out-of-Fold (OOF) cross-validation performance using 5-Fold GroupKFold cross-validation grouped by `EngineID`.

## Target Performance Table

| Target | Model | $R^2$ | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **CompressorHealth** | XGBoost | 0.9168 | 0.0154 | 0.0209 |
| **CompressorHealth** | RandomForest | 0.9184 | 0.0146 | 0.0206 |
| **CompressorHealth** | Ridge | 0.8933 | 0.0178 | 0.0236 |
| | | | | |
| **CombustorHealth** | XGBoost | 0.7447 | 0.0089 | 0.0131 |
| **CombustorHealth** | RandomForest | 0.7689 | 0.0084 | 0.0124 |
| **CombustorHealth** | Ridge | 0.7675 | 0.0090 | 0.0125 |
| | | | | |
| **TurbineHealth** | XGBoost | 0.8053 | 0.0166 | 0.0231 |
| **TurbineHealth** | RandomForest | 0.8080 | 0.0162 | 0.0229 |
| **TurbineHealth** | Ridge | 0.8127 | 0.0172 | 0.0227 |
| | | | | |
| **OverallHealth** | XGBoost | 0.9188 | 0.0098 | 0.0145 |
| **OverallHealth** | RandomForest | 0.9234 | 0.0095 | 0.0141 |
| **OverallHealth** | Ridge | 0.9177 | 0.0106 | 0.0146 |
| | | | | |
| **Thrust_N** | XGBoost | 0.9765 | 2134.8587 | 2676.9541 |
| **Thrust_N** | RandomForest | 0.9536 | 3080.7038 | 3761.5977 |
| **Thrust_N** | Ridge | 0.9739 | 2172.2184 | 2825.2698 |
| | | | | |
| **TSFC_g_N_s** | XGBoost | 0.9817 | 0.0007 | 0.0009 |
| **TSFC_g_N_s** | RandomForest | 0.9706 | 0.0009 | 0.0011 |
| **TSFC_g_N_s** | Ridge | 0.9755 | 0.0007 | 0.0010 |
| | | | | |
