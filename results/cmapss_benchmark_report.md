# NASA C-MAPSS (FD001) RUL Benchmark Report

To validate the generality and physical consistency of the Digital Twin surrogate modeling framework, we tested our pipeline against the industry-standard **NASA C-MAPSS (FD001)** turbofan engine degradation dataset.

## Dataset Characteristics
- **Operating Condition**: Sea Level (Single condition)
- **Fault Mode**: High-Pressure Compressor (HPC) Degradation
- **Train Trajectories**: 100 engines (20,631 total cycles)
- **Test Trajectories**: 100 engines (13,096 cycles, ending prior to failure)

## Feature Engineering (Physical Proxies)
We mapped and engineered similar thermodynamic proxies as used in the turbojet twin:
1. **PR_HPC**: High-Pressure Compressor Pressure Ratio (Sensor 7 / Sensor 5)
2. **TR_HPC**: High-Pressure Compressor Temperature Ratio (Sensor 3 / Sensor 1)
3. **TR_LPT**: Low-Pressure Turbine Temperature Ratio (Sensor 4 / Sensor 2)
4. **Speed_Ratio**: Spool Speed Ratio Nc/Nf (Sensor 9 / Sensor 8)

## Cross-Validation Performance (GroupKFold by EngineID)
| Model | Out-of-Fold $R^2$ | Out-of-Fold MAE (cycles) | Out-of-Fold RMSE (cycles) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | 0.7990 | 13.55 | 18.68 |
| **Random Forest** | 0.7932 | 13.98 | 18.95 |
| **Ridge** | 0.7288 | 17.75 | 21.70 |

## Unseen Test Set Performance (vs Ground Truth RUL)
| Model | Test $R^2$ | Test MAE (cycles) | Test RMSE (cycles) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | 0.8190 | 11.89 | 17.05 |
| **Random Forest** | 0.8090 | 12.46 | 17.51 |
| **Ridge** | 0.7363 | 16.37 | 20.58 |

## Key Takeaway
The GBDT pipeline achieves a **Test $R^2$ score of 0.8190** on the C-MAPSS dataset. This demonstrates that our physics-informed modeling framework is generalizable, industry-compatible, and highly accurate on benchmark predictive maintenance datasets outside of our custom hackathon data.
