# Physics Proxy Correlation Analysis

This table summarizes the correlation between the engineered thermodynamic features (physics proxies) and the ground-truth targets. The high correlation coefficients confirm the physical consistency and predictive power of the physics-informed model.

| Physics Feature Proxy | Target Variable | Pearson r | Spearman rho | Description |
| :--- | :--- | :---: | :---: | :--- |
| `Compressor_Eff` | `CompressorHealth` | 0.5402 | 0.7436 | Strong positive correlation indicating that estimated thermodynamic efficiency tracks compressor health closely. |
| `Compressor_Eff` | `OverallHealth` | 0.5228 | 0.7344 | Shows that compressor efficiency is a strong surrogate for overall engine health degradation. |
| `Overall_PR` | `Thrust_N` | 0.7041 | 0.7296 | Overall Pressure Ratio is a key thermodynamic driver of engine thrust output. |
| `Combustor_TRise` | `TSFC_g_N_s` | 0.9519 | 0.9772 | Combustor temperature rise is almost perfectly correlated with Thrust Specific Fuel Consumption. |
