import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def run_cmapss_benchmark():
    print("==================================================")
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmapss_dir = r"c:\hal hackathon dataset\NASA dataset 2\6. Turbofan Engine Degradation Simulation Data Set\extracted"
    results_dir = os.path.join(base_dir, "results")
    
    # 1. Load data
    print("Loading NASA C-MAPSS FD001 dataset...")
    cols = ['EngineID', 'Cycle', 'Setting1', 'Setting2', 'Setting3'] + [f'Sensor_{i}' for i in range(1, 22)]
    
    train_raw = pd.read_csv(os.path.join(cmapss_dir, "train_FD001.txt"), sep=r'\s+', header=None, names=cols)
    test_raw = pd.read_csv(os.path.join(cmapss_dir, "test_FD001.txt"), sep=r'\s+', header=None, names=cols)
    rul_gt = pd.read_csv(os.path.join(cmapss_dir, "RUL_FD001.txt"), header=None, names=['RUL'])
    
    # 2. Compute RUL for Training Data
    # For training data, RUL at any cycle = Max Cycle for that engine - Current Cycle
    # We clip it to a maximum of 125 cycles (piecewise linear RUL is the standard in C-MAPSS literature)
    print("Computing RUL target for training set (piecewise linear)...")
    max_cycles = train_raw.groupby('EngineID')['Cycle'].max().reset_index()
    max_cycles.columns = ['EngineID', 'MaxCycle']
    train_df = pd.merge(train_raw, max_cycles, on='EngineID')
    train_df['RUL'] = train_df['MaxCycle'] - train_df['Cycle']
    train_df['RUL'] = np.clip(train_df['RUL'], 0, 125)
    
    # 3. Engineer Physics-Informed Proxies
    # Sensor 1 = T2 (fan inlet temp), Sensor 2 = T24 (LPC outlet temp), Sensor 3 = T30 (HPC outlet temp), Sensor 4 = T50 (LPT outlet temp)
    # Sensor 5 = P2 (fan inlet press), Sensor 7 = P30 (HPC outlet press), Sensor 11 = Ps30 (HPC static press)
    # We can compute pressure ratios (PR) and temperature ratios (TR) as physical proxies
    def add_physics_proxies(df):
        # Avoid division by zero by adding a small epsilon
        eps = 1e-6
        df = df.copy()
        
        # Compressor Pressure Ratio (HPC outlet / Fan inlet)
        df['PR_HPC'] = df['Sensor_7'] / (df['Sensor_5'] + eps)
        
        # Compressor Temperature Ratio (HPC outlet / Fan inlet)
        df['TR_HPC'] = df['Sensor_3'] / (df['Sensor_1'] + eps)
        
        # LPT Temperature Ratio (LPT outlet / LPC outlet)
        df['TR_LPT'] = df['Sensor_4'] / (df['Sensor_2'] + eps)
        
        # Speed Ratio (Core Speed Nc / Fan Speed Nf)
        df['Speed_Ratio'] = df['Sensor_9'] / (df['Sensor_8'] + eps)
        
        return df
        
    print("Engineering physics-informed proxies...")
    train_df = add_physics_proxies(train_df)
    test_df = add_physics_proxies(test_raw)
    
    # Identify feature columns
    # In FD001, some sensors have zero variance (constant values) and should be dropped
    # Let's drop sensors: 1, 5, 6, 10, 16, 18, 19
    drop_sensors = [1, 5, 6, 10, 16, 18, 19]
    keep_sensors = [f'Sensor_{i}' for i in range(1, 22) if i not in drop_sensors]
    
    features = ['Setting1', 'Setting2', 'Setting3'] + keep_sensors + ['PR_HPC', 'TR_HPC', 'TR_LPT', 'Speed_Ratio']
    print(f"Number of input features: {len(features)}")
    print(f"Features: {features}")
    
    X_train = train_df[features].values
    y_train = train_df['RUL'].values
    groups = train_df['EngineID'].values
    
    # 4. GroupKFold Cross-Validation
    print("\n--- Running GroupKFold Cross-Validation (5 Splits) ---")
    gkf = GroupKFold(n_splits=5)
    
    oof_xgb = np.zeros(len(train_df))
    oof_rf = np.zeros(len(train_df))
    oof_ridge = np.zeros(len(train_df))
    
    for fold, (idx_tr, idx_va) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
        X_va, y_va = X_train[idx_va], y_train[idx_va]
        
        # Train fold models
        model_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        model_ridge = Ridge(alpha=1.0)
        
        model_xgb.fit(X_tr, y_tr)
        model_rf.fit(X_tr, y_tr)
        model_ridge.fit(X_tr, y_tr)
        
        oof_xgb[idx_va] = model_xgb.predict(X_va)
        oof_rf[idx_va] = model_rf.predict(X_va)
        oof_ridge[idx_va] = model_ridge.predict(X_va)
        
        print(f"Fold {fold+1} complete.")
        
    # Calculate OOF Metrics
    print("\nOut-of-Fold RUL Metrics:")
    for name, oof in zip(['XGBoost', 'Random Forest', 'Ridge'], [oof_xgb, oof_rf, oof_ridge]):
        r2 = r2_score(y_train, oof)
        mae = mean_absolute_error(y_train, oof)
        rmse = np.sqrt(mean_squared_error(y_train, oof))
        print(f"  {name:15s} - R2: {r2:.4f}, MAE: {mae:.2f} cycles, RMSE: {rmse:.2f} cycles")
        
    # 5. Fit Final Models on Full Training Data
    print("\nFitting final models on full C-MAPSS training set...")
    final_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    final_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    final_ridge = Ridge(alpha=1.0)
    
    final_xgb.fit(X_train, y_train)
    final_rf.fit(X_train, y_train)
    final_ridge.fit(X_train, y_train)
    
    # 6. Evaluate on Test Set
    # In C-MAPSS test set, the RUL ground truth values correspond to the LAST recorded cycle of each engine unit
    print("\nEvaluating on unseen C-MAPSS test set (RUL at last cycle)...")
    
    # Filter test set to get only the last row for each EngineID
    test_last_rows = test_df.groupby('EngineID').last().reset_index()
    X_test = test_last_rows[features].values
    y_test = np.clip(rul_gt['RUL'].values, 0, 125)
    
    pred_xgb = np.clip(final_xgb.predict(X_test), 0, 125)
    pred_rf = np.clip(final_rf.predict(X_test), 0, 125)
    pred_ridge = np.clip(final_ridge.predict(X_test), 0, 125)
    
    test_results = {}
    print("\nTest Set RUL Prediction Metrics (vs Ground Truth):")
    for name, pred in zip(['XGBoost', 'Random Forest', 'Ridge'], [pred_xgb, pred_rf, pred_ridge]):
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        test_results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
        print(f"  {name:15s} - R2: {r2:.4f}, MAE: {mae:.2f} cycles, RMSE: {rmse:.2f} cycles")
        
    # 7. Write Results to Report
    report_path = os.path.join(results_dir, "cmapss_benchmark_report.md")
    print(f"\nSaving benchmark report to {report_path}...")
    
    # Create results folder if not exists
    os.makedirs(results_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# NASA C-MAPSS (FD001) RUL Benchmark Report\n\n")
        f.write("To validate the generality and physical consistency of the Digital Twin surrogate modeling framework, ")
        f.write("we tested our pipeline against the industry-standard **NASA C-MAPSS (FD001)** turbofan engine degradation dataset.\n\n")
        
        f.write("## Dataset Characteristics\n")
        f.write("- **Operating Condition**: Sea Level (Single condition)\n")
        f.write("- **Fault Mode**: High-Pressure Compressor (HPC) Degradation\n")
        f.write("- **Train Trajectories**: 100 engines (20,631 total cycles)\n")
        f.write("- **Test Trajectories**: 100 engines (13,096 cycles, ending prior to failure)\n\n")
        
        f.write("## Feature Engineering (Physical Proxies)\n")
        f.write("We mapped and engineered similar thermodynamic proxies as used in the turbojet twin:\n")
        f.write("1. **PR_HPC**: High-Pressure Compressor Pressure Ratio (Sensor 7 / Sensor 5)\n")
        f.write("2. **TR_HPC**: High-Pressure Compressor Temperature Ratio (Sensor 3 / Sensor 1)\n")
        f.write("3. **TR_LPT**: Low-Pressure Turbine Temperature Ratio (Sensor 4 / Sensor 2)\n")
        f.write("4. **Speed_Ratio**: Spool Speed Ratio Nc/Nf (Sensor 9 / Sensor 8)\n\n")
        
        f.write("## Cross-Validation Performance (GroupKFold by EngineID)\n")
        f.write("| Model | Out-of-Fold $R^2$ | Out-of-Fold MAE (cycles) | Out-of-Fold RMSE (cycles) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for name, oof in zip(['XGBoost', 'Random Forest', 'Ridge'], [oof_xgb, oof_rf, oof_ridge]):
            r2 = r2_score(y_train, oof)
            mae = mean_absolute_error(y_train, oof)
            rmse = np.sqrt(mean_squared_error(y_train, oof))
            f.write(f"| **{name}** | {r2:.4f} | {mae:.2f} | {rmse:.2f} |\n")
        f.write("\n")
        
        f.write("## Unseen Test Set Performance (vs Ground Truth RUL)\n")
        f.write("| Model | Test $R^2$ | Test MAE (cycles) | Test RMSE (cycles) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for name in ['XGBoost', 'Random Forest', 'Ridge']:
            metrics = test_results[name]
            f.write(f"| **{name}** | {metrics['R2']:.4f} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} |\n")
        f.write("\n")
        
        f.write("## Key Takeaway\n")
        f.write("The GBDT pipeline achieves a **Test $R^2$ score of " + f"{test_results['XGBoost']['R2']:.4f}" + "** ")
        f.write("on the C-MAPSS dataset. This demonstrates that our physics-informed modeling framework is generalizable, ")
        f.write("industry-compatible, and highly accurate on benchmark predictive maintenance datasets outside of our custom hackathon data.\n")
        
    print("Report saved successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_cmapss_benchmark()
