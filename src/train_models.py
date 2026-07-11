import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from physics_features import add_physics_features

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    print("Loading data...")
    # Load training data
    train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    gt = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    
    # Merge ground truth to get labels for training rows
    train_df = pd.merge(train_raw, gt, on=['EngineID', 'Cycle'], how='inner')
    
    print(f"Training data shape: {train_df.shape}")
    
    # Generate physics features
    train_df = add_physics_features(train_df)
    
    # Define features and targets
    raw_features = [
        'Altitude_m', 'Mach', 'Tamb_K', 'Pamb_Pa', 'RPM_rev_min', 'FuelFlow_kg_s',
        'P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K'
    ]
    physics_features = [
        'Compressor_PR', 'Compressor_TR', 'Compressor_Eff', 
        'Combustor_TRise', 'Combustor_Efficiency_Proxy', 
        'Turbine_ER', 'Turbine_TDrop', 'Turbine_Eff', 
        'Corrected_RPM', 'Corrected_FuelFlow', 'Estimated_Thrust_Proxy'
    ]
    all_features = raw_features + physics_features
    
    targets = [
        'CompressorHealth', 'CombustorHealth', 'TurbineHealth', 
        'OverallHealth', 'Thrust_N', 'TSFC_g_N_s'
    ]
    
    groups = train_df['EngineID'].values
    X_all = train_df[all_features].values
    X_phys = train_df[physics_features].values
    Y = train_df[targets].values
    
    # Set global seed for reproducibility
    np.random.seed(42)
    
    # GroupKFold Cross Validation
    gkf = GroupKFold(n_splits=5)
    
    # Tuned hyperparameters for XGBoost to improve health predictions
    models_xgb = [XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42) for _ in targets]
    models_rf = [RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42) for _ in targets]
    # Scaling features is critical for linear models like Ridge
    models_ridge = [make_pipeline(StandardScaler(), Ridge(alpha=1.0)) for _ in targets]
    
    # Arrays to store OOF (out-of-fold) predictions
    oof_xgb = np.zeros_like(Y)
    oof_rf = np.zeros_like(Y)
    oof_ridge = np.zeros_like(Y)
    
    print("Running GroupKFold CV...")
    for train_idx, val_idx in gkf.split(X_all, Y, groups):
        # XGBoost & Random Forest
        for i in range(len(targets)):
            models_xgb[i].fit(X_all[train_idx], Y[train_idx, i])
            oof_xgb[val_idx, i] = models_xgb[i].predict(X_all[val_idx])
            
            models_rf[i].fit(X_all[train_idx], Y[train_idx, i])
            oof_rf[val_idx, i] = models_rf[i].predict(X_all[val_idx])
            
            # Ridge (physics only with scaling)
            models_ridge[i].fit(X_phys[train_idx], Y[train_idx, i])
            oof_ridge[val_idx, i] = models_ridge[i].predict(X_phys[val_idx])
            
    # Print metrics
    print("\n--- Cross-Validation Results ---")
    for i, target in enumerate(targets):
        print(f"\nTarget: {target}")
        for name, oof in zip(['XGBoost', 'RandomForest', 'Ridge'], [oof_xgb, oof_rf, oof_ridge]):
            r2 = r2_score(Y[:, i], oof[:, i])
            mae = mean_absolute_error(Y[:, i], oof[:, i])
            rmse = np.sqrt(mean_squared_error(Y[:, i], oof[:, i]))
            print(f"  {name:12s} - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
    # Train final models on ALL data
    print("\nTraining final models on full data...")
    final_xgb = []
    final_rf = []
    final_ridge = []
    
    for i, target in enumerate(targets):
        xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(X_all, Y[:, i])
        final_xgb.append(xgb)
        
        rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        rf.fit(X_all, Y[:, i])
        final_rf.append(rf)
        
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X_phys, Y[:, i])
        final_ridge.append(ridge)

    # Uncertainty Ensemble (15 bootstrapped XGB models per target)
    print("\nTraining Uncertainty Ensemble (Bootstrap XGBoost)...")
    ensemble_models = {target: [] for target in targets}
    num_bootstraps = 15
    for i, target in enumerate(targets):
        for b in range(num_bootstraps):
            # Bootstrap sample (sample with replacement)
            # Use deterministic local random generator for indexing
            rng = np.random.default_rng(b + i * num_bootstraps)
            idx = rng.choice(len(train_df), size=len(train_df), replace=True)
            X_boot = X_all[idx]
            Y_boot = Y[idx, i]
            
            model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=b)
            model.fit(X_boot, Y_boot)
            ensemble_models[target].append(model)
            
    # Print Feature Importances for XGBoost
    print("\n--- Final XGBoost Feature Importances ---")
    for i, target in enumerate(targets):
        print(f"\n{target}:")
        importances = final_xgb[i].feature_importances_
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        for f in range(min(5, len(indices))): # Top 5
            print(f"  {all_features[indices[f]]}: {importances[indices[f]]:.4f}")
            
    print("\n--- Ridge Coefficients (Physics Features) ---")
    for i, target in enumerate(targets):
        print(f"\n{target}:")
        # Extract coefficients from the pipeline's Ridge step
        coefs = final_ridge[i].named_steps['ridge'].coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        for f in range(min(5, len(indices))):
            print(f"  {physics_features[indices[f]]}: {coefs[indices[f]]:.4f}")

    # Save models
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save as dicts
    joblib.dump(final_xgb, os.path.join(models_dir, 'xgboost_models.pkl'))
    joblib.dump(final_rf, os.path.join(models_dir, 'rf_models.pkl'))
    joblib.dump(final_ridge, os.path.join(models_dir, 'ridge_models.pkl'))
    joblib.dump(ensemble_models, os.path.join(models_dir, 'ensemble_models.pkl'))
    
    # Save features list to ensure matching during inference
    joblib.dump({'all_features': all_features, 'physics_features': physics_features, 'targets': targets}, os.path.join(models_dir, 'features_targets.pkl'))
    
    print("\nModels saved successfully in models/")

if __name__ == '__main__':
    main()
