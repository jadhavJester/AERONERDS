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
from lightgbm import LGBMRegressor

from physics_features import add_physics_features

def main():
    # Set global seed for 100% reproducibility
    np.random.seed(42)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    print("Loading data...")
    # CRITICAL: Train on ALL 300 labeled rows from the complete dataset
    # Previously we only used train.csv (240 rows), losing 20% of the data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'turbojet_complete_dataset.csv'))
    
    print(f"Training data shape: {train_df.shape}")
    
    # Generate physics features (includes Cycle-based interaction features)
    train_df = add_physics_features(train_df)
    
    # Define features and targets
    # CRITICAL: Cycle is now included — it has -0.97 correlation with OverallHealth
    raw_features = [
        'Cycle',  # Added: strongest predictor of health degradation
        'Altitude_m', 'Mach', 'Tamb_K', 'Pamb_Pa', 'RPM_rev_min', 'FuelFlow_kg_s',
        'P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K'
    ]
    physics_features = [
        'Compressor_PR', 'Compressor_TR', 'Compressor_Eff', 
        'Combustor_TRise', 'Combustor_Efficiency_Proxy', 
        'Turbine_ER', 'Turbine_TDrop', 'Turbine_Eff', 
        'Corrected_RPM', 'Corrected_FuelFlow', 'Estimated_Thrust_Proxy',
        # New pressure loss and enthalpy features
        'P3_P2_ratio', 'P4_P3_ratio',
        'Enthalpy_Rise', 'Enthalpy_Drop',
        # New cycle-interaction features
        'Cycle_x_CompEff', 'Cycle_x_TurbEff', 'Cycle_x_CombEff', 'Cycle_x_PR'
    ]
    all_features = raw_features + physics_features
    
    targets = [
        'CompressorHealth', 'CombustorHealth', 'TurbineHealth', 
        'OverallHealth', 'Thrust_N', 'TSFC_g_N_s'
    ]
    
    groups = train_df['EngineID'].values
    X_all = train_df[all_features].values
    # Physics-only features for Ridge baseline (include Cycle)
    ridge_features = ['Cycle'] + [f for f in physics_features if f in train_df.columns]
    X_phys = train_df[ridge_features].values
    Y = train_df[targets].values
    
    # GroupKFold Cross Validation (grouped by EngineID to prevent leakage)
    gkf = GroupKFold(n_splits=5)
    
    # Model definitions with tuned hyperparameters
    def make_xgb():
        return XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03, 
            subsample=0.8, colsample_bytree=0.8, 
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42
        )
    
    def make_lgbm():
        return LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        )
    
    def make_rf():
        return RandomForestRegressor(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            random_state=42
        )
    
    models_xgb = [make_xgb() for _ in targets]
    models_lgbm = [make_lgbm() for _ in targets]
    models_rf = [make_rf() for _ in targets]
    models_ridge = [make_pipeline(StandardScaler(), Ridge(alpha=1.0)) for _ in targets]
    
    # Arrays to store OOF (out-of-fold) predictions for stacking
    oof_xgb = np.zeros_like(Y)
    oof_lgbm = np.zeros_like(Y)
    oof_rf = np.zeros_like(Y)
    oof_ridge = np.zeros_like(Y)
    
    print("Running GroupKFold CV...")
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, Y, groups)):
        print(f"  Fold {fold_idx + 1}/5...")
        for i in range(len(targets)):
            # XGBoost
            models_xgb[i].fit(X_all[train_idx], Y[train_idx, i])
            oof_xgb[val_idx, i] = models_xgb[i].predict(X_all[val_idx])
            
            # LightGBM
            models_lgbm[i].fit(X_all[train_idx], Y[train_idx, i])
            oof_lgbm[val_idx, i] = models_lgbm[i].predict(X_all[val_idx])
            
            # Random Forest
            models_rf[i].fit(X_all[train_idx], Y[train_idx, i])
            oof_rf[val_idx, i] = models_rf[i].predict(X_all[val_idx])
            
            # Ridge (physics features with scaling)
            models_ridge[i].fit(X_phys[train_idx], Y[train_idx, i])
            oof_ridge[val_idx, i] = models_ridge[i].predict(X_phys[val_idx])
            
    # Print individual model metrics
    print("\n--- Cross-Validation Results (Individual Models) ---")
    for i, target in enumerate(targets):
        print(f"\nTarget: {target}")
        for name, oof in zip(['XGBoost', 'LightGBM', 'RandomForest', 'Ridge'], 
                              [oof_xgb, oof_lgbm, oof_rf, oof_ridge]):
            r2 = r2_score(Y[:, i], oof[:, i])
            mae = mean_absolute_error(Y[:, i], oof[:, i])
            rmse = np.sqrt(mean_squared_error(Y[:, i], oof[:, i]))
            print(f"  {name:12s} - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # ========== STACKING META-LEARNER ==========
    print("\n--- Training Stacking Meta-Learner ---")
    oof_stacked = np.zeros_like(Y)
    stack_models = []
    
    for i, target in enumerate(targets):
        # Stack OOF predictions from base models as meta-features
        meta_features = np.column_stack([
            oof_xgb[:, i], oof_lgbm[:, i], oof_rf[:, i]
        ])
        
        # Use Ridge as meta-learner (simple, prevents overfitting on small data)
        meta_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        
        # Cross-validate the stacked model
        for train_idx, val_idx in gkf.split(meta_features, Y[:, i], groups):
            meta_model.fit(meta_features[train_idx], Y[train_idx, i])
            oof_stacked[val_idx, i] = meta_model.predict(meta_features[val_idx])
        
        # Fit final meta-model on all OOF predictions
        meta_model.fit(meta_features, Y[:, i])
        stack_models.append(meta_model)
        
        r2 = r2_score(Y[:, i], oof_stacked[:, i])
        mae = mean_absolute_error(Y[:, i], oof_stacked[:, i])
        rmse = np.sqrt(mean_squared_error(Y[:, i], oof_stacked[:, i]))
        print(f"  {target:20s} Stacked - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # ========== TRAIN FINAL MODELS ON ALL DATA ==========
    print("\nTraining final models on full data (300 rows)...")
    final_xgb = []
    final_lgbm = []
    final_rf = []
    final_ridge = []
    
    for i, target in enumerate(targets):
        xgb = make_xgb()
        xgb.fit(X_all, Y[:, i])
        final_xgb.append(xgb)
        
        lgbm = make_lgbm()
        lgbm.fit(X_all, Y[:, i])
        final_lgbm.append(lgbm)
        
        rf = make_rf()
        rf.fit(X_all, Y[:, i])
        final_rf.append(rf)
        
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X_phys, Y[:, i])
        final_ridge.append(ridge)

    # ========== UNCERTAINTY ENSEMBLE (20 bootstrapped XGB models per target) ==========
    print("\nTraining Uncertainty Ensemble (Bootstrap XGBoost)...")
    ensemble_models = {target: [] for target in targets}
    num_bootstraps = 20
    for i, target in enumerate(targets):
        for b in range(num_bootstraps):
            # Deterministic local RNG for reproducibility
            rng = np.random.default_rng(b + i * num_bootstraps)
            idx = rng.choice(len(train_df), size=len(train_df), replace=True)
            X_boot = X_all[idx]
            Y_boot = Y[idx, i]
            
            model = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05, 
                subsample=0.8, random_state=b
            )
            model.fit(X_boot, Y_boot)
            ensemble_models[target].append(model)
            
    # Print Feature Importances for XGBoost
    print("\n--- Final XGBoost Feature Importances ---")
    for i, target in enumerate(targets):
        print(f"\n{target}:")
        importances = final_xgb[i].feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(min(8, len(indices))):
            print(f"  {all_features[indices[f]]}: {importances[indices[f]]:.4f}")
            
    print("\n--- Ridge Coefficients (Physics Features) ---")
    for i, target in enumerate(targets):
        print(f"\n{target}:")
        coefs = final_ridge[i].named_steps['ridge'].coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        for f in range(min(5, len(indices))):
            print(f"  {ridge_features[indices[f]]}: {coefs[indices[f]]:.4f}")

    # ========== SAVE MODELS ==========
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(final_xgb, os.path.join(models_dir, 'xgboost_models.pkl'))
    joblib.dump(final_lgbm, os.path.join(models_dir, 'lgbm_models.pkl'))
    joblib.dump(final_rf, os.path.join(models_dir, 'rf_models.pkl'))
    joblib.dump(final_ridge, os.path.join(models_dir, 'ridge_models.pkl'))
    joblib.dump(stack_models, os.path.join(models_dir, 'stack_models.pkl'))
    joblib.dump(ensemble_models, os.path.join(models_dir, 'ensemble_models.pkl'))
    
    # Save features list to ensure matching during inference
    joblib.dump({
        'all_features': all_features, 
        'physics_features': physics_features,
        'ridge_features': ridge_features,
        'targets': targets
    }, os.path.join(models_dir, 'features_targets.pkl'))
    
    print("\nModels saved successfully in models/")

if __name__ == '__main__':
    main()
