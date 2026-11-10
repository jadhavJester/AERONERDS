import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

try:  # package import: `python -m src.predict` / `import src.predict`
    from src.physics_features import add_physics_features
    from src.degradation_trend import DegradationTrend
except ImportError:  # plain script: `python src/predict.py`
    from physics_features import add_physics_features
    from degradation_trend import DegradationTrend

def stack_predict(base_models, meta_model, X):
    """Stacked prediction: run each base model, then feed their outputs
    to the meta-learner as its features."""
    base_preds = [model.predict(X) for model in base_models]
    meta_features = np.column_stack(base_preds)
    return meta_model.predict(meta_features)


def bootstrap_percentiles(models, X, low=10, high=90):
    """Uncertainty band across bootstrap-ensemble predictions.

    Returns the ``low``th and ``high``th percentiles (default 10/90)
    of the per-row predictions of ``models``.
    """
    boot_preds = np.array([model.predict(X) for model in models])
    return (
        np.percentile(boot_preds, low, axis=0),
        np.percentile(boot_preds, high, axis=0),
    )


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    print("Loading models...")
    xgb_models = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_models.pkl'))
    lgbm_models = joblib.load(os.path.join(BASE_DIR, 'models', 'lgbm_models.pkl'))
    rf_models = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_models.pkl'))
    stack_models = joblib.load(os.path.join(BASE_DIR, 'models', 'stack_models.pkl'))
    ensemble_models = joblib.load(os.path.join(BASE_DIR, 'models', 'ensemble_models.pkl'))
    metadata = joblib.load(os.path.join(BASE_DIR, 'models', 'features_targets.pkl'))
    
    all_features = metadata['all_features']
    targets = metadata['targets']
    
    print("Loading test data...")
    test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    gt = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    
    print("Computing features...")
    test_df = add_physics_features(test_raw)
    X_test = test_df[all_features].values
    
    print("Running stacked inference...")
    predictions = { 'EngineID': test_df['EngineID'], 'Cycle': test_df['Cycle'] }
    
    for i, target in enumerate(targets):
        # Stack meta-learner prediction (final output)
        pred_stacked = stack_predict(
            [xgb_models[i], lgbm_models[i], rf_models[i]],
            stack_models[i],
            X_test,
        )

        predictions[target] = pred_stacked

        # Uncertainty from bootstrap ensemble (10th and 90th percentiles)
        p10, p90 = bootstrap_percentiles(ensemble_models[target], X_test)

        predictions[f'{target}_p10'] = p10
        predictions[f'{target}_p90'] = p90

    preds_df = pd.DataFrame(predictions)
    
    # Save predictions
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    print("Predictions saved to results/predictions.csv")
    
    # Validate against ground truth
    print("\n--- Validation against Ground Truth ---")
    val_df = pd.merge(preds_df, gt, on=['EngineID', 'Cycle'], suffixes=('_pred', '_gt'))
    
    for target in targets:
        y_true = val_df[target + '_gt'].values
        y_pred = val_df[target + '_pred'].values
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"\n{target}:")
        print(f"  R2:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
    print("\n--- RUL Estimation for Fleet (Test Engines) ---")
    engines = test_df['EngineID'].unique()
    for engine in sorted(engines):
        engine_mask = val_df['EngineID'] == engine
        engine_cycles = val_df.loc[engine_mask, 'Cycle'].values
        engine_health = val_df.loc[engine_mask, 'OverallHealth_pred'].values
        
        dt = DegradationTrend(maintenance_threshold=0.75)
        success = dt.fit(engine_cycles, engine_health)
        if success:
            crossover = dt.estimate_rul_crossover()
            print(f"Engine {engine}: Trend={dt.func_type:11s} | Est. crossover cycle: {crossover:.1f}")
        else:
            print(f"Engine {engine}: Not enough data to fit trend.")

if __name__ == '__main__':
    main()
