import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def create_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def load_and_split_data_by_region(input_path='dataset/final_dataset.csv', target_column='carbon_intensity'):
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    
    # Add same time features as train_model.py
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Lag features
    df['carbon_lag_1h'] = df.groupby('region')['carbon_intensity'].shift(1)
    df['carbon_lag_24h'] = df.groupby('region')['carbon_intensity'].shift(24)
    
    df = df.dropna()
    
    regions = df['region'].unique()
    region_datasets = {}
    
    for region in regions:
        region_df = df[df['region'] == region].reset_index(drop=True)
        split_idx = int(len(region_df) * 0.8)
        train_df = region_df.iloc[:split_idx]
        test_df = region_df.iloc[split_idx:]
        region_datasets[region] = (train_df, test_df)
    
    global_test_df = df.iloc[int(len(df) * 0.8):]
    return region_datasets, global_test_df

def get_feature_columns(df, target_column):
    exclude_cols = ['timestamp', target_column]
    return [col for col in df.columns if col not in exclude_cols]

def train_local_model(train_df, region, target_column):
    feature_cols = get_feature_columns(train_df, target_column)
    
    # Encode region
    train_df = pd.get_dummies(train_df, columns=['region'], drop_first=True)
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', target_column]]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_column]
    
    # Use optimized XGBoost parameters
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    print(f"Trained local model for {region} (samples: {len(X_train)})")
    return model, feature_cols

def evaluate_model(model, test_df, feature_cols, target_column, model_name):
    # Encode region
    test_df = pd.get_dummies(test_df, columns=['region'], drop_first=True)
    # Ensure all columns exist
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    X_test = test_df[feature_cols]
    y_true = test_df[target_column]
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'target': target_column,
        'model_type': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }

class FederatedAveragingEnsemble:
    def __init__(self, local_models):
        self.local_models = local_models
        self.n_models = len(local_models)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.local_models])
        return np.mean(predictions, axis=0)

def federated_averaging_ensemble(local_models):
    return FederatedAveragingEnsemble(local_models)

def run_federated_learning(target_column):
    print(f"\n{'='*70}")
    print(f"Running Federated Learning for {target_column} prediction")
    print(f"{'='*70}")
    
    region_datasets, global_test_df = load_and_split_data_by_region(target_column=target_column)
    feature_cols = get_feature_columns(next(iter(region_datasets.values()))[0], target_column)
    
    local_models = {}
    local_metrics = []
    
    print("\n=== Training Local Models per Region ===")
    for region, (train_df, test_df) in region_datasets.items():
        model, _ = train_local_model(train_df, region, target_column)
        local_models[region] = model
        
        # Evaluate local model on regional test set
        metrics = evaluate_model(model, test_df, feature_cols, target_column, f"local_{region.lower()}")
        local_metrics.append(metrics)
        print(f"Region {region} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2_score']:.4f}")
        
        # Save local model
        joblib.dump(model, f'models/local_model_{region.lower()}.pkl')
    
    print("\n=== Federated Averaging (FedAvg) Aggregation ===")
    global_model = federated_averaging_ensemble(list(local_models.values()))
    joblib.dump(global_model, f'models/global_federated_model_{target_column}.pkl')
    print(f"Aggregated global model saved to models/global_federated_model_{target_column}.pkl")
    
    print("\n=== Evaluating Global Model ===")
    X_global_test = global_test_df[feature_cols]
    y_global_true = global_test_df[target_column]
    y_global_pred = global_model.predict(X_global_test)
    
    global_mae = mean_absolute_error(y_global_true, y_global_pred)
    global_rmse = np.sqrt(mean_squared_error(y_global_true, y_global_pred))
    global_r2 = r2_score(y_global_true, y_global_pred)
    
    global_metrics = {
        'target': target_column,
        'model_type': 'global_federated',
        'mae': global_mae,
        'rmse': global_rmse,
        'r2_score': global_r2
    }
    print(f"Global Model | MAE: {global_mae:.4f} | RMSE: {global_rmse:.4f} | R²: {global_r2:.4f}")
    
    return local_metrics + [global_metrics]

def main():
    print("=== Phase 4: Federated Learning for Carbon-Aware Workload Optimization ===")
    create_directories()
    
    all_metrics = []
    
    # Run federated learning for carbon intensity prediction
    carbon_metrics = run_federated_learning('carbon_intensity')
    all_metrics.extend(carbon_metrics)
    
    # Run federated learning for temperature prediction
    temp_metrics = run_federated_learning('temperature')
    all_metrics.extend(temp_metrics)
    
    # Save all federated learning metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('results/federated_metrics.csv', index=False)
    
    print("\n" + "="*70)
    print("Federated Learning Complete!")
    print(f"Results saved to results/federated_metrics.csv")
    print("\nFinal Performance Summary:")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()