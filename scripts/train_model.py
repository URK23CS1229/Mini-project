import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def create_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

def load_and_prepare_data(input_path='dataset/final_dataset.csv'):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by region and timestamp FIRST
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    
    # Add time features - BIGGEST ACCURACY IMPROVEMENT
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Cyclic encoding for hour (preserves 23→0 continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Lag features - carbon intensity is highly autocorrelated
    df['carbon_lag_1h'] = df.groupby('region')['carbon_intensity'].shift(1)
    df['carbon_lag_24h'] = df.groupby('region')['carbon_intensity'].shift(24)
    
    # Handle missing values
    df = df.dropna()
    
    return df

def time_based_train_test_split(df, train_size=0.8):
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def evaluate_model(y_true, y_pred, model_name, target_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance for {target_name}:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'target': target_name,
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name, target_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted {target_name} - {model_name}')
    plt.savefig(f'results/plots/{target_name}_{model_name}_actual_vs_predicted.png')
    plt.close()

def plot_feature_importance(model, feature_names, target_name, model_name):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(f'Feature Importance - {target_name} ({model_name})')
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'results/plots/{target_name}_{model_name}_feature_importance.png')
    plt.close()

def train_and_evaluate_models(train_df, test_df, target_column, model_save_path):
    # Define features and target
    exclude_cols = ['timestamp', target_column, 'carbon_intensity', 'temperature']
    if target_column == 'carbon_intensity':
        exclude_cols.remove('carbon_intensity')
    if target_column == 'temperature':
        exclude_cols.remove('temperature')
    
    # Encode region as one-hot
    if 'region' in train_df.columns:
        all_regions = pd.concat([train_df, test_df])['region'].unique()
        train_df['region'] = pd.Categorical(train_df['region'], categories=all_regions)
        test_df['region'] = pd.Categorical(test_df['region'], categories=all_regions)
        
        train_df = pd.get_dummies(train_df, columns=['region'], drop_first=True)
        test_df = pd.get_dummies(test_df, columns=['region'], drop_first=True)
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_column]
    X_test = test_df[feature_cols]
    y_test = test_df[target_column]
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Enhanced models with better hyperparameters
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }
    
    results = []
    best_model = None
    best_score = -np.inf
    best_model_name = None
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {target_column}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, model_name, target_column)
        results.append(metrics)
        
        # Print sample predictions
        print(f"\nSample Predictions for {target_column} ({model_name}):")
        sample_df = pd.DataFrame({
            'Actual': y_test.head(5).values,
            'Predicted': y_pred[:5]
        })
        print(sample_df)
        
        # Generate plots
        plot_actual_vs_predicted(y_test, y_pred, model_name, target_column)
        plot_feature_importance(model, feature_cols, target_column, model_name)
        
        # Track best model
        if metrics['r2_score'] > best_score:
            best_score = metrics['r2_score']
            best_model = model
            best_model_name = model_name
    
    # Save best model
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model for {target_column}: {best_model_name} (R² = {best_score:.4f})")
    print(f"Model saved to {model_save_path}")
    
    return results, best_model, best_model_name

def main():
    print("=== Phase 3: Machine Learning for Carbon-Aware Workload Optimization ===")
    
    # Create required directories
    create_directories()
    
    # Load and prepare data
    print("\nLoading and preparing dataset...")
    df = load_and_prepare_data()
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Time-based train-test split
    train_df, test_df = time_based_train_test_split(df)
    print(f"\nTrain set size: {len(train_df)} rows")
    print(f"Test set size: {len(test_df)} rows")
    
    # Train models for carbon_intensity
    print("\n" + "="*60)
    print("Training models for Carbon Intensity prediction")
    print("="*60)
    carbon_results, _, _ = train_and_evaluate_models(
        train_df, test_df, 
        target_column='carbon_intensity',
        model_save_path='models/carbon_model.pkl'
    )
    
    # Train models for temperature
    print("\n" + "="*60)
    print("Training models for Temperature prediction")
    print("="*60)
    temp_results, _, _ = train_and_evaluate_models(
        train_df, test_df, 
        target_column='temperature',
        model_save_path='models/temp_model.pkl'
    )
    
    # Combine and save all metrics
    all_results = carbon_results + temp_results
    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv('results/model_metrics.csv', index=False)
    print("\n" + "="*60)
    print("All model metrics saved to results/model_metrics.csv")
    print("\nFinal Model Comparison:")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()