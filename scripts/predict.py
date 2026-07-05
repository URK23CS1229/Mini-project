import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Required for unpickling federated ensemble model
class FederatedAveragingEnsemble:
    def __init__(self, local_models):
        self.local_models = local_models
        self.n_models = len(local_models)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.local_models])
        return np.mean(predictions, axis=0)

def create_directories():
    os.makedirs('results/plots', exist_ok=True)

def load_latest_data(input_path='dataset/final_dataset.csv', lookback_hours=24):
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    df = df.dropna()
    
    # Get latest N hours of data
    latest_timestamp = df['timestamp'].max()
    cutoff_timestamp = latest_timestamp - pd.Timedelta(hours=lookback_hours)
    latest_data = df[df['timestamp'] >= cutoff_timestamp].reset_index(drop=True)
    
    return latest_data, df

def create_prediction_features(current_data, future_hour):
    features = current_data.copy()
    
    # Carbon model features
    carbon_feature_cols = [
        'temperature', 'cpu_usage', 'workload', 
        'energy_consumption', 'cost'
    ]
    
    # Temperature model features
    temp_feature_cols = [
        'carbon_intensity', 'cpu_usage', 'workload', 
        'energy_consumption', 'cost'
    ]
    
    future_timestamp = current_data['timestamp'].iloc[-1] + pd.Timedelta(hours=future_hour)
    
    X_carbon = features[carbon_feature_cols].iloc[[-1]]
    X_temp = features[temp_feature_cols].iloc[[-1]]
    
    return X_carbon, X_temp, future_timestamp

def predict_next_hours(region_data, carbon_model, temp_model, hours_ahead=3):
    predictions = []
    current_state = region_data.copy()
    
    for hour in range(1, hours_ahead + 1):
        # Prepare input features for this prediction step
        X_carbon, X_temp, pred_timestamp = create_prediction_features(current_state, hour)
        
        # Generate predictions
        pred_carbon = carbon_model.predict(X_carbon)[0]
        pred_temp = temp_model.predict(X_temp)[0]
        
        predictions.append({
            'timestamp': pred_timestamp,
            'region': region_data['region'].iloc[0],
            'predicted_carbon_intensity': round(pred_carbon, 2),
            'predicted_temperature': round(pred_temp, 4)
        })
        
        # Update current state with predictions for next iteration
        new_row = current_state.iloc[[-1]].copy()
        new_row['timestamp'] = pred_timestamp
        new_row['carbon_intensity'] = pred_carbon
        new_row['temperature'] = pred_temp
        current_state = pd.concat([current_state, new_row], ignore_index=True)
    
    return pd.DataFrame(predictions)

def plot_predictions(historical_df, predictions_df, region):
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 24 hours)
    region_hist = historical_df[historical_df['region'] == region]
    cutoff = predictions_df['timestamp'].min() - pd.Timedelta(hours=24)
    recent_hist = region_hist[region_hist['timestamp'] >= cutoff]
    
    # Carbon intensity plot
    plt.subplot(2, 1, 1)
    plt.plot(recent_hist['timestamp'], recent_hist['carbon_intensity'], label='Historical', marker='.')
    plt.plot(predictions_df['timestamp'], predictions_df['predicted_carbon_intensity'], 
             label='Predicted', color='red', marker='o', linestyle='--')
    plt.title(f'{region} - Carbon Intensity Forecast (Next 3 Hours)')
    plt.ylabel('gCO2/kWh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Temperature plot
    plt.subplot(2, 1, 2)
    plt.plot(recent_hist['timestamp'], recent_hist['temperature'], label='Historical', marker='.', color='green')
    plt.plot(predictions_df['timestamp'], predictions_df['predicted_temperature'], 
             label='Predicted', color='orange', marker='o', linestyle='--')
    plt.title(f'{region} - Temperature Forecast (Next 3 Hours)')
    plt.xlabel('Timestamp')
    plt.ylabel('°C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/forecast_{region.lower()}.png')
    plt.close()

def main():
    print("=== Phase 5: Prediction Pipeline for Carbon-Aware Workload Optimization ===")
    
    create_directories()
    
    # Load models (using XGBoost models directly for prediction)
    print("\nLoading trained models...")
    carbon_model = joblib.load('models/carbon_model.pkl')
    temp_model = joblib.load('models/temp_model.pkl')
    print("Models loaded successfully")
    
    # Load data
    print("\nLoading dataset...")
    latest_data, full_df = load_latest_data()
    regions = latest_data['region'].unique()
    print(f"Loaded latest 24 hours of data for regions: {', '.join(regions)}")
    
    # Generate predictions for each region
    all_predictions = []
    
    print("\nGenerating 3-hour ahead predictions for all regions...")
    for region in regions:
        print(f"\nProcessing region: {region}")
        region_data = latest_data[latest_data['region'] == region].reset_index(drop=True)
        
        if len(region_data) < 1:
            print(f"Warning: Insufficient data for region {region}, skipping")
            continue
        
        region_predictions = predict_next_hours(region_data, carbon_model, temp_model, hours_ahead=3)
        all_predictions.append(region_predictions)
        
        # Generate prediction plots
        plot_predictions(full_df, region_predictions, region)
        print(f"Generated {len(region_predictions)} predictions for {region}")
        print(region_predictions.to_string(index=False))
    
    # Combine and save all predictions
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions = final_predictions.sort_values(['timestamp', 'region']).reset_index(drop=True)
    
    final_predictions.to_csv('results/predictions.csv', index=False)
    
    print("\n" + "="*70)
    print("Prediction Complete!")
    print(f"Total predictions generated: {len(final_predictions)}")
    print(f"Results saved to results/predictions.csv")
    print(f"Forecast plots saved to results/plots/")
    print("\nFinal Predictions Summary:")
    print(final_predictions.to_string(index=False))

if __name__ == "__main__":
    main()