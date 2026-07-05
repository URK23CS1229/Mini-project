import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_time_series_dataset(input_csv='dataset/final_dataset.csv', output_csv='dataset/processed_dataset.csv'):
    # Load dataset
    df = pd.read_csv(input_csv)
    
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort dataset in chronological order
    df = df.sort_values(by=['region', 'timestamp']).reset_index(drop=True)
    
    # Generate time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Create lag features for time-series learning
    df = df.sort_values(by=['region', 'timestamp'])
    df['carbon_lag1'] = df.groupby('region')['carbon_intensity'].shift(1)
    df['temp_lag1'] = df.groupby('region')['temperature'].shift(1)
    df['energy_lag1'] = df.groupby('region')['energy_consumption'].shift(1)
    
    # Handle missing values created by lag operations
    df = df.dropna()
    
    # One-hot encode categorical region column
    df = pd.get_dummies(df, columns=['region'], drop_first=False)
    
    # Normalize numerical columns using MinMax scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    numerical_cols = [
        'carbon_intensity', 'temperature', 'cpu_usage', 
        'workload', 'energy_consumption', 'cost',
        'carbon_lag1', 'temp_lag1', 'energy_lag1',
        'hour', 'day_of_week'
    ]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Ensure no missing values
    df = df.dropna()
    
    # Save processed dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Preprocessed dataset saved to {output_csv}")
    print(f"Shape of processed dataset: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    preprocess_time_series_dataset()