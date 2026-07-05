import pandas as pd
import joblib
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Testing data loading...")

# Test CSV files
csv_files = [
    'results/predictions.csv',
    'results/scheduling_results.csv',
    'results/scaling_results.csv',
    'results/full_region_scores.csv',
    'results/federated_metrics.csv',
    'results/model_metrics.csv'
]

for csv_file in csv_files:
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"[OK] {csv_file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"[ERROR] {csv_file}: {str(e)}")
    else:
        print(f"[MISSING] {csv_file}: FILE NOT FOUND")

print("\nTesting model loading...")

# Test model files
model_files = [
    'models/carbon_model.pkl',
    'models/temp_model.pkl',
    'models/global_federated_model_carbon_intensity.pkl',
    'models/global_federated_model_temperature.pkl'
]

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            model = joblib.load(model_file)
            print(f"[OK] {model_file}: Loaded successfully - {type(model).__name__}")
        except Exception as e:
            print(f"[ERROR] {model_file}: {str(e)}")
    else:
        print(f"[MISSING] {model_file}: FILE NOT FOUND")

print("\nAll tests completed!")
