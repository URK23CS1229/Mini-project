#!/usr/bin/env python3
"""
Manual input testing script for real-time carbon-aware scheduling predictions.
Accepts user inputs, generates predictions, and computes optimal scheduling decisions.
"""

import os
import sys
import datetime
import joblib
import numpy as np
from typing import Dict, Tuple, Optional, List


class FederatedAveragingEnsemble:
    """Placeholder class for federated model deserialization."""
    def __init__(self, local_models):
        self.local_models = local_models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.local_models])
        return np.mean(predictions, axis=0)


def load_models() -> Tuple[Optional[object], Optional[object]]:
    """Load trained carbon and temperature prediction models."""
    carbon_model = None
    temp_model = None
    
    model_paths = [
        ('carbon', 'models/global_federated_model_carbon_intensity.pkl'),
        ('carbon', 'models/carbon_model.pkl'),
        ('temp', 'models/global_federated_model_temperature.pkl'),
        ('temp', 'models/temp_model.pkl')
    ]
    
    for model_type, path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                if model_type == 'carbon' and carbon_model is None:
                    carbon_model = model
                    print(f"OK Loaded carbon model: {path}")
                elif model_type == 'temp' and temp_model is None:
                    temp_model = model
                    print(f"OK Loaded temperature model: {path}")
            except Exception as e:
                print(f"Warning: Failed to load {path}: {str(e).encode('ascii', 'ignore').decode()}")
    
    if carbon_model is None or temp_model is None:
        print("\n❌ Error: Could not load required models.")
        print("Ensure trained models exist in models/ directory.")
        sys.exit(1)
    
    return carbon_model, temp_model


def get_region_input(region_name: str) -> Dict:
    """Get and validate input for a single region."""
    print(f"\n--- Enter values for {region_name} ---")
    
    while True:
        try:
            cpu_usage = float(input(f"CPU usage (0-100%) for {region_name}: "))
            if 0 <= cpu_usage <= 100:
                break
            print("CPU usage must be between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            temperature = float(input(f"Current temperature (°C) for {region_name}: "))
            if -50 <= temperature <= 60:
                break
            print("Temperature must be between -50 and 60 °C.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            cost = float(input(f"Current cost ($/kWh) for {region_name}: "))
            if cost >= 0:
                break
            print("Cost cannot be negative.")
        except ValueError:
            print("Please enter a valid number.")
    
    return {
        'region': region_name,
        'cpu_usage': cpu_usage,
        'temperature': temperature,
        'cost': cost
    }


def get_user_input() -> Dict[str, Dict]:
    """Get and validate user input for all three regions."""
    print("\n" + "="*60)
    print("CARBON-AWARE WORKLOAD SCHEDULER - REGION INPUT")
    print("="*60)
    print("\nEnter separate values for each region below:")
    
    regions = ['US', 'India', 'Europe']
    all_inputs = {}
    
    for region in regions:
        all_inputs[region] = get_region_input(region)
    
    return all_inputs


def generate_features(input_data: Dict) -> Dict:
    """Generate derived and time-based features from input data."""
    now = datetime.datetime.now()
    
    workload = input_data['cpu_usage'] * 1.2
    energy_consumption = (0.6 * input_data['cpu_usage']) + (0.4 * input_data['temperature'])
    
    hour = now.hour
    day_of_week = now.weekday()
    
    default_lags = {
        'US': {'carbon': 310, 'temp': 20, 'energy': 42},
        'India': {'carbon': 340, 'temp': 28, 'energy': 45},
        'Europe': {'carbon': 245, 'temp': 15, 'energy': 38}
    }
    region_lag = default_lags[input_data['region']]
    
    region_encoding = {
        'US': {'region_Europe': 0, 'region_India': 0, 'region_US': 1},
        'India': {'region_Europe': 0, 'region_India': 1, 'region_US': 0},
        'Europe': {'region_Europe': 1, 'region_India': 0, 'region_US': 0}
    }
    
    features = {
        'cpu_usage': input_data['cpu_usage'],
        'workload': workload,
        'energy_consumption': energy_consumption,
        'cost': input_data['cost'],
        'hour': hour,
        'day_of_week': day_of_week,
        'carbon_lag1': region_lag['carbon'],
        'temp_lag1': region_lag['temp'],
        'energy_lag1': region_lag['energy'],
        **region_encoding[input_data['region']]
    }
    
    return features


def predict_values(carbon_model, temp_model, features: Dict) -> Tuple[float, float]:
    """Predict carbon intensity and temperature using loaded models."""
    feature_order = ['temperature', 'cpu_usage', 'workload', 'energy_consumption', 'cost']
    
    feature_vector = np.array([[
        features['cpu_usage'] * 0.28,
        features['cpu_usage'],
        features['workload'],
        features['energy_consumption'],
        features['cost']
    ]])
    
    predicted_carbon = carbon_model.predict(feature_vector)[0]
    predicted_temp = temp_model.predict(feature_vector)[0]
    
    predicted_carbon = max(100, min(500, predicted_carbon))
    predicted_temp = max(-20, min(50, predicted_temp))
    
    return predicted_carbon, predicted_temp


def compute_score(predicted_carbon: float, 
                  energy: float, 
                  cost: float, 
                  predicted_temp: float,
                  weights: Optional[Dict] = None) -> float:
    """Compute scheduling score using weighted formula."""
    if weights is None:
        weights = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.1}
    
    score = (weights['alpha'] * predicted_carbon +
             weights['beta'] * energy +
             weights['gamma'] * cost +
             weights['delta'] * predicted_temp)
    
    return score


def display_results(input_data: Dict, 
                    features: Dict, 
                    predicted_carbon: float, 
                    predicted_temp: float, 
                    score: float) -> None:
    """Display formatted results in console."""
    print("\n" + "="*60)
    print("PREDICTION & SCHEDULING RESULTS")
    print("="*60)
    
    print("\nINPUT VALUES:")
    print(f"  Region:          {input_data['region']}")
    print(f"  CPU Usage:       {input_data['cpu_usage']:.1f}%")
    print(f"  Temperature:     {input_data['temperature']:.2f} °C")
    print(f"  Cost:            ${input_data['cost']:.4f} /kWh")
    
    print("\nDERIVED FEATURES:")
    print("\nPREDICTIONS:")
    print("\nSCHEDULING SCORE:")
    print(f"  Composite Score:  {score:.4f}")
    print("\n  (Lower score = Better region for workload placement)")
    
    if score < 120:
        recommendation = "EXCELLENT - Highly recommended for workload placement"
    elif score < 150:
        recommendation = "GOOD - Suitable for workload placement"
    elif score < 180:
        recommendation = "FAIR - Acceptable but not optimal"
    else:
        recommendation = "POOR - Avoid placing workload here"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    print("="*60)


def compare_all_regions(input_data: Dict, carbon_model: object, temp_model: object) -> None:
    """Compare scores across all three regions for reference."""
    print("\n" + "="*60)
    print("CROSS-REGION COMPARISON")
    print("="*60)
    
    regions = ['US', 'India', 'Europe']
    region_scores = []
    
    for region in regions:
        region_input = input_data.copy()
        region_input['region'] = region
        features = generate_features(region_input)
        pred_carbon, pred_temp = predict_values(carbon_model, temp_model, features)
        score = compute_score(pred_carbon, features['energy_consumption'], 
                              region_input['cost'], pred_temp)
        region_scores.append((region, score, pred_carbon, pred_temp))
    
    region_scores.sort(key=lambda x: x[1])
    
    print(f"\n{'Rank':<6} {'Region':<10} {'Score':<12} {'Carbon (gCO2/kWh)':<20} {'Temp (C)':<10}")
    print("-" * 60)
    
    for i, (region, score, carbon, temp) in enumerate(region_scores, 1):
        marker = "*" if i == 1 else " "
        print(f"{marker} {i:<4} {region:<10} {score:<12.2f} {carbon:<20.2f} {temp:<10.2f}")
    
    print(f"\nBEST REGION: {region_scores[0][0]} with score {region_scores[0][1]:.2f}")
    print("="*60 + "\n")


def run_demo(carbon_model, temp_model):
    """Run demonstration with sample input values for all regions."""
    print("\n" + "="*60)
    print("DEMO MODE - Using sample input values for all regions")
    print("="*60)
    
    sample_inputs = {
        'US': {'region': 'US', 'cpu_usage': 82.0, 'temperature': 28.5, 'cost': 0.18},
        'India': {'region': 'India', 'cpu_usage': 65.0, 'temperature': 32.0, 'cost': 0.12},
        'Europe': {'region': 'Europe', 'cpu_usage': 75.0, 'temperature': 18.0, 'cost': 0.21}
    }
    
    print("\nSample Inputs:")
    for region, data in sample_inputs.items():
        print(f"  {region}: CPU={data['cpu_usage']}%, Temp={data['temperature']}°C, Cost=${data['cost']}/kWh")
    
    results = process_all_regions(sample_inputs, carbon_model, temp_model)
    display_comparison(results)


def process_all_regions(all_inputs: Dict[str, Dict], 
                        carbon_model, 
                        temp_model,
                        weights: Optional[Dict] = None) -> List[Dict]:
    """Process all three regions, compute predictions and scores."""
    results = []
    
    for region, input_data in all_inputs.items():
        features = generate_features(input_data)
        predicted_carbon, predicted_temp = predict_values(carbon_model, temp_model, features)
        score = compute_score(predicted_carbon, features['energy_consumption'], 
                              input_data['cost'], predicted_temp, weights)
        
        results.append({
            'region': region,
            'input': input_data,
            'features': features,
            'predicted_carbon': predicted_carbon,
            'predicted_temp': predicted_temp,
            'score': score
        })
    
    results.sort(key=lambda x: x['score'])
    return results


def display_comparison(results: List[Dict]) -> None:
    """Display formatted comparison table for all regions."""
    print("\n" + "="*90)
    print("REGION COMPARISON RESULTS")
    print("="*90)
    
    header = f"{'Rank':<5} {'Region':<8} {'CPU %':<7} {'Temp °C':<9} {'Cost $':<9} {'Energy':<9} {'Carbon':<10} {'Pred Temp':<11} {'Score':<10}"
    print(header)
    print("-" * 90)
    
    for i, res in enumerate(results, 1):
        marker = "*" if i == 1 else " "
        print(f"{marker} {i:<4} {res['region']:<8} {res['input']['cpu_usage']:<7.1f} "
              f"{res['input']['temperature']:<9.2f} {res['input']['cost']:<9.4f} "
              f"{res['features']['energy_consumption']:<9.2f} {res['predicted_carbon']:<10.2f} "
              f"{res['predicted_temp']:<11.2f} {res['score']:<10.2f}")
    
    print("="*90)
    print(f"\nOPTIMAL REGION: {results[0]['region']} with score {results[0]['score']:.2f}")
    print(f"   (Lowest score = Best region for workload placement)")
    print("="*90)


def main():
    carbon_model, temp_model = load_models()
    
    import sys
    try:
        if not sys.stdin.isatty():
            print("\nRunning in non-interactive mode - launching demo...")
            run_demo(carbon_model, temp_model)
            return
    except:
        print("\nRunning in non-interactive mode - launching demo...")
        run_demo(carbon_model, temp_model)
        return
    
    while True:
        all_inputs = get_user_input()
        results = process_all_regions(all_inputs, carbon_model, temp_model)
        display_comparison(results)
        
        while True:
            choice = input("\nTest another set of inputs? (y/n): ").strip().lower()
            if choice in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'.")
        
        if choice == 'n':
            print("\nThank you for using the Carbon-Aware Scheduler!")
            break


if __name__ == "__main__":
    main()
