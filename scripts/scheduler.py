import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_directories():
    os.makedirs('results/plots', exist_ok=True)

def load_prediction_data(prediction_path='results/predictions.csv', dataset_path='dataset/final_dataset.csv'):
    predictions = pd.read_csv(prediction_path)
    predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
    
    # Load historical data to get base energy and cost patterns
    historical = pd.read_csv(dataset_path)
    historical['timestamp'] = pd.to_datetime(historical['timestamp'])
    
    # Calculate average energy consumption and cost per region
    region_metrics = historical.groupby('region').agg({
        'energy_consumption': 'mean',
        'cost': 'mean'
    }).reset_index()
    
    return predictions, region_metrics

def calculate_scheduling_score(row, weights):
    score = (
        weights['alpha'] * row['predicted_carbon_intensity'] +
        weights['beta'] * row['energy_consumption'] +
        weights['gamma'] * row['cost'] +
        weights['delta'] * abs(row['predicted_temperature'])
    )
    return score

def generate_scheduling_decisions(predictions, region_metrics, weights):
    # Merge regional metrics with predictions
    df = predictions.merge(region_metrics, on='region', how='left')
    
    # Calculate scheduling score for each region at each timestamp
    df['score'] = df.apply(lambda x: calculate_scheduling_score(x, weights), axis=1)
    
    # Find optimal region with minimum score for each timestamp
    scheduling_results = []
    
    for timestamp in df['timestamp'].unique():
        timestamp_data = df[df['timestamp'] == timestamp]
        
        if len(timestamp_data) == 0:
            continue
            
        # Find region with lowest score
        optimal_idx = timestamp_data['score'].idxmin()
        optimal_row = timestamp_data.loc[optimal_idx]
        
        scheduling_results.append({
            'timestamp': timestamp,
            'selected_region': optimal_row['region'],
            'score': round(optimal_row['score'], 4),
            'carbon_intensity': round(optimal_row['predicted_carbon_intensity'], 2),
            'energy_consumption': round(optimal_row['energy_consumption'], 4),
            'cost': round(optimal_row['cost'], 4),
            'temperature': round(optimal_row['predicted_temperature'], 4)
        })
    
    results_df = pd.DataFrame(scheduling_results)
    return results_df, df

def plot_scheduling_results(scheduling_df, full_scores_df):
    # Plot 1: Selected regions over time
    plt.figure(figsize=(12, 6))
    region_counts = scheduling_df['selected_region'].value_counts()
    plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette('viridis', 3))
    plt.title('Workload Distribution Across Regions')
    plt.savefig('results/plots/scheduling_distribution.png')
    plt.close()
    
    # Plot 2: Score comparison over time for all regions
    plt.figure(figsize=(14, 7))
    for region in full_scores_df['region'].unique():
        region_data = full_scores_df[full_scores_df['region'] == region]
        plt.plot(region_data['timestamp'], region_data['score'], marker='o', label=region)
    
    plt.title('Region Scheduling Scores Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Scheduling Score (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/score_comparison.png')
    plt.close()
    
    # Plot 3: Factor contribution breakdown
    factors = ['carbon_intensity', 'energy_consumption', 'cost', 'temperature']
    factor_names = ['Carbon Intensity', 'Energy Consumption', 'Cost', 'Temperature']
    
    avg_values = scheduling_df[factors].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(factor_names, avg_values.values, color=sns.color_palette('Set2', 4))
    plt.title('Average Factor Values in Selected Regions')
    plt.ylabel('Average Value')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/factor_contribution.png')
    plt.close()

def main():
    print("=== Phase 6: Carbon-Aware Workload Scheduling System ===")
    
    create_directories()
    
    # Configurable weights for scoring function
    weights = {
        'alpha': 0.4,  # Carbon intensity weight
        'beta': 0.3,   # Energy consumption weight
        'gamma': 0.2,  # Cost weight
        'delta': 0.1   # Temperature weight
    }
    
    print("\nScheduling Weights Configuration:")
    for key, value in weights.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading prediction data and regional metrics...")
    predictions, region_metrics = load_prediction_data()
    print(f"Loaded {len(predictions)} predictions for {len(region_metrics)} regions")
    
    # Generate scheduling decisions
    print("\nCalculating scheduling scores and selecting optimal regions...")
    scheduling_results, full_scores = generate_scheduling_decisions(predictions, region_metrics, weights)
    
    # Save results
    scheduling_results.to_csv('results/scheduling_results.csv', index=False)
    full_scores.to_csv('results/full_region_scores.csv', index=False)
    
    # Generate visualizations
    print("\nGenerating scheduling visualizations...")
    plot_scheduling_results(scheduling_results, full_scores)
    
    print("\n" + "="*70)
    print("Scheduling Complete!")
    print(f"Generated {len(scheduling_results)} scheduling decisions")
    print(f"Results saved to results/scheduling_results.csv")
    print(f"Full region scores saved to results/full_region_scores.csv")
    print(f"Plots saved to results/plots/")
    
    print("\nScheduling Summary:")
    print(scheduling_results.to_string(index=False))
    
    print("\nRegion Selection Distribution:")
    print(scheduling_results['selected_region'].value_counts().to_string())

if __name__ == "__main__":
    main()