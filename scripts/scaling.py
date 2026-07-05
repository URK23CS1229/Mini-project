#!/usr/bin/env python3
"""
Manual workload scaling implementation for carbon-aware federated scheduling.
Computes region scores and distributes workload using inverse normalization.
"""

import os
import csv
import argparse
from typing import List, Dict, Tuple


def load_region_data(input_file: str) -> Tuple[List[str], Dict[str, List[Dict]]]:
    """
    Load region data from input CSV file.
    Expects columns: timestamp, region, predicted_carbon_intensity, 
                     predicted_temperature, energy_consumption, cost
    """
    timestamps = []
    data_by_timestamp = {}
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row['timestamp']
            if timestamp not in data_by_timestamp:
                data_by_timestamp[timestamp] = []
                timestamps.append(timestamp)
            
            data_by_timestamp[timestamp].append({
                'region': row['region'],
                'carbon': float(row['predicted_carbon_intensity']),
                'temperature': float(row['predicted_temperature']),
                'energy': float(row['energy_consumption']),
                'cost': float(row['cost'])
            })
    
    return timestamps, data_by_timestamp


def compute_scores(regions: List[Dict], 
                   alpha: float = 0.4, 
                   beta: float = 0.3, 
                   gamma: float = 0.2, 
                   delta: float = 0.1) -> List[Tuple[str, float]]:
    """
    Compute scheduling score for each region using weighted formula.
    Score = α×Carbon + β×Energy + γ×Cost + δ×Temperature
    Lower score is better.
    """
    scores = []
    for region in regions:
        score = (alpha * region['carbon'] + 
                 beta * region['energy'] + 
                 gamma * region['cost'] + 
                 delta * region['temperature'])
        scores.append((region['region'], score))
    
    return scores


def inverse_normalize_scores(scores: List[Tuple[str, float]], 
                             smoothing: float = 0.5) -> List[Tuple[str, float, float]]:
    """
    Convert scores to workload percentages using inverse normalization.
    Handles edge cases: zero scores, identical scores, extreme values.
    
    Method:
    1. Normalize scores to 0-1 range
    2. Apply smoothing parameter to control distribution sharpness
    3. Compute inverse of adjusted scores
    4. Normalize inverses to sum to 100%
    5. Ensure total sums exactly to 100%
    
    Smoothing: 0 = all equal, 1 = original sharp inverse, >1 = more extreme
    """
    if not scores:
        return []
    
    epsilon = 1e-10
    score_values = [s + epsilon for (_, s) in scores]
    
    min_score = min(score_values)
    max_score = max(score_values)
    score_range = max_score - min_score + epsilon
    
    normalized_scores = [(s - min_score) / score_range for s in score_values]
    adjusted_scores = [(ns * smoothing) + (1 - smoothing) + epsilon for ns in normalized_scores]
    
    inverses = [1.0 / s for s in adjusted_scores]
    total_inverse = sum(inverses)
    
    if total_inverse == 0:
        percentage = 100.0 / len(scores)
        return [(r, s, percentage) for (r, s) in scores]
    
    percentages = [inv / total_inverse * 100.0 for inv in inverses]
    
    total = sum(percentages)
    diff = 100.0 - total
    percentages[0] += diff
    
    return [(r, s, p) for (r, s), p in zip(scores, percentages)]


def process_all_timestamps(timestamps: List[str], 
                           data_by_timestamp: Dict[str, List[Dict]],
                           weights: Dict[str, float],
                           smoothing: float = 0.5) -> List[Dict]:
    """
    Process all timestamps to compute scores and workload percentages.
    """
    results = []
    
    for ts in timestamps:
        regions = data_by_timestamp[ts]
        scores = compute_scores(
            regions,
            alpha=weights['alpha'],
            beta=weights['beta'],
            gamma=weights['gamma'],
            delta=weights['delta']
        )
        scaled_results = inverse_normalize_scores(scores, smoothing)
        
        for region, score, percentage in scaled_results:
            results.append({
                'timestamp': ts,
                'region': region,
                'score': score,
                'workload_percentage': percentage
            })
    
    return results


def save_results(results: List[Dict], output_file: str) -> None:
    """
    Save scaling results to CSV file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fieldnames = ['timestamp', 'region', 'score', 'workload_percentage']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            writer.writerow({
                'timestamp': row['timestamp'],
                'region': row['region'],
                'score': f"{row['score']:.6f}",
                'workload_percentage': f"{row['workload_percentage']:.4f}"
            })


def main():
    parser = argparse.ArgumentParser(
        description='Manual workload scaling for carbon-aware scheduling'
    )
    parser.add_argument('--input', 
                        default='results/full_region_scores.csv',
                        help='Input CSV file with predictions and metrics')
    parser.add_argument('--output', 
                        default='results/scaling_results.csv',
                        help='Output CSV file for scaling results')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Weight for carbon intensity (default: 0.4)')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for energy consumption (default: 0.3)')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Weight for cost (default: 0.2)')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='Weight for temperature (default: 0.1)')
    parser.add_argument('--smoothing', type=float, default=0.5,
                        help='Smoothing factor (0=equal distribution, 1=sharp inverse, default: 0.5)')
    
    args = parser.parse_args()
    
    total_weight = args.alpha + args.beta + args.gamma + args.delta
    if total_weight <= 0:
        raise ValueError("Sum of weights must be greater than zero")
    
    weights = {
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'delta': args.delta
    }
    
    timestamps, data_by_timestamp = load_region_data(args.input)
    results = process_all_timestamps(timestamps, data_by_timestamp, weights, args.smoothing)
    save_results(results, args.output)
    
    print(f"Processed {len(timestamps)} timestamps")
    print(f"Results saved to {args.output}")
    
    print("\nSample results:")
    for i, res in enumerate(results[:9]):
        if i % 3 == 0:
            print(f"\nTimestamp: {res['timestamp']}")
        print(f"  {res['region']}: Score={res['score']:.4f}, Workload={res['workload_percentage']:.2f}%")


if __name__ == "__main__":
    main()
