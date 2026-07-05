import pandas as pd
import numpy as np
import os

def generate_carbon_aware_dataset(input_csv='weatherHistory.csv', output_csv='dataset/final_dataset.csv'):
    """
    Generate synthetic dataset for carbon-aware federated workload optimization
    with cloud-based scheduling for geo-distributed data centers.
    
    Args:
        input_csv (str): Path to input weather dataset CSV file
        output_csv (str): Path where final dataset will be saved
        
    Returns:
        pandas.DataFrame: Final dataset with columns:
            timestamp, region, carbon_intensity, temperature, cpu_usage, 
            workload, energy_consumption, cost
    """
    
    # 1. Load the weather dataset CSV
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found")
    
    df = pd.read_csv(input_csv)
    
    # 2. Clean and format timestamp and temperature columns
    df = df.rename(columns={'Formatted Date': 'timestamp', 'Temperature (C)': 'temperature'})
    # Handle mixed timezones by converting to UTC and making naive (removing timezone info)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    df = df[['timestamp', 'temperature']]  # Remove unnecessary columns
    df = df.dropna()  # Ensure no missing values
    
    # 3. Generate realistic carbon intensity values
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Extract hour from timestamp
    df['hour'] = df['timestamp'].dt.hour
    
    # Base carbon intensity between 200-500 gCO2/kWh
    base_intensity = np.random.uniform(200, 500, size=len(df))
    
    # Adjust for time of day: higher during peak hours 9 AM-6 PM (9-18)
    peak_factor = np.where((df['hour'] >= 9) & (df['hour'] <= 18), 100, 0)
    
    # Adjust for temperature: higher temp -> slightly higher carbon
    # For each degree above 20°C, add 2; below 20°C, subtract 2 (clipped to avoid extreme values)
    temp_factor = (df['temperature'] - 20) * 2
    temp_factor = np.clip(temp_factor, -50, 50)
    
    # Combine factors and add some random noise
    carbon_intensity = base_intensity + peak_factor + temp_factor + np.random.normal(0, 10, size=len(df))
    
    # Ensure carbon intensity stays within realistic bounds (200-500) after adjustments
    carbon_intensity = np.clip(carbon_intensity, 200, 500)
    
    df['carbon_intensity'] = carbon_intensity.round(2)
    
    # Drop the helper column 'hour'
    df = df.drop(columns=['hour'])
    
    # 4. Create region-wise datasets (India, USA, Europe)
    regions = ['India', 'USA', 'Europe']
    region_factors = {'India': 1.1, 'USA': 1.0, 'Europe': 0.8}
    
    # List to hold dataframes for each region
    region_dfs = []
    
    for region in regions:
        # Duplicate the dataframe for each region
        region_df = df.copy()
        region_df['region'] = region
        
        # Adjust carbon intensity by regional factor
        region_df['carbon_intensity'] = (region_df['carbon_intensity'] * region_factors[region]).round(2)
        
        # 5. Simulate additional features for each region
        # CPU usage: 60-90 during daytime (9AM-6PM), 30-60 during night
        region_df['hour'] = region_df['timestamp'].dt.hour
        # Daytime condition
        daytime = (region_df['hour'] >= 9) & (region_df['hour'] <= 18)
        # Generate CPU usage based on time of day
        cpu_usage = np.where(daytime,
                             np.random.uniform(60, 90, size=len(region_df)),
                             np.random.uniform(30, 60, size=len(region_df)))
        region_df['cpu_usage'] = cpu_usage.round(2)
        
        # Workload: cpu_usage × random factor (1.2–2.0)
        workload_factor = np.random.uniform(1.2, 2.0, size=len(region_df))
        region_df['workload'] = (region_df['cpu_usage'] * workload_factor).round(2)
        
        # Energy consumption: 0.6 × cpu_usage + 0.4 × temperature
        region_df['energy_consumption'] = (0.6 * region_df['cpu_usage'] + 0.4 * region_df['temperature']).round(2)
        
        # Cost: region-specific ranges
        if region == 'India':
            cost_range = (0.10, 0.15)
        elif region == 'USA':
            cost_range = (0.18, 0.25)
        else:  # Europe
            cost_range = (0.15, 0.20)
        region_df['cost'] = np.random.uniform(cost_range[0], cost_range[1], size=len(region_df)).round(2)
        
        # Drop the helper column 'hour'
        region_df = region_df.drop(columns=['hour'])
        
        region_dfs.append(region_df)
    
    # 6. Combine all regions into a single dataset
    final_df = pd.concat(region_dfs, ignore_index=True)
    
    # Reorder columns as per example
    final_df = final_df[['timestamp', 'region', 'carbon_intensity', 'temperature', 
                         'cpu_usage', 'workload', 'energy_consumption', 'cost']]
    
    # 7. Save the final dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"Dataset generated successfully as {output_csv}")
    print(f"Shape of final dataset: {final_df.shape}")
    print("\nFirst few rows:")
    print(final_df.head())
    
    return final_df

if __name__ == "__main__":
    generate_carbon_aware_dataset()