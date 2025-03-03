import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path where the raw battery data files are stored
# You might need to adjust this path to where your raw data files are located
RAW_DATA_PATH = "raw_data"
OUTPUT_DATA_PATH = "data"

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_features_from_battery_file(file_path, battery_id):
    """
    Extract features from a battery cycle data file.
    
    Parameters
    ----------
    file_path : str
        Path to the battery data file
    battery_id : str
        Identifier for the battery
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted features for each cycle
    """
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Initialize list to store cycle data
    cycle_features = []
    
    # Get unique cycle numbers
    cycle_numbers = df['cycleNumber'].unique()
    
    for cycle in cycle_numbers:
        # Get data for this cycle
        cycle_data = df[df['cycleNumber'] == cycle]
        
        # Skip cycles with very few data points (likely incomplete)
        if len(cycle_data) < 10:
            continue
            
        # Calculate features for this cycle
        
        # Voltage features
        max_voltage = cycle_data['Ecell_V'].max()
        min_voltage = cycle_data['Ecell_V'].min()
        mean_voltage = cycle_data['Ecell_V'].mean()
        
        # Current features
        max_current = cycle_data['I_mA'].max()
        min_current = cycle_data['I_mA'].min()
        mean_current = cycle_data['I_mA'].mean()
        
        # Energy features
        charge_energy = cycle_data['EnergyCharge_W_h'].max()
        discharge_energy = cycle_data['EnergyDischarge_W_h'].max()
        
        # Capacity features
        charge_capacity = cycle_data['QCharge_mA_h'].max()
        discharge_capacity = cycle_data['QDischarge_mA_h'].max()
        
        # Temperature features
        avg_temperature = cycle_data['Temperature__C'].mean()
        max_temperature = cycle_data['Temperature__C'].max()
        
        # Time features
        # Find charging and discharging segments
        charging_data = cycle_data[cycle_data['I_mA'] > 0]
        discharging_data = cycle_data[cycle_data['I_mA'] < 0]
        
        # Calculate charging and discharging times (in seconds)
        if len(charging_data) > 0:
            charge_time = charging_data['time_s'].max() - charging_data['time_s'].min()
        else:
            charge_time = 0
            
        if len(discharging_data) > 0:
            discharge_time = discharging_data['time_s'].max() - discharging_data['time_s'].min()
        else:
            discharge_time = 0
        
        # Calculate energy efficiency
        energy_efficiency = discharge_energy / charge_energy if charge_energy > 0 else 0
        
        # Calculate voltage drop rate during discharge
        if len(discharging_data) > 10:
            voltage_drop = discharging_data['Ecell_V'].max() - discharging_data['Ecell_V'].min()
            discharge_duration = discharging_data['time_s'].max() - discharging_data['time_s'].min()
            voltage_drop_rate = voltage_drop / discharge_duration if discharge_duration > 0 else 0
        else:
            voltage_drop_rate = 0
        
        # Append features for this cycle
        cycle_features.append({
            'battery_id': battery_id,
            'cycle_number': cycle,
            'charge_capacity': charge_capacity,
            'discharge_capacity': discharge_capacity,
            'charge_time': charge_time,
            'discharge_time': discharge_time,
            'energy_efficiency': energy_efficiency,
            'max_voltage': max_voltage,
            'min_voltage': min_voltage,
            'avg_temperature': avg_temperature,
            'max_temperature': max_temperature,
            'voltage_drop_rate': voltage_drop_rate,
            'charge_energy': charge_energy,
            'discharge_energy': discharge_energy,
            'mean_voltage': mean_voltage,
            'max_current': max_current,
            'min_current': min_current,
            'mean_current': mean_current
        })
    
    # Create DataFrame from the extracted features
    return pd.DataFrame(cycle_features)


def process_all_battery_files():
    """
    Process all battery files and combine into one features dataset.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data directory '{RAW_DATA_PATH}' not found.")
        print(f"Please make sure to create this directory and place the battery files (VAH*.csv) in it.")
        return None
    
    # Get all battery files
    battery_files = [f for f in os.listdir(RAW_DATA_PATH) if f.startswith('VAH') and f.endswith('.csv')]
    
    if not battery_files:
        print(f"Error: No battery data files found in '{RAW_DATA_PATH}'.")
        print("Please place the VAH*.csv files in this directory.")
        return None
    
    print(f"Found {len(battery_files)} battery files.")
    
    # Process each battery file
    all_features = []
    
    for file_name in battery_files:
        battery_id = file_name.split('.')[0]  # Get battery ID from filename (e.g., VAH01)
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        
        try:
            battery_features = extract_features_from_battery_file(file_path, battery_id)
            all_features.append(battery_features)
            print(f"Processed {file_name}: {len(battery_features)} cycles.")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Combine all features into one DataFrame
    if all_features:
        all_features_df = pd.concat(all_features, ignore_index=True)
        print(f"Total dataset: {len(all_features_df)} cycles from {len(battery_files)} batteries.")
        return all_features_df
    else:
        print("No features extracted.")
        return None


def split_and_save_data(features_df, test_size=0.2, random_state=42):
    """
    Split the features dataset into training and testing sets and save them.
    """
    # Create output directory
    ensure_dir(OUTPUT_DATA_PATH)
    
    # Split data by batteries to avoid data leakage
    batteries = features_df['battery_id'].unique()
    train_batteries, test_batteries = train_test_split(
        batteries, test_size=test_size, random_state=random_state
    )
    
    # Create train and test sets
    train_df = features_df[features_df['battery_id'].isin(train_batteries)]
    test_df = features_df[features_df['battery_id'].isin(test_batteries)]
    
    print(f"Training set: {len(train_df)} cycles from {len(train_batteries)} batteries.")
    print(f"Testing set: {len(test_df)} cycles from {len(test_batteries)} batteries.")
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DATA_PATH, 'battery_features_train.csv')
    test_path = os.path.join(OUTPUT_DATA_PATH, 'battery_features_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")


if __name__ == "__main__":
    print("Starting battery data preprocessing...")
    
    # Process all battery files
    features_df = process_all_battery_files()
    
    if features_df is not None:
        # Check if there are null values and handle them
        null_counts = features_df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"Warning: Found {null_counts.sum()} null values:")
            print(null_counts[null_counts > 0])
            
            # Fill null values with appropriate methods
            features_df = features_df.fillna({
                'discharge_capacity': features_df['discharge_capacity'].median(),
                'charge_capacity': features_df['charge_capacity'].median(),
                'energy_efficiency': features_df['energy_efficiency'].median(),
                'voltage_drop_rate': 0
            })
            
        # Split and save the data
        split_and_save_data(features_df)
        
        print("Data preprocessing completed successfully!")
    else:
        print("Failed to process battery data.")
        print("Please check that the raw data files exist and are accessible.")
