import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks  # <-- added import
import os
import requests
import zipfile
import re
import shutil
from pathlib import Path

def download_and_extract():
    # URL of the dataset
    url = "https://kilthub.cmu.edu/ndownloader/articles/14226830/versions/3"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"Downloading data from {url}...")  # progress print
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Determine filename from header or use default name
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        filename_match = re.search(r'filename=(.+)', content_disposition)
        if filename_match:
            filename = filename_match.group(1).strip('"')
        else:
            filename = "data.zip"
    else:
        filename = "data.zip"
    
    file_path = data_dir / filename
    
    # Save downloaded file
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded file saved to {file_path}")
    
    # Extract zip file directly to data directory
    print(f"Extracting {filename} directly to data folder...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if not member.endswith('/'):
                member_filename = os.path.basename(member)
                source = zip_ref.open(member)
                target = open(os.path.join(data_dir, member_filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
    csv_files = list(data_dir.glob('*.csv'))
    txt_files = list(data_dir.glob('*.txt'))
    print(f"Extraction complete. Found {len(csv_files)} CSV files and {len(txt_files)} TXT files in {data_dir}")
    print(f"The original zip file has been kept at {file_path}")
    print("Data download and extraction completed")

def process_file(file_path):
    print(f"Starting processing for {file_path}")
    # Load dataset
    data_cleaned = pd.read_csv(file_path)
    print("Dataset loaded")
    
    data_cleaned['cycle_diff'] = data_cleaned['cycleNumber'].diff()
    
    # Identify cycle drops and rises
    drops = data_cleaned[data_cleaned['cycle_diff'] < -1].copy()
    rises = data_cleaned[data_cleaned['cycle_diff'] > 1].copy()
    print(f"Found {len(drops)} cycle drops and {len(rises)} cycle rises")
    
    # Mark test cycles based on transitions
    test_cycles = [0]
    for idx in drops.index:
        transition_cycle = data_cleaned.loc[idx, 'cycleNumber']
        if transition_cycle not in test_cycles:
            test_cycles.append(transition_cycle)
        if idx > 0:
            prev_cycle = data_cleaned.loc[idx - 1, 'cycleNumber']
            if prev_cycle not in test_cycles:
                test_cycles.append(prev_cycle)
        rise_points = rises[rises.index > idx]
        if not rise_points.empty:
            rise_idx = rise_points.index[0]
            rise_cycle = data_cleaned.loc[rise_idx, 'cycleNumber']
            if rise_cycle not in test_cycles:
                test_cycles.append(rise_cycle)
            next_cycle = rise_cycle + 1
            if next_cycle not in test_cycles:
                test_cycles.append(next_cycle)
    print(f"Marked test cycles: {test_cycles}")
    data_cleaned['is_test_cycle'] = data_cleaned['cycleNumber'].isin(test_cycles)
    data_cleaned = data_cleaned.drop('cycle_diff', axis=1)
    
    # Separate regular (non-test) cycles for continuous cycle correction
    data_regular = data_cleaned[~data_cleaned['is_test_cycle']].copy().reset_index(drop=True)
    data_regular['cycle_diff'] = data_regular['cycleNumber'].diff()
    data_regular['corrected_cycle'] = data_regular['cycleNumber'].copy()
    
    cycle_correction = 0
    drops_found = 0
    print("Starting cycle correction for non-test cycles")
    for i in range(1, len(data_regular)):
        if data_regular.loc[i, 'cycle_diff'] < -1:
            drop_correction = data_regular.loc[i-1, 'cycleNumber'] - data_regular.loc[i, 'cycleNumber']
            cycle_correction += drop_correction
            drops_found += 1
            print(f"Cycle drop at index {i}: from {data_regular.loc[i-1, 'cycleNumber']} to {data_regular.loc[i, 'cycleNumber']}, correction: {drop_correction}")
        if cycle_correction > 0:
            data_regular.loc[i, 'corrected_cycle'] = data_regular.loc[i, 'cycleNumber'] + cycle_correction
    print(f"Cycle correction done: total correction {cycle_correction}, drops found {drops_found}")
    
    # Update the original data: for non-test cycles, update cycle numbers; then interpolate to smooth changes
    data_cleaned.loc[~data_cleaned['is_test_cycle'], 'cycleNumber'] = data_regular['corrected_cycle'].values
    data_cleaned['cycleNumber'] = data_cleaned['cycleNumber'].ffill().bfill().astype(int)
    
    # Create a test dataset from the cleaned data
    data_test = data_cleaned[data_cleaned['is_test_cycle']].copy()
    
    print("Starting discharge peak detection")
    # Find peaks for discharge cycles in 'QDischarge_mA_h'
    peaks_indices, _ = find_peaks(data_test['QDischarge_mA_h'], height=2400, distance=10000)
    
    peak_times = data_test['time_s'].iloc[peaks_indices].values
    peak_values = data_test['QDischarge_mA_h'].iloc[peaks_indices].values
    
    # Ensure a 'discharge_peak_value' column exists in the full dataset
    if 'discharge_peak_value' not in data_cleaned.columns:
        data_cleaned['discharge_peak_value'] = np.nan
    
    # For each detected peak, find the closest index in the full data and assign the peak value
    for time, value in zip(peak_times, peak_values):
        closest_idx = (data_cleaned['time_s'] - time).abs().idxmin()
        data_cleaned.loc[closest_idx, 'discharge_peak_value'] = value
    data_cleaned['discharge_peak_value'] = data_cleaned['discharge_peak_value'].interpolate(method='linear').bfill().ffill()
    print("Discharge peak detection and interpolation completed")
    
    return data_cleaned

if __name__ == "__main__":
    #download_and_extract()
    files = ['data/VAH01.csv', 'data/VAH17.csv']
    for f in files:
        cleaned_data = process_file(f)
        out_file = f.replace('.csv', '.pkl')
        with open(out_file, 'wb') as file:
            pickle.dump(cleaned_data, file)
        print(f"Saved cleaned data to {out_file}")
