import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def process_battery_data(data, verbose=True):
    # Function to extract features from the dataset
    def extract_features(df):
        features = {}
        
        # Min current (mA)
        features['min_current'] = df['I_mA'].min()
        
        # Min voltage (V)
        features['min_voltage'] = df['Ecell_V'].min()
        
        # Max temperature (°C)
        features['max_temperature'] = df['Temperature__C'].max()
        
        return features
    
    # Apply the feature extraction function to each cycle
    unique_cycles = data['cycleNumber'].unique()
    if verbose:
        print(f"Found {len(unique_cycles)} unique cycles")
    
    cycle_features = []
    for cycle in unique_cycles:
        if verbose and cycle % 100 == 0:
            print(f"Extracting features for cycle {cycle}")
        cycle_data = data[data['cycleNumber'] == cycle]
        features = extract_features(cycle_data)
        features['cycleNumber'] = cycle
        cycle_features.append(features)

    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(cycle_features)

    # Merge the features back to the original data
    features_df = data.merge(features_df, on='cycleNumber', how='left')
    features_df = features_df[['time_s', 'min_current', 'min_voltage', 'max_temperature', 'cycleNumber']]

    return features_df


# Create a transformer using FunctionTransformer
battery_transformer = FunctionTransformer(
    func=process_battery_data,
    kw_args={'verbose': False},
    validate=False
)

# Create complete pipeline
pipeline = Pipeline([
    ('feature_extractor', battery_transformer),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, verbose=0, n_jobs=-1))
])

def get_estimator():
    return pipeline