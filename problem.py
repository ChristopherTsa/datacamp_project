import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

problem_title = 'Predicting eVTOL Battery Degradation'
_target_column_name = 'discharge_capacity'
_prediction_label_names = [0]  # Regression problem
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()

# Define score types for regression
score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
    rw.score_types.RelativeRMSE(name='rel_rmse', precision=3),
    R2(name='r2', precision=3),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)

def _preprocess_data(X_df):
    """Preprocess features according to our extraction pipeline."""
    # Handle missing data by filling with the mean value of each column
    X_df.fillna(X_df.mean(), inplace=True)
    
    # Replace any infinite values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df.fillna(X_df.mean(), inplace=True)
    
    # Identify columns to exclude from scaling
    columns_to_exclude = ['cycleNumber'] if 'cycleNumber' in X_df.columns else []
    
    # Make sure all columns to be scaled are numeric
    numeric_cols = X_df.select_dtypes(include=np.number).columns
    features_to_scale = X_df[numeric_cols].drop(columns=columns_to_exclude, errors='ignore')
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_scale)
    
    # Convert back to DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
    
    # Add back excluded columns
    for col in columns_to_exclude:
        if col in X_df.columns:
            scaled_features_df[col] = X_df[col].values
            
    return scaled_features_df

# Data loading functions
def _read_data(path, filename='battery_features.csv'):
    """Read and prepare training and testing data."""
    data = pd.read_csv(os.path.join(path, 'data', filename))
    y_array = data[_target_column_name].values
    X_df = data.drop(columns=[_target_column_name])
    
    # Apply preprocessing
    X_df = _preprocess_data(X_df)
    
    return X_df, y_array

def get_train_data(path='.'):
    """Get training data."""
    return _read_data(path, filename='battery_features_train.csv')

def get_test_data(path='.'):
    """Get testing data."""
    return _read_data(path, filename='battery_features_test.csv')