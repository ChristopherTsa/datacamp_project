import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

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
    rw.score_types.R2(name='r2', precision=3),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)

# Data loading functions
def _read_data(path, filename='battery_features.csv'):
    """Read and prepare training and testing data."""
    data = pd.read_csv(os.path.join(path, 'data', filename))
    y_array = data[_target_column_name].values
    X_df = data.drop(columns=[_target_column_name])
    return X_df, y_array

def get_train_data(path='.'):
    """Get training data."""
    return _read_data(path, filename='battery_features_train.csv')

def get_test_data(path='.'):
    """Get testing data."""
    return _read_data(path, filename='battery_features_test.csv')