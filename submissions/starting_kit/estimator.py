import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def get_estimator():
    """Build a model pipeline for battery degradation prediction."""
    
    # Define feature groups
    numeric_features = [
        'charge_capacity', 'energy_efficiency', 'voltage_drop_rate',
        'avg_temperature', 'max_voltage', 'min_voltage', 'charge_time',
        'discharge_time', 'cycle_number'
    ]
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features)
        ]
    )
    
    # Create full pipeline
    pipeline = make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    )
    
    return pipeline