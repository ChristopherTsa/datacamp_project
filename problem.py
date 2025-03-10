import rampwf as rw
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

problem_title = 'eVTOL Battery Capacity Prediction Challenge'

# This is a regression problem predicting discharge_peak_value
Predictions = rw.prediction_types.make_regression()

# Using the standard regression workflow
workflow = rw.workflows.Estimator()

# Appropriate metrics for battery capacity prediction
score_types = [
    rw.score_types.RMSE(name='rmse', precision=3)
]


def get_cv(X, y):
    cv = TimeSeriesSplit(n_splits=5, test_size=int(0.2 * len(X)))
    return cv.split(X, y)


def load_data(path='.', file='VAH01.pkl'):
    path = Path(path) / "data"
    
    with open(path / file, 'rb') as f:
        data = pickle.load(f)
    
    y = data['discharge_peak_value']
    X = data.drop('discharge_peak_value', axis=1)
    
    return X, y


def get_train_data(path='.'):
    file = 'VAH01.pkl'
    return load_data(path, file)


def get_test_data(path='.'):
    file = 'VAH17.pkl'
    return load_data(path, file)