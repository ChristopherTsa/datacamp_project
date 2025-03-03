import numpy as np
from rampwf.score_types.base import BaseScoreType


class R2(BaseScoreType):
    """R^2 (coefficient of determination) regression score.
    
    This class implements the R^2 score, also known as the coefficient of determination.
    R^2 represents the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).
    """
    is_lower_the_better = False
    minimum = -np.inf
    maximum = 1.0

    def __init__(self, name='r2', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """Calculate R^2 score.
        
        Parameters
        ----------
        y_true : array, shape = [n_samples]
            True values
        y_pred : array, shape = [n_samples]
            Predicted values
            
        Returns
        -------
        float
            R^2 score
        """
        # Calculate the mean of true values
        y_mean = np.mean(y_true)
        
        # Calculate the total sum of squares
        ss_total = np.sum((y_true - y_mean) ** 2)
        
        # Calculate the residual sum of squares
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        # Calculate and return R^2
        if ss_total == 0:
            # Avoid division by zero
            return 0
        else:
            return 1 - (ss_residual / ss_total)
