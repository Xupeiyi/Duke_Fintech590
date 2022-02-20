import numpy as np


def frobenius_norm(matrix):
    """
    The Frobenius Norm of matrix A is 
        ||𝐴|| = \sqrt{\sum_{i=1}^n \sum_{j=1}^n a_{ij}^2}.
    """
    return np.sqrt(np.square(matrix).sum())


# =================================
# Expotentially Weighted Covariance
# =================================


def exponential_weights(lambda_: float, nlags: int) -> np.array:
    """
    Calculate the weights from T = t-1 to t-nlags if current period is T = t.
    """
    weights = np.array([(1-lambda_) * (lambda_**(lag-1)) for lag in range(1, nlags+1)])
    weights /= weights.sum()  # normalized weights
    return weights


def cal_ewcov(data: np.matrix, lambda_: float):
    """
    Calculate the exponentially weighted covariance of a dataset matrix.
    Equation:
    
        \sigma_t^2 = (1 − \lambda) \sum_{i=1}^{\infty}{\lambda^{i-1} (x_{t-i} - \bar{x})^2}
    
    params:
        - data: The dataset has n observations(ordered from the most recent to the least recent ) 
                on m varaibles. Can be denoted as [[x1_t-1, x1_t-2, ...., x1_t-n],
                                                    ...,
                                                   [xm_t-1, xm_t-2, ...., xm_t-n]].
                The dataset should be an instance of np.matrix (but not np.array). 
        
        - lambda_: to put how much weight on t-1's forecast variance
    """
    
    deviation = data - data.mean(axis=1)
    weights = exponential_weights(lambda_, data.shape[1])

    # apply the weights to the deviation
    return deviation @ np.diag(weights) @ deviation.T
