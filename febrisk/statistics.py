import numpy as np


# ==========================
# Covariances
# ==========================
def cal_std(cov):
    """Calculate the standard deviations with a covariance matrix."""
    return np.sqrt(np.diag(cov))


def cal_corr(cov):
    """Calculate the correlation matrix with a covariance matrix."""
    std = cal_std(cov)
    inversed_std = np.diag(1 / std)
    corr = inversed_std @ cov @ inversed_std
    return corr


def cal_cov(corr, std):
    """
    Calculate the covariance matrix with a correlation matrix
    and a standard deviations array.
    """
    std = np.diag(std)
    return std @ corr @ std


# ==================================
# Matrix's Norms
# ==================================
def manhattan_distance(arr):
    return np.abs(arr).sum()


def frobenius_norm(matrix):
    """
    The Frobenius Norm of matrix A is 
        ||𝐴|| = \sqrt{\sum_{i=1}^n \sum_{j=1}^n a_{ij}^2}.
    """
    return np.sqrt(np.square(matrix).sum())


# =================================
# Exponentially Weighted Covariance
# =================================
def exponential_weights(lambda_: float, nlags: int) -> np.array:
    """
    Calculate the weights from T = t-1 to t-nlags if current period is T = t.
    """
    weights = np.array([(1 - lambda_) * (lambda_ ** (lag - 1)) for lag in range(1, nlags + 1)])
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
    return deviation @ np.diag(weights) @ deviation.T


# ===========================================
# Principal Component Analysis
# ============================================
class PCA:
    """
    Apply PCA to a n*n covariance matrix sigma.
    """
    
    def __init__(self, sigma, delta=1e-8):
        eig_vals, eig_vecs = np.linalg.eigh(sigma)
        
        # only keep positive eigen values and vectors
        is_positive = eig_vals > delta
        eig_vals = eig_vals[is_positive]  # eigen values can have very tiny imaginary parts
        eig_vecs = eig_vecs[:, is_positive]
        
        # sort the eigen values and eigen vectors in a descending order
        desc_ranking = np.argsort(eig_vals)[::-1]
        self._eig_vals = eig_vals[desc_ranking]
        self._eig_vecs = eig_vecs[:, desc_ranking]
        
        # calculate explained variance ratio (evr)
        self._evr = self._eig_vals / self._eig_vals.sum()
        
        # set the last value to 1 to eliminate rounding errors of floating point numbers
        self._cumulative_evr = self._evr.cumsum()
        self._cumulative_evr[-1] = 1
    
    @property
    def explained_variance(self):
        return self._eig_vals
    
    @property
    def explained_variance_ratio(self):
        return self._evr
    
    @property
    def cumulative_evr(self):
        return self._cumulative_evr
    
    @property
    def eig_vecs(self):
        return self._eig_vecs
