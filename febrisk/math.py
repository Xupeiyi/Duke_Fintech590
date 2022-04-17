import inspect
from typing import Callable
import numpy as np


# ==========================
# Covariance
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
# PSD
# ==================================
def manhattan_distance(arr):
    return np.abs(arr).sum()


def frobenius_norm(matrix):
    """
    The Frobenius Norm of matrix A is
        ||𝐴|| = \sqrt{\sum_{i=1}^n \sum_{j=1}^n a_{ij}^2}.
    """
    return np.sqrt(np.square(matrix).sum())


def is_psd(matrix, tolerance=1e-8):
    """
    Examine if matrix is PSD by checking if its eigen values are all non-negative.
    
    params:
        - matrix: a real symmetric matrix.
        - tolerance
    """
    # examine if a matrix is real symmetric
    if abs(matrix - matrix.T).sum() > tolerance:
        raise ValueError("This function is for real symmetric matrices!")
    
    eig_values, _ = np.linalg.eigh(matrix)
    return all(eig_values > -tolerance)


def near_psd(corr):
    """
    Rebonato and Jackel's method for finding an acceptable PSD matrix.
    corr: a correlation matrix.
    """
    # update the eigen value and scale
    eig_vals, eig_vecs = np.linalg.eigh(corr)
    eig_vals[eig_vals < 0] = 0
    
    ts = 1 / (np.square(eig_vecs) @ eig_vals)
    sqrt_T = np.diagflat(np.sqrt(ts))
    sqrt_lambda = np.diag(np.sqrt(eig_vals))
    
    root = sqrt_T @ eig_vecs @ sqrt_lambda  # B = sqrt(𝑇) * S * sqrt(Λ')
    near_corr = root @ root.T
    
    return near_corr


def projection_u(matrix):
    """Projection U sets diagonal elements to 1."""
    new_matrix = matrix.copy()
    np.fill_diagonal(new_matrix, 1)
    return new_matrix


def projection_s(matrix):
    """
    Projection S reconstructs a matrix by setting the
    negative eigen values of the original matrix to zero.
    """
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    eig_vals[eig_vals < 0] = 0
    return eig_vecs @ np.diag(eig_vals) @ eig_vecs.T


def nearest_psd(corr, max_iter=100, tolerance=1e-9):
    """
    Use Higham's method to generate the nearest PSD of a correlation matrix.
    There is no gaurantee that this function can generate a PSD.

    params:
        - corr: the correlation matrix.
        - max_iter: maximum number of iterations.
        - tolerance: break the iteration if we are not able to improving
                     the frobenius norm by at least tolerance value.
    """
    # ΔS0 = 0, Y0 = A, γ0 = max float
    delta_s = 0
    y = corr
    prev_gamma = np.finfo(np.float64).max
    
    # Loop k ∈ 1... max Iterations
    for i in range(max_iter):
        r = y - delta_s      # Rk = Yk-1 − ΔSk-1
        x = projection_s(r)  # Xk = Ps(Rk)
        delta_s = x - r      # ΔSk = Xk − Rk
        y = projection_u(x)  # Yk = Pu(Xk)
        gamma = frobenius_norm(y - corr)
        
        # if |γk-1 − γk | < tol then break
        if abs(gamma - prev_gamma) < tolerance:
            break
        prev_gamma = gamma
    
    return y


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


# ============================================
# Partial Derivative
# ============================================
def first_derivative(f, x, delta):
    return (f(x+delta) - f(x-delta)) / (2*delta)


def second_derivative(f, x, delta):
    return (f(x+delta) + f(x-delta) - 2*f(x)) / delta**2


def cal_partial_derivative(f: Callable, order: int, arg_name: str, delta=1e-3) -> Callable:
    """
    Return the partial derivative of a function with respect to one of its arguments.

    params:
        - f : the original function
        - order: the order of derivative
        - arg_name: the name of the variable that the partial derivative is with respect to
        - delta: precision
    """
    arg_names = list(inspect.signature(f).parameters.keys())  # the name of funtion f's arguments
    derivative_fs = {1: first_derivative, 2: second_derivative}

    def partial_derivative(*args, **kwargs):
        args_dict = dict(list(zip(arg_names, args)) + list(kwargs.items()))
        arg_val = args_dict.pop(arg_name)

        def partial_f(x):
            p_kwargs = {arg_name:x, **args_dict}
            return f(**p_kwargs)

        return derivative_fs[order](partial_f, arg_val, delta)

    return partial_derivative
