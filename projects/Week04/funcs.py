from collections import namedtuple

import numpy as np


# https://github.com/raphaelvallat/pingouin/blob/master/pingouin/multivariate.py
def multivariate_normality(X, alpha=.05):
    """Henze-Zirkler multivariate normality test.
    Parameters
    ----------
    X : np.array
        Data matrix of shape (n_samples, n_features).
    alpha : float
        Significance level.
    Returns
    -------
    hz : float
        The Henze-Zirkler test statistic.
    pval : float
        P-value.
    normal : boolean
        True if X comes from a multivariate normal distribution.
    See Also
    --------
    normality : Test the univariate normality of one or more variables.
    homoscedasticity : Test equality of variance.
    sphericity : Mauchly's test for sphericity.
    Notes
    -----
    The Henze-Zirkler test [1]_ has a good overall power against alternatives
    to normality and works for any dimension and sample size.
    Adapted to Python from a Matlab code [2]_ by Antonio Trujillo-Ortiz and
    tested against the
    `MVN <https://cran.r-project.org/web/packages/MVN/MVN.pdf>`_ R package.
    Rows with missing values are automatically removed.
    References
    ----------
    .. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
       tests for multivariate normality. Communications in Statistics-Theory
       and Methods, 19(10), 3595-3617.
    .. [2] Trujillo-Ortiz, A., R. Hernandez-Walls, K. Barba-Rojo and L.
       Cupul-Magana. (2007). HZmvntest: Henze-Zirkler's Multivariate
       Normality Test. A MATLAB file.
    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset('multivariate')
    >>> X = data[['Fever', 'Pressure', 'Aches']]
    >>> pg.multivariate_normality(X, alpha=.05)
    HZResults(hz=0.540086101851555, pval=0.7173686509622385, normal=True)
    """
    from scipy.stats import lognorm

    # Check input and remove missing values
    X = np.asarray(X)
    assert X.ndim == 2, 'X must be of shape (n_samples, n_features).'
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    assert n >= 3, 'X must have at least 3 rows.'
    assert p >= 2, 'X must have at least two columns.'

    # Covariance matrix
    S = np.cov(X, rowvar=False, bias=True)
    S_inv = np.linalg.pinv(S).astype(X.dtype)  # Preserving original dtype
    difT = X - X.mean(0)

    # Squared-Mahalanobis distances
    Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
    Y = np.linalg.multi_dot([X, S_inv, X.T])
    Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + \
        np.tile(np.diag(Y.T), (n, 1))

    # Smoothing parameter
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * \
        (n**(1 / (p + 4)))

    # Is matrix full-rank (columns are linearly independent)?
    if np.linalg.matrix_rank(S) == p:
        hz = n * (1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk))) - 2
                  * ((1 + (b**2))**(-p / 2)) * (1 / n)
                  * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
                  + ((1 + (2 * (b**2)))**(-p / 2)))
    else:
        hz = n * 4

    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2
    # Mean and variance
    mu = 1 - a**(-p / 2) * (1 + p * b**2 / a + (p * (p + 2)
                                                * (b**4)) / (2 * a**2))
    si2 = 2 * (1 + 4 * b**2)**(-p / 2) + 2 * a**(-p) * \
        (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4)) \
        - 4 * wb**(-p / 2) * (1 + (3 * p * b**4) / (2 * wb)
                              + (p * (p + 2) * b**8) / (2 * wb**2))

    # Lognormal mean and variance
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2 + mu**2) / mu**2))

    # P-value
    pval = lognorm.sf(hz, psi, scale=np.exp(pmu))
    normal = True if pval > alpha else False

    HZResults = namedtuple('HZResults', ['hz', 'pval', 'normal'])
    return HZResults(hz=hz, pval=pval, normal=normal)


# Exponentially Weighted Covariance
def exponential_weights(lambda_, nlags):
    weights = np.array([(1-lambda_) * (lambda_**(lag-1)) for lag in range(1,nlags+1)])
    weights /= weights.sum() # normalized weights
    return weights

def cal_ewcov(data, lambda_):
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
    
    dev = data - data.mean(axis=1) # deviation
    weights = exponential_weights(lambda_, data.shape[1])
    
    # apply the weights to the deviation
    cov = dev @ np.diag(weights) @ dev.T
    return cov