import scipy
import numpy as np


def VaR(data, mean, alpha: float = 0.05):
    """
    Calculate the VaR given a 1-d array of data.
    Currently 2-d array input is not supported.
    """
    return mean - np.quantile(data, q=alpha, method='midpoint')


def VaR_kde(data, mean, alpha: float = 0.05, **kwargs):
    """
    Calculate the Var based on a sequence of simulations.
    Smooth the distribution of the simulations using Gaussian KDE.
    """
    kde = scipy.stats.gaussian_kde(data, bw_method=kwargs.get('bw_method', None),
                                   weights=kwargs.get('weights', None))
    
    def cdf_equals_alpha(upper_bound):
        return kde.integrate_box(0, upper_bound) - alpha
    
    return mean - scipy.optimize.fsolve(cdf_equals_alpha, x0=kwargs.get('x0', mean))[0]


def expected_shortfall(data, alpha=0.05):
    """
    Calculate the expected shortfall given a 1-d array of data.
    """
    sorted_data = np.sort(data)
    VaR_idx = int(alpha * sorted_data.shape[-1])
    return -np.mean(sorted_data[:VaR_idx], axis=-1)
