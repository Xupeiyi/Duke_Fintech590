from copy import deepcopy
import numpy as np


def cal_arith_return(prices):
    """
    Compute the arithmetic return given a sequence of prices.
    The prices should be in order from least to most recent.
    """
    price_chg_pct = (prices / prices.shift(1))[1:]
    return price_chg_pct - 1


def cal_log_return(prices):
    """
    Compute the arithmetic return given a sequence of prices.
    The prices should be in order from least to most recent.
    """
    price_chg_pct = (prices / prices.shift(1))[1:]
    return np.log(price_chg_pct)


def cal_return(prices, method='arithmetic'):
    funcs = {
        'arithmetic': cal_arith_return,
        'log': cal_log_return
    }
    return funcs[method](prices)


def update_weights(weights, returns):
    """
    Update the weights of each asset in a portfolio given the initial
    weight and returns. The initial weight and returns starts at the
    same period.

    params:
        - weights: np.arrays, shape(n,)
        - returns: np.arrays, shape(t, n)
    return:
        - updated_weights: np.arrays, shape(t, n)
    """
    latest_weights = deepcopy(weights)
    updated_weights = np.empty(shape=(returns.shape[0], len(latest_weights)), dtype=float)
    
    for i in range(returns.shape[0]):
        updated_weights[i, :] = latest_weights
        latest_weights *= (1 + returns[i, :])
        latest_weights /= sum(latest_weights)
    
    return updated_weights
