from copy import deepcopy
import numpy as np
import statsmodels.api as sm


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


def cal_return_attribution(weights, returns):
    """
    Calculate the return attribution of each asset in a portfolio given the initial
    weight and returns. The initial weight and returns starts at the same period.

    params:
        - weights: np.arrays, shape(n,)
        - returns: np.arrays, shape(t, n)
    return:
        - return attribution: np.arrays, shape(n,)
    """
    assert len(weights) == returns.shape[1], \
        "weights and returns should have the same number of columns"
    updated_weights = update_weights(weights, returns)
    pfl_returns = (updated_weights * returns).sum(axis=1)
    pfl_total_return = (1 + pfl_returns).prod(axis=0) - 1
    k = np.log(1 + pfl_total_return) / pfl_total_return
    carino_k = np.array([np.log(1 + pfl_returns) / (pfl_returns * k)]).T
    return (returns * updated_weights * carino_k).sum(axis=0)


def cal_risk_attribution(weights, returns):
    """
    Calculate the risk attribution of each asset in a portfolio given the initial
    weight and returns. The initial weight and returns starts at the same period.

    params:
        - weights: np.arrays, shape(n,)
        - returns: np.arrays, shape(t, n)
    return:
        - risk attribution: np.arrays, shape(n,)
    """
    assert len(weights) == returns.shape[1], \
        "weights and returns should have the same number of columns"
    updated_weights = update_weights(weights, returns)
    weighted_returns = (updated_weights * returns)
    pfl_returns = weighted_returns.sum(axis=1)
    pfl_risk = pfl_returns.std()
    
    nassets = len(weights)
    risk_attribution = np.empty(nassets, dtype=float)
    for i in range(nassets):
        # calculate betas
        # r_i = alpha + sum(beta * r_pfl) + error
        model = sm.OLS(weighted_returns[:, i], sm.add_constant(pfl_returns))
        results = model.fit()
        
        # sigma_i = beta * sigma_pfl
        risk_attribution[i] = results.params[1] * pfl_risk
    
    return risk_attribution
