import scipy.optimize
import numpy as np
from febrisk.risk import expected_shortfall


# =================================
# Maximum Sharpe Ratio
# =================================

def cal_pfl_sharpe_ratio(weights, mean, covariance, r_f):
    returns = mean @ weights.T
    std = np.sqrt(weights @ covariance @ weights.T)
    return (returns - r_f) / std
    

def build_optimized_portfolio(mean, covariance, r_f):
    nassets = len(mean)
    bounds = ((0, 1) for _ in range(nassets))
    x0 = np.ones(nassets) / nassets
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    results = scipy.optimize.minimize(lambda w: -1*cal_pfl_sharpe_ratio(w, mean, covariance, r_f),
                                      x0=x0, bounds=bounds, constraints=constraints)
    weights = results.x
    return weights


# =================================
# Risk Parity on Standard Deviation
# =================================

def cal_pfl_volatility(weights, covariance):
    return np.sqrt(weights @ covariance @ weights.T)


def cal_component_std(weights, covariance):
    pfl_vol = cal_pfl_volatility(weights, covariance)
    return weights * (covariance @ weights.T) / pfl_vol


def cal_component_std_sse(weights, covariance, budget=None):
    if not budget:
        budget = np.ones_like(weights)

    component_std = cal_component_std(weights, covariance) / budget
    dev = component_std - np.mean(component_std)
    return dev @ dev.T


def build_risk_parity_portfolio_on_std(covariance, budget=None):
    nassets = covariance.shape[0]
    
    x0 = np.array([1 / nassets for _ in range(nassets)])
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = ((0, 1) for _ in range(nassets))
    results = scipy.optimize.minimize(lambda w: 1e5*cal_component_std_sse(w, covariance, budget),
                                      x0=x0, bounds=bounds, constraints=constraints)
    return results.x


# =================================
# Risk Parity on Expected Shortfall
# =================================
def cal_pfl_es(weights, returns):
    return expected_shortfall(returns @ weights.T)


def cal_component_es(weights, returns, delta=1e-6):
    nassets = len(weights)
    es = cal_pfl_es(weights, returns)
    component_es = np.empty(nassets, dtype=float)

    for i in range(nassets):
        weight_i = weights[i]
        weights[i] += delta
        component_es[i] = weight_i * (cal_pfl_es(weights, returns) - es) / delta
        weights[i] -= delta
    
    return component_es


def cal_component_es_sse(weights, returns, budget=None, delta=1e-6):
    if not budget:
        budget = np.ones_like(weights)
    component_es = cal_component_es(weights, returns, delta) / budget
    dev = component_es - np.mean(component_es)
    return dev @ dev.T


def build_risk_parity_portfolio_on_es(returns, budget=None):
    nassets = returns.shape[1]
    x0 = np.array([1 / nassets for _ in range(nassets)])
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = ((0, 1) for _ in range(nassets))
    results = scipy.optimize.minimize(lambda w: 1e5*cal_component_es_sse(w, returns, budget),
                                      x0=x0, bounds=bounds, constraints=constraints)
    return results.x
