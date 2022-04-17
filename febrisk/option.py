from typing import List, Tuple

import scipy
import numpy as np

from febrisk.math import cal_partial_derivative

STD_NORMAL_CDF = scipy.stats.norm(0, 1).cdf
STD_NORMAL_PDF = scipy.stats.norm(0, 1).pdf


def cal_d1(price, strike, ttm, carry_cost, ivol):
    return (np.log(price / strike) + (carry_cost + 0.5 * ivol ** 2) * ttm) / (ivol * np.sqrt(ttm))


def cal_d2(d1, ivol, ttm):
    return d1 - ivol * np.sqrt(ttm)


def cal_ttm(curr_date, expire_date):
    return (expire_date - curr_date).days / 365


def bsm(is_call: bool, price, strike, ttm, r_f, div_rate, ivol):
    b = r_f - div_rate  # cost of carry = risk free rate - dividend rate
    is_call = 1 if is_call else -1
    
    d1 = cal_d1(price, strike, ttm, b, ivol)
    d2 = cal_d2(d1, ivol, ttm)
    return is_call * (price * np.e ** ((-div_rate) * ttm) * STD_NORMAL_CDF(is_call * d1)
                      - strike * np.e ** (-r_f * ttm) * STD_NORMAL_CDF(is_call * d2))


def implied_vol(is_call: bool, price, strike, ttm, r_f, div_rate, opt_value):
    def equation(ivol):
        return bsm(is_call, price, strike, ttm, r_f, div_rate, ivol) - opt_value
    
    return scipy.optimize.fsolve(equation, x0=(0.5,))[0]


# ====================
# Greeks
# ====================

# 1. delta
cal_euro_delta_num = cal_partial_derivative(bsm, 1, 'price')


def cal_euro_delta(is_call, price, strike, ttm, r_f, div_rate, ivol):
    b = r_f - div_rate  # cost of carry = risk free rate - dividend rate
    is_call = 0 if is_call else 1
    d1 = cal_d1(price, strike, ttm, b, ivol)
    return np.exp((b - r_f) * ttm) * (STD_NORMAL_CDF(d1) - is_call)


# 2. gamma
cal_euro_gamma_num = cal_partial_derivative(bsm, 2, 'price')


def cal_euro_gamma(price, strike, ttm, r_f, div_rate, ivol):
    b = r_f - div_rate
    d1 = cal_d1(price, strike, ttm, b, ivol)
    return (STD_NORMAL_PDF(d1) * np.exp((b - r_f) * ttm)) / (price * ivol * np.sqrt(ttm))


# 3. vega
cal_euro_vega_num = cal_partial_derivative(bsm, 1, 'ivol')


def cal_euro_vega(price, strike, ttm, r_f, div_rate, ivol):
    b = r_f - div_rate
    d1 = cal_d1(price, strike, ttm, b, ivol)
    return price * np.exp((b - r_f) * ttm) * STD_NORMAL_PDF(d1) * np.sqrt(ttm)


# 4. vega
def cal_euro_theta_num(is_call, price, strike, ttm, r_f, div_rate, ivol):
    cal_neg_theta = cal_partial_derivative(bsm, 1, 'ttm')
    return -1 * cal_neg_theta(is_call, price, strike, ttm, r_f, div_rate, ivol)


def cal_euro_theta(is_call, price, strike, ttm, r_f, div_rate, ivol):
    is_call = 1 if is_call else -1
    b = r_f - div_rate
    d1 = cal_d1(price, strike, ttm, b, ivol)
    d2 = cal_d2(d1, ivol, ttm)
    
    term1 = -price * np.exp((b - r_f) * ttm) * STD_NORMAL_PDF(d1) * ivol / (2 * np.sqrt(ttm))
    term2 = -1 * is_call * (b - r_f) * price * np.exp((b - r_f) * ttm) * STD_NORMAL_CDF(d1 * is_call)
    term3 = -1 * is_call * r_f * strike * np.exp(-r_f * ttm) * STD_NORMAL_CDF(d2 * is_call)
    
    return term1 + term2 + term3


# 5. rho
cal_euro_rho_num = cal_partial_derivative(bsm, 1, 'r_f')


def cal_euro_rho(is_call, price, strike, ttm, r_f, div_rate, ivol):
    is_call = 1 if is_call else -1
    b = r_f - div_rate
    N = scipy.stats.norm(0, 1).cdf
    d1 = cal_d1(price, strike, ttm, b, ivol)
    d2 = cal_d2(d1, ivol, ttm)
    return is_call * ttm * strike * np.exp(-r_f * ttm) * N(is_call * d2)


# 6. carry rho
def cal_euro_carry_rho_num(is_call, price, strike, ttm, r_f, div_rate, ivol):
    cal_neg_rho = cal_partial_derivative(bsm, 1, 'div_rate')
    return -1 * cal_neg_rho(is_call, price, strike, ttm, r_f, div_rate, ivol)


def cal_euro_carry_rho(is_call, price, strike, ttm, r_f, div_rate, ivol):
    is_call = 1 if is_call else -1
    b = r_f - div_rate
    d1 = cal_d1(price, strike, ttm, b, ivol)
    return is_call * ttm * price * np.exp((b - r_f) * ttm) * STD_NORMAL_CDF(is_call * d1)


# ==================
# Binomial Trees
# ==================
def nnodes(nperiods):
    return (nperiods + 2) * (nperiods + 1) // 2


def node_idx(i, j):
    return nnodes(j - 1) + i


def cal_price_chg_prob(u, d, dt, carry_cost):
    pu = (np.exp(carry_cost * dt) - d) / (u - d)
    return pu, 1 - pu


def cal_price_chg(ivol, dt):
    u = np.exp(ivol * np.sqrt(dt))
    return u, 1 / u


def bt_american_continuous_div(is_call, price, strike, ttm, r_f, div_rate, ivol, nperiods):
    is_call = 1 if is_call else -1
    dt = ttm / nperiods
    u, d = cal_price_chg(ivol, dt)
    pu, pd = cal_price_chg_prob(u, d, dt, r_f - div_rate)
    
    option_values = np.empty(nnodes(nperiods), dtype=float)
    for j in range(nperiods, -1, -1):
        for i in range(j, -1, -1):
            idx = node_idx(i, j)
            curr_price = price * u ** i * d ** (j - i)
            option_values[idx] = max(0, is_call * (curr_price - strike))
            if j < nperiods:
                value_no_exercise = np.exp(-r_f * dt) * (pu * option_values[node_idx(i + 1, j + 1)]
                                                         + pd * option_values[node_idx(i, j + 1)])
                option_values[idx] = max(option_values[idx], value_no_exercise)
    
    return option_values[0]


def bt_american(is_call: bool, price, strike, ttm, r_f, div_rate,
                ivol, nperiods, dividends: List[Tuple] = None):
    """
    Caculate the value of american options by binomial tree.
    """
    
    if not dividends or dividends[0][0] > nperiods:
        return bt_american_continuous_div(is_call, price, strike, ttm, r_f, div_rate, ivol, nperiods)
    
    is_call = 1 if is_call else -1
    dt = ttm / nperiods
    u, d = cal_price_chg(ivol, dt)
    pu, pd = cal_price_chg_prob(u, d, dt, r_f - div_rate)
    
    div_time, div_amount = dividends[0]  # do not use dividends.pop(0),
    # because it will modify the dividends argument of the function on the previous stack
    option_values = np.empty(nnodes(div_time), dtype=float)
    
    # update the parameters for the next recursion
    new_dividends = [(time - div_time, amount) for time, amount in dividends[1:]]
    new_ttm = ttm - div_time * dt
    new_nperiods = nperiods - div_time
    
    for j in range(div_time, -1, -1):
        for i in range(j, -1, -1):
            curr_price = price * u ** i * d ** (j - i)
            value_exercise = max(0, is_call * (curr_price - strike))
            if j < div_time:
                value_no_exercise = np.exp(-r_f * dt) * \
                                    (pu * option_values[node_idx(i + 1, j + 1)] + pd * option_values[
                                        node_idx(i, j + 1)])
            else:
                value_no_exercise = bt_american(True if is_call == 1 else False, curr_price - div_amount, strike,
                                                new_ttm,
                                                r_f, div_rate, ivol, new_nperiods, new_dividends)
            option_values[node_idx(i, j)] = max(value_exercise, value_no_exercise)
    
    return option_values[0]


if __name__ == '__main__':
    print(nnodes(2) == 6)
    print(node_idx(1, 1) == 2)
    ans = bt_american_continuous_div(is_call=False, price=100, strike=100, ttm=0.5, r_f=0.08, div_rate=0,
                                     ivol=0.3, nperiods=100)
    print(abs(ans - 6.856849) < 1e-5)
    
    ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5, r_f=0.08, div_rate=0,
                      ivol=0.3, nperiods=100, dividends=[(1, 1.0), (2, 1.0)])
    print(abs(ans - 9.215344) < 1e-5)
