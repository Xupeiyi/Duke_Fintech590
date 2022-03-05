import scipy
import numpy as np


def bsm(is_call: bool, price, strike, ttm, r_f, hc, vol):
    b = r_f - hc  # risk free rate - holding cost
    is_call = 2 * int(is_call) - 1
    N = scipy.stats.norm(0, 1).cdf

    d1 = (np.log(price/strike) + (b + 0.5*vol**2)*ttm) / (vol*np.sqrt(ttm))
    d2 = d1 - vol*np.sqrt(ttm)
    return is_call * (price*np.e**((-hc)*ttm)*N(is_call*d1) - strike*np.e**(-r_f*ttm)*N(is_call*d2))


def implied_vol(is_call: bool, price, strike, ttm, r_f, hc, opt_value):
    def equation(vol):
        return bsm(is_call, price, strike, ttm, r_f, hc, vol) - opt_value

    return scipy.optimize.fsolve(equation, x0=(0.5))[0]