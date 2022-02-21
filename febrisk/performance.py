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
