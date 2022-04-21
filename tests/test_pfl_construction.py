import os
from unittest import TestCase, main
import numpy as np
import pandas as pd

from febrisk.math import manhattan_distance
from febrisk.dist_fit import TFitter
from febrisk.simulation import CopulaSimulator
from febrisk.pfl_construction import (
    build_optimized_portfolio,
    build_risk_parity_portfolio_on_std,
    build_risk_parity_portfolio_on_es
)

curr_file_dir = os.path.dirname(os.path.realpath(__file__))

class BuildOptimizedPortfolioTest(TestCase):
    
    def test_result_is_correct(self):
        mean = np.array([0.14172810469924474, 0.17017181265120937,
                         0.11025398590140209, 0.13648926641109915,
                         0.07102713642777535])
        covariance = np.array([[0.0654412, 0.0399929, 0.00013588, 0.0118799, -0.00323781],
                               [0.0399929, 0.0652365, -0.00156837, 0.022863, -0.00155483],
                               [0.00013588, -0.00156837, 0.0229777, 0.00791532, 0.00912831],
                               [0.0118799, 0.022863, 0.00791532, 0.0545269, 0.00576461],
                               [-0.00323781, -0.00155483, 0.00912831, 0.00576461, 0.022242]])
        result = build_optimized_portfolio(mean, covariance, 0.0025)
        answer = np.array([0.10075988181538116, 0.20950981862533438,
                           0.43839111238558587, 0.08118475735284336,
                           0.17015442982085532])
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-3)


class BuildRiskParityPortfolioOnStdTest(TestCase):
    
    def test_result_is_correct(self):
        covariance = np.array([[0.0654412, 0.0399929, 0.00013588, 0.0118799, -0.00323781],
                               [0.0399929, 0.0652365, -0.00156837, 0.022863, -0.00155483],
                               [0.00013588, -0.00156837, 0.0229777, 0.00791532, 0.00912831],
                               [0.0118799, 0.022863, 0.00791532, 0.0545269, 0.00576461],
                               [-0.00323781, -0.00155483, 0.00912831, 0.00576461, 0.022242]])
        result = build_risk_parity_portfolio_on_std(covariance)
        answer = np.array([0.15420278, 0.14256632, 0.26515987, 0.15078634, 0.28728468])
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-3)
        

class BuildRiskParityPortfolioOnEsTest(TestCase):
    
    def test_result_is_correct(self):
        all_rets = pd.read_csv(curr_file_dir + '/DailyReturn.csv', parse_dates=['Date']).set_index('Date')
        stocks = ['AAPL', 'MSFT', 'BRK-B', 'CSCO', 'JNJ']
        copula = CopulaSimulator()
        copula.fit(all_rets[stocks].values.T, [TFitter() for _ in stocks])
        sim_rets = copula.simulate(5000)
        result = build_risk_parity_portfolio_on_es(sim_rets.T)
        answer = np.array([0.149, 0.135, 0.264, 0.142, 0.310])
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=0.1)
        
        
if __name__ == '__main__':
    main()