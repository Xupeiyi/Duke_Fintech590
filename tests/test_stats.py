from unittest import TestCase, main

import numpy as np
import pandas as pd

from febrisk.stats import exponential_weights, cal_ewcov


class ExponentialWeightsTest(TestCase):
    
    def test_result_is_correct(self):
        results = exponential_weights(0.97, 4)
        answer = [0.26153548, 0.25368942, 0.24607873, 0.23869637]
        difference = (results - answer).sum()
        self.assertAlmostEqual(0, difference, delta=1e-8)


class CalEwcovTest(TestCase):

    def test_result_is_the_same_with_lecture_code(self):
        daily_return = pd.read_csv("DailyReturn.csv").iloc[::-1, 1:].T
        daily_return = np.matrix(daily_return)
        result = cal_ewcov(daily_return, 0.97)
        
        answer = pd.read_csv("DailyReturnEwCov.csv")
        difference = (result - answer.values).sum()
        self.assertAlmostEqual(0, difference, delta=1e-8)
        

if __name__ == '__main__':
    main()