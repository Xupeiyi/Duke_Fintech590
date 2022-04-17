import os
from unittest import TestCase, main
import numpy as np
import pandas as pd
from febrisk.performance import update_weights, cal_return_attribution, cal_risk_attribution
from febrisk.math import manhattan_distance


curr_file_dir = os.path.dirname(os.path.realpath(__file__))


class UpdateWeightsTest(TestCase):
    
    def test_result_is_correct(self):
        weights = np.array([0.10075988181538116, 0.20950981862533438,
                            0.43839111238558587, 0.08118475735284336,
                            0.17015442982085532])
        returns = pd.read_csv(curr_file_dir + "/UpdatedReturns.csv", header=None).values
        ans = pd.read_csv(curr_file_dir + "/UpdatedWeights.csv", header=None).values
        result = update_weights(weights, returns)
        self.assertAlmostEqual(0, manhattan_distance(result - ans))


class CalReturnAttributionTest(TestCase):
    
    def test_result_is_correct(self):
        weights = np.array([0.10075988181538116, 0.20950981862533438,
                            0.43839111238558587, 0.08118475735284336,
                            0.17015442982085532])
        returns = pd.read_csv(curr_file_dir + "/UpdatedReturns.csv", header=None).values
        result = cal_return_attribution(weights, returns)
        answer = np.array([-0.0046, -0.0071, -0.0037, -0.0075, -0.0022])
        self.assertAlmostEqual(0, manhattan_distance(answer-result), delta=1e-3)


class CalRiskAttributionTest(TestCase):
    
    def test_result_is_correct(self):
        weights = np.array([0.10075988181538116, 0.20950981862533438,
                            0.43839111238558587, 0.08118475735284336,
                            0.17015442982085532])
        returns = pd.read_csv(curr_file_dir + "/UpdatedReturns.csv", header=None).values
        result = cal_risk_attribution(weights, returns)
        answer = np.array([0.0016, 0.0032, 0.0048, 0.0009, 0.0014])
        self.assertAlmostEqual(0, manhattan_distance(answer-result), delta=1e-3)


if __name__ == '__main__':
    main()
