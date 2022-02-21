from unittest import TestCase, main

import numpy as np
import pandas as pd
from sklearn import decomposition

from febrisk.statistics import exponential_weights, cal_ewcov, PCA, manhattan_distance


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
    
    
class PCATest(TestCase):
    
    def test_result_is_same_with_sklearn(self):
        data = np.array([[1, 2, 3, 4],
                         [4, 9, 6, 8],
                         [7, 2, 9, 10]])

        cov = np.cov(data)
        pca = PCA(cov)
        
        skl_pca = decomposition.PCA()
        skl_pca.fit(data.T)   # input need to be transposed
        
        self.assertAlmostEqual(
            0, manhattan_distance(pca.explained_variance - skl_pca.explained_variance_),
            delta=1e-8
        )


if __name__ == '__main__':
    main()
