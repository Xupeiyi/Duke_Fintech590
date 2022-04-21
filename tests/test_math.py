import os
from unittest import TestCase, main

import numpy as np
import pandas as pd
from sklearn import decomposition

from febrisk.math import (exponential_weights, cal_ewcov, PCA,
                          manhattan_distance, is_psd, near_psd, nearest_psd,
                          cal_partial_derivative)


curr_file_dir = os.path.dirname(os.path.realpath(__file__))


class ExponentialWeightsTest(TestCase):
    
    def test_result_is_correct(self):
        results = exponential_weights(0.97, 4)
        answer = np.array([0.23869637, 0.24607873, 0.25368942, 0.26153548])
        self.assertAlmostEqual(0, manhattan_distance(results - answer), delta=1e-8)


class CalEwcovTest(TestCase):

    def test_result_is_the_same_with_lecture_code(self):
        daily_returns = pd.read_csv(curr_file_dir + "/DailyReturn.csv").set_index('Date').values
        result = cal_ewcov(daily_returns, 0.97)
        answer = pd.read_csv(curr_file_dir + "/DailyReturnEwCov.csv").values
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-8)
    
    def test_can_be_used_on_1d_array(self):
        daily_returns = pd.read_csv(curr_file_dir + "/DailyReturn.csv").set_index('Date')['AAPL'].values
        result = cal_ewcov(daily_returns, 0.97)
        answer = 0.00026875230284538243
        self.assertAlmostEqual(answer, result, delta=1e-8)
    
    
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


def a_non_psd_matrix(size):
    mat = np.array(np.full((size, size), 0.9), dtype='float64')
    np.fill_diagonal(mat, 1)
    mat[0, 1] = 0.7357
    mat[1, 0] = 0.7357
    return mat


def a_psd_matrix(size):
    mat = np.array(np.full((size, size), 0.9))
    np.fill_diagonal(mat, 1)
    mat[0, 1] = 1
    mat[1, 0] = 1
    return mat


class IsPsdTest(TestCase):
    
    def test_return_false_for_non_psd(self):
        non_psd = a_non_psd_matrix(10)
        self.assertFalse(is_psd(non_psd))
    
    def test_return_true_for_psd(self):
        psd = a_psd_matrix(10)
        self.assertTrue(is_psd(psd))


class NearPsdTest(TestCase):
    
    def test_result_is_the_same_with_lecture_code(self):
        non_psd = a_non_psd_matrix(5)
        result = near_psd(non_psd)
        answer = np.array([[1.0,      0.735701, 0.899997, 0.899997, 0.899997],
                           [0.735701, 1.0,      0.899997, 0.899997, 0.899997],
                           [0.899997, 0.899997, 1.0,      0.9,      0.9],
                           [0.899997, 0.899997, 0.9,      1.0,      0.9],
                           [0.899997, 0.899997, 0.9,      0.9,      1.0]])
        
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-5)
    
    def test_result_is_psd(self):
        non_psd = a_non_psd_matrix(5)
        result = near_psd(non_psd)
        self.assertTrue(is_psd(result))


class NearestPsdTest(TestCase):
    
    def test_result_is_the_same_with_lecture_code(self):
        non_psd = a_non_psd_matrix(5)
        result = nearest_psd(non_psd)
        answer = np.array([[1.0, 0.735704, 0.899998, 0.899998, 0.899998],
                           [0.735704, 1.0, 0.899998, 0.899998, 0.899998],
                           [0.899998, 0.899998, 1.0, 0.900001, 0.900001],
                           [0.899998, 0.899998, 0.900001, 1.0, 0.900001],
                           [0.899998, 0.899998, 0.900001, 0.900001, 1.0]])
        
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-5)
    
    def test_result_is_psd(self):
        non_psd = a_non_psd_matrix(5)
        result = nearest_psd(non_psd)
        self.assertTrue(is_psd(result))


class CalPartialDerivativeTest(TestCase):
    
    def test_first_derivative(self):
        def foo(a, b, c=3):
            return a**2 + 3*b + c
        
        df_da = cal_partial_derivative(foo, 1, 'a')
        df_db = cal_partial_derivative(foo, 1, 'b')
        self.assertAlmostEqual(df_da(3, 4, c=5), 2 * 3)
        self.assertAlmostEqual(df_db(3, 4, c=5), 3)
    
    def test_second_derivative(self):
        def foo(a, b, c=3):
            return a**2 + 3*b + c
        
        d2f_da2 = cal_partial_derivative(foo, 2, 'a')
        self.assertAlmostEqual(d2f_da2(3, 4, c=5), 2)
    

if __name__ == '__main__':
    main()
