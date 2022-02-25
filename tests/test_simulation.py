from unittest import TestCase, main

import scipy
import numpy as np

from febrisk.statistics import manhattan_distance
from febrisk.dist_fit import NormalFitter, TFitter
from febrisk.simulation import chol_psd, CopulaSimulator


def a_psd_matrix(size):
    mat = np.array(np.full((size, size), 0.9))
    np.fill_diagonal(mat, 1)
    mat[0, 1] = 1
    mat[1, 0] = 1
    return mat


class CholPsdTest(TestCase):
    
    def test_result_is_correct(self):
        psd = a_psd_matrix(5)
        result = chol_psd(psd)
        answer = np.array([[1.0,  0.0,  0.0,         0.0,         0.0],
                           [1.0,  0.0,  0.0,         0.0,         0.0],
                           [0.9,  0.0,  0.435889894, 0.0,         0.0],
                           [0.9,  0.0,  0.20647416,  0.38388595,  0.0],
                           [0.9,  0.0,  0.20647416,  0.123391911, 0.363514589]])
        self.assertAlmostEqual(0, manhattan_distance(result - answer), delta=1e-8)


class CopulaSimulationTest(TestCase):
    
    def test_distribution_of_simulated_data_is_close_enough(self):
        x1 = scipy.stats.t(loc=1, df=3, scale=1.5).rvs(100000)
        x23 = scipy.stats.multivariate_normal([0.5, -0.2], [[1.3, 0.7], [0.7, 1]]).rvs(100000).T
        x = np.array([x1, x23[0, :], x23[1, :]])
        
        cpl = CopulaSimulator()
        fitters = [TFitter(), NormalFitter(), NormalFitter()]
        cpl.fit(x, fitters)
        sim_x = cpl.simulate(100000)
        
        # test the simulated data has almost the same covariance
        diff = manhattan_distance(np.cov(sim_x) - np.cov(x))
        self.assertTrue(diff < 1, f"The difference between covariance matrices is {diff}")
        self.assertAlmostEqual(0.7, np.cov(sim_x[1:, :])[0, 1], delta=1e-1)
        
        # test the marginal distribution is almost the same
        delta = 3e-1
        loc, df, scale = fitters[0].fitted_params
        self.assertAlmostEqual(1, loc, delta=delta)
        self.assertAlmostEqual(3, df, delta=delta)
        self.assertAlmostEqual(1.5, scale, delta=delta)
        
        loc, scale = fitters[1].fitted_params
        self.assertAlmostEqual(0.5, loc, delta=1e-1)
        self.assertAlmostEqual(1.3, scale, delta=delta)
        
        loc, scale = fitters[2].fitted_params
        self.assertAlmostEqual(-0.2, loc, delta=1e-1)
        self.assertAlmostEqual(1, scale, delta=delta)
    
    
if __name__ == '__main__':
    main()
