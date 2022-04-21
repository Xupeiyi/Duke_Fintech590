from unittest import TestCase, main

import scipy
import numpy as np

from febrisk.math import manhattan_distance, examine_normality
from febrisk.simulation import chol_psd, CholeskySimulator, PCASimulator, CopulaSimulator


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


class CholeskySimulatorTest(TestCase):
    
    def test_simulated_cov_almost_equals_original_cov(self):
        cov = np.array([[1.3, 0.7], [0.7, 1]])
        simulator = CholeskySimulator(cov)
        sim_data = simulator.simulate(10000)
        sim_cov = np.cov(sim_data, rowvar=False)
        self.assertAlmostEqual(0, manhattan_distance(cov - sim_cov), delta=0.3)

    def test_simulated_data_follows_normal_dist(self):
        cov = np.array([[1.3, 0.7], [0.7, 1]])
        simulator = CholeskySimulator(cov)
        sim_data = simulator.simulate(1000)
        self.assertTrue(examine_normality(sim_data))


class PCASimulatorTest(TestCase):
    
    def test_simulated_cov_almost_equals_original_cov(self):
        cov = np.array([[1.3, 0.7], [0.7, 1]])
        simulator = PCASimulator(cov)
        sim_data = simulator.simulate(10000)
        sim_cov = np.cov(sim_data, rowvar=False)
        self.assertAlmostEqual(0, manhattan_distance(cov - sim_cov), delta=0.3)
    
    def test_simulated_data_follows_normal_dist(self):
        cov = np.array([[1.3, 0.7], [0.7, 1]])
        simulator = PCASimulator(cov)
        sim_data = simulator.simulate(1000)
        self.assertTrue(examine_normality(sim_data))
 

class CopulaSimulationTest(TestCase):
    
    def test_distribution_of_simulated_data_is_close_enough(self):
        x1 = scipy.stats.t(loc=1, df=3, scale=1.5).rvs(100000)
        x23 = scipy.stats.multivariate_normal([0.5, -0.2], [[1.3, 0.7], [0.7, 1]]).rvs(100000)
        x = np.hstack([x1[:, np.newaxis], x23])
        cov = np.cov(x, rowvar=False)
        
        # generate simulated data
        dists = [scipy.stats.t(loc=1, df=3, scale=1.5),
                 scipy.stats.norm(loc=0.5, scale=1.3),
                 scipy.stats.norm(loc=-0.2, scale=1)]
        cpl = CopulaSimulator(x, dists)
        sim_x = cpl.simulate(100000)
        sim_cov = np.cov(sim_x, rowvar=False)
        
        # test the simulated data has almost the same covariance
        diff = manhattan_distance(sim_cov - cov)
        self.assertAlmostEqual(0, diff, delta=1)
        self.assertAlmostEqual(0.7, np.cov(sim_x[:, 1:], rowvar=False)[0, 1], delta=1e-1)
    
    
if __name__ == '__main__':
    main()
