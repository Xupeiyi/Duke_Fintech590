from unittest import TestCase, main

import numpy as np

from febrisk.stats import manhattan_distance
from febrisk.simulation import chol_psd


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
        

if __name__ == '__main__':
    main()
