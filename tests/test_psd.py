from unittest import TestCase, main
import numpy as np

from febrisk.psd import is_psd, near_psd, nearest_psd


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


class TestIsPsd(TestCase):

    def test_return_false_for_non_psd(self):
        non_psd = a_non_psd_matrix(10)
        self.assertFalse(is_psd(non_psd))

    def test_return_true_for_psd(self):
        psd = a_psd_matrix(10)
        self.assertTrue(is_psd(psd))


class TestNearPsd(TestCase):

    def test_result_is_the_same_with_lecture_code(self):
        non_psd = a_non_psd_matrix(5)
        result = near_psd(non_psd)
        answer =  np.array([[1.0,       0.735701,  0.899997,  0.899997,  0.899997],
                            [0.735701,  1.0,       0.899997,  0.899997,  0.899997],
                            [0.899997,  0.899997,  1.0,       0.9,       0.9],
                            [0.899997,  0.899997,  0.9,       1.0,       0.9],
                            [0.899997,  0.899997,  0.9,       0.9,       1.0]])
        difference = abs(result - answer).sum()
        self.assertAlmostEqual(0, difference, delta=1e-5)

    def test_result_is_psd(self):
        pass


class TestNearestPsd(TestCase):

    def test_result_is_the_same_with_lecture_code(self):
        non_psd = a_non_psd_matrix(5)
        result = nearest_psd(non_psd)
        answer = np.array([[1.0     ,  0.735704,  0.899998,  0.899998,  0.899998],
                           [0.735704,  1.0     ,  0.899998,  0.899998,  0.899998],
                           [0.899998,  0.899998,  1.0     ,  0.900001,  0.900001],
                           [0.899998,  0.899998,  0.900001,  1.0     ,  0.900001],
                           [0.899998,  0.899998,  0.900001,  0.900001,  1.0     ]])
        
        difference = abs(result - answer).sum()
        self.assertAlmostEqual(0, difference, delta=1e-5)
    
    def test_result_is_psd(self):
        pass

   
if __name__ == '__main__':
    main()
        
