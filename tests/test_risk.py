from unittest import TestCase, main

import numpy as np

from febrisk.risk import VaR, expected_shortfall


class VaRTest(TestCase):
    
    def test_can_work_on_1d_array(self):
        pass
    
    def test_can_work_on_2d_array(self):
        pass
    
    
class ExpectedShortfallTest(TestCase):
    
    def test_can_work_on_1d_array(self):
        data_1d = np.array([5, 4, 3, 8, 9, 1])
        result = expected_shortfall(data_1d, 0.6)
        self.assertAlmostEqual((1+3+4)/3, result)
    
    # def test_can_work_on_2d_array(self):
    #     data_2d = np.array([[7, 5, 6, 3],
    #                         [2, 3, 4, 6],
    #                         [8, 3, 5, 2]])
    #     result = expected_shortfall(data_2d, 0.6)
    #     answer = np.array([4, 2.5, 2.5])
