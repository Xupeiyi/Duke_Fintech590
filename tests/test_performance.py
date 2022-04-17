from unittest import TestCase, main
import numpy as np
import pandas as pd
from febrisk.performance import update_weights
from febrisk.math import manhattan_distance


class UpdateWeightsTests(TestCase):
    
    def test_result_is_correct(self):
        weights = np.array([0.10075988181538116, 0.20950981862533438,
                            0.43839111238558587, 0.08118475735284336,
                            0.17015442982085532])
        returns = pd.read_csv("UpdatedReturns.csv", header=None).values
        ans = pd.read_csv("UpdatedWeights.csv", header=None).values
        result = update_weights(weights, returns)
        self.assertAlmostEqual(0, manhattan_distance(result - ans))


if __name__ == '__main__':
    main()
