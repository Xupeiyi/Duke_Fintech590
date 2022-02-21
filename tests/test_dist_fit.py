from unittest import TestCase, main
import scipy.stats
from febrisk.dist_fit import TFitter


class TFitterTest(TestCase):
    
    def test_parameter_estimation_is_close_to_actual_value(self):
        loc, df, scale = 2, 10, 5
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)
        
        sim_loc, sim_df, sim_scale = TFitter().fit(x=sim_data, x0=(1, 1, 1)).x
        self.assertTrue(loc-0.3 < sim_loc < loc+0.3)
        self.assertTrue(df-1 < sim_df < df+1)
        self.assertTrue(scale-0.2 < sim_scale < scale+0.2)
        
    def test_result_is_close_to_scipy_implementation(self):
        loc, df, scale = 1, 7, 9
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)
        
        sim_loc, sim_df, sim_scale = TFitter().fit(x=sim_data, x0=(1, 1, 1)).x
        scipy_df, scipy_loc, scipy_scale = scipy.stats.t.fit(sim_data)
        
        self.assertAlmostEqual(scipy_loc, sim_loc, delta=1e-4)
        self.assertAlmostEqual(scipy_df, sim_df, delta=1e-4)
        self.assertAlmostEqual(scipy_scale, sim_scale, delta=1e-4)
        
    
if __name__ == '__main__':
    main()