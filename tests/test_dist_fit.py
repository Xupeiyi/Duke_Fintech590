from unittest import TestCase, main
import scipy.stats
from febrisk.dist_fit import TFitter, NormalFitter


class TFitterTest(TestCase):
    
    def test_parameter_estimation_is_close_to_actual_value(self):
        loc, df, scale = 2, 10, 5
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)
        
        fitter = TFitter()
        fitter.fit(x=sim_data)
        sim_loc, sim_df, sim_scale = fitter.fitted_params
        self.assertTrue(loc-0.5 < sim_loc < loc+0.5)
        self.assertTrue(df-1 < sim_df < df+1)
        self.assertTrue(scale-0.5 < sim_scale < scale+0.5)
        
    def test_result_is_close_to_scipy_implementation(self):
        loc, df, scale = 1, 7, 9
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)
        
        fitter = TFitter()
        fitter.fit(x=sim_data, x0=(1, 1, 1))
        sim_loc, sim_df, sim_scale = fitter.fitted_params
        scipy_df, scipy_loc, scipy_scale = scipy.stats.t.fit(sim_data)
        
        delta = 5e-4
        self.assertAlmostEqual(scipy_loc, sim_loc, delta=delta)
        self.assertAlmostEqual(scipy_df, sim_df, delta=delta)
        self.assertAlmostEqual(scipy_scale, sim_scale, delta=delta)

    def test_constraints_are_updated(self):
        loc, df, scale = 10, 7, 9
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)

        fitter = TFitter(constraints=({"type": "eq", "fun": lambda x: x[0]},))
        fitter.fit(x=sim_data, x0=(10, 1, 1))
        sim_loc, _, _ = fitter.fitted_params

        self.assertEqual(0, sim_loc)

    def test_fitter_has_its_own_constraints_when_constructed(self):
        loc, df, scale = 10, 7, 9
        sim_data = scipy.stats.t(loc=loc, df=df, scale=scale).rvs(100000)

        fitter1 = TFitter(constraints=({"type": "eq", "fun": lambda x: x[0]-8},))
        fitter1.fit(x=sim_data, x0=(10, 1, 1))
        sim_loc, _, _ = fitter1.fitted_params
        self.assertEqual(8, sim_loc)

        fitter2 = TFitter(constraints=({"type": "eq", "fun": lambda x: x[0]-20},))
        fitter2.fit(x=sim_data, x0=(10, 1, 1))
        sim_loc, _, _ = fitter2.fitted_params
        self.assertEqual(20, sim_loc)


class NormalFitterTest(TestCase):
    
    def test_parameter_estimation_is_close_to_actual_value(self):
        loc, scale = 3, 6
        sim_data = scipy.stats.norm(loc=loc, scale=scale).rvs(100000)
        
        fitter = NormalFitter()
        fitter.fit(x=sim_data)
        sim_loc, sim_scale = fitter.fitted_params
        
        delta = 0.5
        self.assertAlmostEqual(loc, sim_loc, delta=delta)
        self.assertAlmostEqual(scale, sim_scale, delta=delta)
        
    def test_result_is_close_to_scipy_implementation(self):
        loc, scale = 3, 6
        sim_data = scipy.stats.norm(loc=loc, scale=scale).rvs(100000)
        
        fitter = NormalFitter()
        fitter.fit(x=sim_data)
        sim_loc, sim_scale = fitter.fitted_params
        scipy_loc, scipy_scale = scipy.stats.norm.fit(sim_data)
        
        delta = 0.5
        self.assertAlmostEqual(scipy_loc, sim_loc, delta=delta)
        self.assertAlmostEqual(scipy_scale, sim_scale, delta=delta)

    
if __name__ == '__main__':
    main()
