from datetime import datetime
from unittest import TestCase, main
from febrisk.option import (
    cal_ttm, bsm, bt_american,
    cal_euro_delta, cal_euro_delta_num,
    cal_euro_gamma, cal_euro_gamma_num,
    cal_euro_vega, cal_euro_vega_num,
    cal_euro_theta, cal_euro_theta_num,
    cal_euro_rho, cal_euro_rho_num,
    cal_euro_carry_rho, cal_euro_carry_rho_num
)


price = 165
strike = 165
r_f = 0.0025
div_rate = 0.0053
ivol = 0.2
curr_date = datetime(2022, 3, 13)
expire_date = datetime(2022, 4, 15)
ttm = cal_ttm(curr_date, expire_date)


class CalEuroDeltaTest(TestCase):
    
    def test_call(self):
        call_delta_closed = cal_euro_delta(True, price, strike, ttm, r_f, div_rate, ivol)
        call_delta_num = cal_euro_delta_num(True, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(call_delta_closed, call_delta_num)
    
    def test_put(self):
        put_delta_closed = cal_euro_delta(False, price, strike, ttm, r_f, div_rate, ivol)
        put_delta_num = cal_euro_delta_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(put_delta_closed, put_delta_num)
        

class CalEuroGammaTest(TestCase):
    
    def test_call_or_put(self):
        gamma_closed = cal_euro_gamma(price, strike, ttm, r_f, div_rate, ivol)
        gamma_num1 = cal_euro_gamma_num(True, price, strike, ttm, r_f, div_rate, ivol)
        gamma_num2 = cal_euro_gamma_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(gamma_closed, gamma_num1)
        self.assertAlmostEqual(gamma_closed, gamma_num2)
    

class CalEuroVegaTest(TestCase):
    
    def test_call_or_put(self):
        vega_closed = cal_euro_vega(price, strike, ttm, r_f, div_rate, ivol)
        vega_num1 = cal_euro_vega_num(True, price, strike, ttm, r_f, div_rate, ivol)
        vega_num2 = cal_euro_vega_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(vega_closed, vega_num1, 6)
        self.assertAlmostEqual(vega_closed, vega_num2, 6)


class CalEuroThetaTest(TestCase):
    
    def test_call(self):
        call_theta_closed = cal_euro_theta(True, price, strike, ttm, r_f, div_rate, ivol)
        call_theta_num = cal_euro_theta_num(True, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(call_theta_closed, call_theta_num, 3)
    
    def test_put(self):
        put_theta_closed = cal_euro_theta(False, price, strike, ttm, r_f, div_rate, ivol)
        put_theta_num = cal_euro_theta_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(put_theta_closed, put_theta_num, 3)
    

class CalEuroRhoTest(TestCase):
    
    def test_call(self):
        call_rho_closed = cal_euro_rho(True, price, strike, ttm, r_f, div_rate, ivol)
        call_rho_num = cal_euro_rho_num(True, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(call_rho_closed, call_rho_num, 3)
        
    def test_put(self):
        put_rho_closed = cal_euro_rho(False, price, strike, ttm, r_f, div_rate, ivol)
        put_rho_num = cal_euro_rho_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(put_rho_closed, put_rho_num, 3)
    
    
class CalEuroCarryRhoTest(TestCase):
    
    def test_call(self):
        call_carry_rho_closed = cal_euro_carry_rho(True, price, strike, ttm, r_f, div_rate, ivol)
        call_carry_rho_num = cal_euro_carry_rho_num(True, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(call_carry_rho_closed, call_carry_rho_num, 6)

    def test_put(self):
        put_carry_rho_closed = cal_euro_carry_rho(False, price, strike, ttm, r_f, div_rate, ivol)
        put_carry_rho_num = cal_euro_carry_rho_num(False, price, strike, ttm, r_f, div_rate, ivol)
        self.assertAlmostEqual(put_carry_rho_closed, put_carry_rho_num, 6)
        
    
class BTAmericanTest(TestCase):
    def test_result_is_correct_with_continuous_dividends(self):
        ans = bt_american(is_call=False, price=100, strike=100, ttm=0.5, r_f=0.08, 
                          div_rate=0.02, ivol=0.3, nperiods=100)
        self.assertAlmostEqual(7.15591, ans, 5)

        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5, r_f=0.08, 
                          div_rate=0.04, ivol=0.3, nperiods=2)
        self.assertAlmostEqual(8.267778, ans, 5)

        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5, r_f=0.08, 
                          div_rate=0.04, ivol=0.3, nperiods=100)
        self.assertAlmostEqual(9.183950, ans, 5)

    def test_result_is_correct_with_discrete_dividends(self):
        # call
        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5,r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=2, dividends=[(1, 1.0)])
        self.assertAlmostEqual(9.116786, ans, 5)

        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5,r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=100, dividends=[(50, 1.0)])
        self.assertAlmostEqual(9.842210, ans, 5)

        ans = bt_american(is_call=True, price=100, strike=100,ttm=0.5,r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=100, dividends=[(1, 1.0)])
        self.assertAlmostEqual(9.79070, ans, 5)

        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5, r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=100, dividends=[(1, 1.0), (2, 1.0)])
        self.assertAlmostEqual(9.215344, ans, 5)

        # put
        ans = bt_american(is_call=False, price=120, strike=100, ttm=0.5,r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=2, dividends=[(1, 1.0)])
        self.assertAlmostEqual(2.54262, ans, 5)
    
    def test_corner_cases(self):
        ans = bt_american(is_call=True, price=100, strike=100, ttm=0.5, r_f=0.08, 
                          div_rate=0, ivol=0.3, nperiods=100, dividends=[(0, 1.0)])
        self.assertAlmostEqual(9.789634, ans, 5)

        ans1 = bt_american(is_call=True, price=99, strike=100, ttm=0.5, r_f=0.08, 
                           div_rate=0, ivol=0.3, nperiods=100)
        self.assertAlmostEqual(ans, ans1, 5)


if __name__ == '__main__':
    main()
