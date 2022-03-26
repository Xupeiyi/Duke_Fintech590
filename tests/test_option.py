from unittest import TestCase, main
from febrisk.option import bsm, bt_american_continuous_div, bt_american


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
