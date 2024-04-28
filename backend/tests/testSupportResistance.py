import unittest
import pandas as pd
from support_resistance import is_Suppport_Level, is_Resistance_Level, compare_price_levels, get_support_resistance_levels, check_1_3_6_months

class TestSupportResistance(unittest.TestCase):

    def test_is_Support_Level(self):
        # Define test data
        df = pd.DataFrame({'Low': [100, 90, 80, 85, 95]})
        i = 2

        # Call the function under test
        result = is_Suppport_Level(df, i)

        # Assert the expected result
        self.assertTrue(result)

    def test_is_Resistance_Level(self):
        # Define test data
        df = pd.DataFrame({'High': [75, 85, 95, 90, 80]})
        i = 2

        # Call the function under test
        result = is_Resistance_Level(df, i)

        # Assert the expected result
        self.assertTrue(result)

    def test_compare_price_levels(self):
        # Define test data
        resistance_levels = [(0, 100), (1, 110)]
        support_levels = [(2, 90), (3, 80)]
        current_price = 95
        mean = 5

        # Call the function under test
        result = compare_price_levels(resistance_levels, support_levels, current_price, mean)

        # Assert the expected result
        self.assertEqual(result, 'Maybe')

    def test_get_support_resistance_levels(self):
        # Define test data
        ticker_symbol = 'AAPL'
        start_date = '2023-01-01'
        end_date = '2023-02-01'

        # Call the function under test
        result = get_support_resistance_levels(ticker_symbol, start_date, end_date)

        # Assert the expected result
        self.assertIn(result, ['Yes', 'No', 'Maybe'])

    def test_check_1_3_6_months(self):
        # Define test data
        ticker_symbols = ['AAPL', 'GOOGL', 'MSFT']

        # Call the function under test
        result_buy, result_waitlist = check_1_3_6_months(ticker_symbols)

        # Assert the expected result
        self.assertIsInstance(result_buy, list)
        self.assertIsInstance(result_waitlist, list)

if __name__ == '__main__':
    unittest.main()
