import unittest
from unittest.mock import MagicMock
import datetime
from alpaca_orders import sellstock, sellstock_instant, buystock, set_sell_orders, check_positions

class TestTradingFunctions(unittest.TestCase):

    def setUp(self):
        # Mock the TradingClient and Account objects
        self.mock_trading_client = MagicMock()
        self.mock_account = MagicMock()
        self.mock_position = MagicMock()
        self.mock_position.symbol = 'AAPL'
        self.mock_position.market_value = '1000'
        self.mock_position.cost_basis = '950'
        self.mock_position.qty = 10
        self.mock_trading_client.get_all_positions.return_value = [self.mock_position]
        self.mock_trading_client.get_account.return_value = self.mock_account

    def test_sellstock(self):
        # Call the function under test
        sellstock('AAPL', 5)

        # Assert that submit_order is called with the correct parameters
        self.mock_trading_client.submit_order.assert_called_once_with(
            order_data=MagicMock(symbol='AAPL', type="trailing_stop", qty=5, side='SELL', trail_percent=1, time_in_force='DAY')
        )

    def test_sellstock_instant(self):
        # Call the function under test
        sellstock_instant('AAPL', 5)

        # Assert that submit_order is called with the correct parameters
        self.mock_trading_client.submit_order.assert_called_once_with(
            order_data=MagicMock(symbol='AAPL', qty=5, side='SELL', time_in_force='DAY')
        )

    def test_buystock(self):
        # Call the function under test
        buystock('AAPL')

        # Assert that submit_order is called with the correct parameters
        self.mock_trading_client.submit_order.assert_called_once_with(
            order_data=MagicMock(symbol='AAPL', notional=500, side='BUY', time_in_force='DAY')
        )

    def test_set_sell_orders(self):
        # Call the function under test
        set_sell_orders('DAY')

        # Assert that submit_order is called with the correct parameters for each position
        self.mock_trading_client.submit_order.assert_called_once_with(
            order_data=MagicMock(symbol='AAPL', qty=10, side='SELL', trail_percent=1, time_in_force='DAY')
        )

    def test_check_positions(self):
        # Call the function under test
        check_positions()

        # Assert that submit_order is called with the correct parameters for each position
        self.mock_trading_client.submit_order.assert_called_once_with(
            order_data=MagicMock(symbol='AAPL', qty=10, side='SELL', time_in_force='DAY')
        )

if __name__ == '__main__':
    unittest.main()
