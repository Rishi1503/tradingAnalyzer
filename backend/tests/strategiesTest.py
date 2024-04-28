import unittest
from unittest.mock import patch
from strategies import check_all_stocks

class TestCheckAllStocks(unittest.TestCase):
    
    @patch('strategies.yf.download')
    @patch('strategies.trading_client.get_asset')
    @patch('strategies.trading_client.get_all_positions')
    @patch('builtins.open', unittest.mock.mock_open(read_data='AAPL\nMSFT\nGOOG\n'))
    def test_normal_execution(self, mock_positions, mock_asset, mock_download):
        # Mocking data for testing
        mock_positions.return_value = []
        mock_asset.return_value.tradable = True
        mock_download.return_value = MagicMock()
        
        # Call the function
        result = check_all_stocks('input.csv')
        
        # Assert the result
        self.assertEqual(result, ['AAPL', 'MSFT', 'GOOG'])

    # Add more test cases here based on the outlined scenarios

if __name__ == '__main__':
    unittest.main()
