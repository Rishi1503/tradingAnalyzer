import unittest
from unittest.mock import patch, MagicMock
from getStockListings import get_stock_listings

class stockListingsTest(unittest.TestCase):

    @patch('getStockListings.csv.writer')
    @patch('builtins.open', create=True)
    def test_output_file_generation(self, mock_open, mock_csv_writer):
        # Mock the open function and csv.writer to check if the output file is generated correctly
        mock_csv_writer.return_value = MagicMock()

        # Call the function under test
        get_stock_listings()

        # Check if the file is opened with the correct path and mode
        mock_open.assert_called_once_with('filtered_stocks.csv', 'w', newline='')

    def test_successful_execution(self):
        # Call the function under test
        stock_list = get_stock_listings()

        # Check if the function returns a non-empty list of stock symbols
        self.assertTrue(stock_list)
        self.assertIsInstance(stock_list, list)

if __name__ == '__main__':
    unittest.main()
