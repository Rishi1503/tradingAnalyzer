from datapackage import Package
import csv

def get_stock_listings():

    package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')

    # Initialize an empty array to store the stock symbols
    stock_symbols = []
    # Iterate through the resources
    for resource in package.resources:
        # Check if the resource is processed tabular data in CSV format
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            # Read the processed tabular data
            data = resource.read()

            # Iterate through the data rows
            for row in data:
                # Get the stock symbol from the appropriate column (e.g., 'symbol')
                stock_symbol = row[0]

                # Append the stock symbol to the array
                stock_symbols.append(stock_symbol)
    package = Package('https://datahub.io/core/nyse-other-listings/datapackage.json')
    print(len(stock_symbols))
    # Iterate through the resources
    for resource in package.resources:
        # Check if the resource is processed tabular data in CSV format
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            # Read the processed tabular data
            data = resource.read()

            # Iterate through the data rows
            for row in data:
                # Get the stock symbol from the appropriate column (e.g., 'symbol')
                stock_symbol = row[0]

                # Append the stock symbol to the array
                stock_symbols.append(stock_symbol)
    print(len(stock_symbols))
    stock_symbols = list(set(stock_symbols))
    print(len(stock_symbols))
    return stock_screener(stock_symbols)

def stock_screener(stock_symbols):
    filtered_stocks = []
    output_file = 'filtered_stocks.csv'  # Specify the output file path
    for symbol in stock_symbols:
        try:
            data = yf.download(symbol, period='1mo', interval='1wk')

            # Check share price
            current_price = data['Close'][-1]
            # Check weekly percentage change
            weekly_pct_change = (data['Close'][-1] - data['Close'][-2]) / data['Close'][-2] * 100

            if current_price > 20 and (weekly_pct_change > 2 or weekly_pct_change < 5):
                filtered_stocks.append(symbol)  # Add the stock to the filtered list
        except Exception as e:
            print({e})
    # Write filtered stocks to a file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol'])  # Write the header
        for stock in filtered_stocks:
            writer.writerow([stock])  # Write each filtered stock symbol
    print("Stock screener done: ", len(filtered_stocks))
    return filtered_stocks

def getFinalStockList():
    stocks = get_stock_listings()
    for stock in stocks:
        print(stock)
    return stocks

getFinalStockList()