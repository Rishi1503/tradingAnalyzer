import yfinance as yf

def check_moving_average(symbol, period=25):
    # Get the historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the moving average
    stock_data['MA'] = stock_data['Close'].rolling(window=period).mean()
    
    # Get the last closing price
    current_price = stock_data['Close'][-1]
    
    # Get the last moving average value
    last_ma = stock_data['MA'][-1]
    
    # Determine if the stock should be bought or sold
    if current_price > last_ma:
        return f"{symbol}: Buy"
    else:
        return f"{symbol}: Sell"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = check_moving_average(stock_symbol)
print(result)

def check_bullish_pattern(symbol, short_period=10, long_period=30):
    # Get the historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the moving averages
    stock_data['short_MA'] = stock_data['Close'].rolling(window=short_period).mean()
    stock_data['long_MA'] = stock_data['Close'].rolling(window=long_period).mean()
    
    # Get the last values of the moving averages
    last_short_ma = stock_data['short_MA'][-1]
    last_long_ma = stock_data['long_MA'][-1]
    
    # Determine if a bullish pattern exists
    if last_short_ma > last_long_ma:
        return f"{symbol}: Bullish pattern detected (Golden Cross)"
    else:
        return f"{symbol}: No bullish pattern detected"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = check_bullish_pattern(stock_symbol)
print(result)

import yfinance as yf

def check_momentum(symbol, period=14):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the rate of change (ROC) over the specified period
    stock_data['ROC'] = stock_data['Close'].pct_change(period)
    
    # Get the last ROC value
    last_roc = stock_data['ROC'][-1]
    
    # Determine if the momentum indicates upward or downward movement
    if last_roc > 0:
        return f"{symbol}: Momentum indicates upward movement"
    elif last_roc < 0:
        return f"{symbol}: Momentum indicates downward movement"
    else:
        return f"{symbol}: No significant momentum detected"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = check_momentum(stock_symbol)
print(result)

#dont use this too often!!! breakout strategy
def should_buy_stock(symbol, lookback_period=20, breakout_threshold=0.02):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the price range (high-low) over the lookback period
    stock_data['Range'] = stock_data['High'].rolling(window=lookback_period).max() - stock_data['Low'].rolling(window=lookback_period).min()
    
    # Get the last price range value
    last_range = stock_data['Range'][-1]
    
    # Calculate the breakout level
    breakout_level = stock_data['Close'][-1] + (breakout_threshold * last_range)
    
    # Determine if the current price is above the breakout level
    if stock_data['Close'][-1] > breakout_level:
        return f"{symbol}: Buy the stock (Price > Breakout Level)"
    else:
        return f"{symbol}: Do not buy the stock (Price < Breakout Level)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = should_buy_stock(stock_symbol)
print(result)

def check_on_balance_volume(symbol):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate OBV
    stock_data['OBV'] = (stock_data['Volume'] * ((stock_data['Close'] - stock_data['Open']) / stock_data['Open'])).cumsum()
    
    # Determine if the OBV is increasing or decreasing
    obv_trend = stock_data['OBV'].diff().fillna(0)
    
    # Check the most recent OBV trend
    if obv_trend[-1] > 0:
        return f"{symbol}: Buy the stock (OBV indicates buying pressure)"
    else:
        return f"{symbol}: Do not buy the stock (OBV does not indicate buying pressure)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = check_on_balance_volume(stock_symbol)
print(result)

#Accumulation/Distribution line 
def should_buy_or_sell_stock_with_adline(symbol):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the A/D Line
    stock_data['A/D Line'] = ((stock_data['Close'] - stock_data['Low']) - (stock_data['High'] - stock_data['Close'])) / (stock_data['High'] - stock_data['Low'])
    stock_data['A/D Line'] *= stock_data['Volume']
    stock_data['A/D Line'] = stock_data['A/D Line'].cumsum()
    
    # Determine the most recent A/D Line value
    adline_value = stock_data['A/D Line'][-1]
    
    # Check if the A/D Line suggests buying or selling
    if adline_value > 0:
        return f"{symbol}: Buy the stock (A/D Line indicates buying pressure)"
    elif adline_value < 0:
        return f"{symbol}: Sell the stock (A/D Line indicates selling pressure)"
    else:
        return f"{symbol}: Do not take any action (A/D Line is neutral)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = should_buy_or_sell_stock_with_adline(stock_symbol)
print(result)

#dont rely too much on it
def should_buy_stock_with_adx(symbol):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the ADX and Directional Indicators (+DI and -DI)
    stock_data['DM+'] = stock_data['High'].diff()
    stock_data['DM-'] = stock_data['Low'].diff().abs()
    stock_data['TR'] = stock_data[['High', 'Low', 'Close']].max(axis=1) - stock_data[['High', 'Low', 'Close']].min(axis=1)
    
    stock_data['+DI'] = (stock_data['DM+'] > stock_data['DM-']) & (stock_data['DM+'] > 0) * stock_data['DM+']
    stock_data['-DI'] = (stock_data['DM-'] > stock_data['DM+']) & (stock_data['DM-'] > 0) * stock_data['DM-']
    
    stock_data['+DI14'] = stock_data['+DI'].rolling(window=14).sum() / stock_data['TR'].rolling(window=14).sum() * 100
    stock_data['-DI14'] = stock_data['-DI'].rolling(window=14).sum() / stock_data['TR'].rolling(window=14).sum() * 100
    
    stock_data['DX'] = abs(stock_data['+DI14'] - stock_data['-DI14']) / (stock_data['+DI14'] + stock_data['-DI14']) * 100
    stock_data['ADX'] = stock_data['DX'].rolling(window=14).mean()
    
    # Determine the most recent ADX and Directional Indicator values
    adx_value = stock_data['ADX'][-1]
    di_plus = stock_data['+DI14'][-1]
    di_minus = stock_data['-DI14'][-1]
    
    # Check the ADX and Directional Indicator values to determine the trend
    if adx_value > 20:
        if di_plus > di_minus:
            return f"{symbol}: Buy the stock (Uptrend - ADX above 20 and DI+ above DI-)"
        elif di_minus > di_plus:
            return f"{symbol}: Do not buy the stock (Downtrend - ADX above 20 and DI- above DI+)"
    else:
        return f"{symbol}: Do not take any action (Weak trend or ranging period - ADX below 20)"
        

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = should_buy_stock_with_adx(stock_symbol)
print(result)

def is_stock_worth_buying_with_aroon(symbol, period=15):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the Aroon-Up and Aroon-Down
    stock_data['High_period'] = stock_data['High'].rolling(window=period+1).max()
    stock_data['Low_period'] = stock_data['Low'].rolling(window=period+1).min()
    
    stock_data['Aroon-Up'] = ((period - stock_data['High'].rolling(window=period+1).apply(lambda x: x.argmax())) / period) * 100
    stock_data['Aroon-Down'] = ((period - stock_data['Low'].rolling(window=period+1).apply(lambda x: x.argmin())) / period) * 100
    
    # Determine the most recent Aroon values
    aroon_up = stock_data['Aroon-Up'][-1]
    aroon_down = stock_data['Aroon-Down'][-1]
    
    # Check if the Aroon values suggest buying the stock
    if aroon_up > 80 and aroon_down < 20:
        return f"{symbol}: Buy the stock (Aroon-Up is above 80 and Aroon-Down is below 20)"
    else:
        return f"{symbol}: Do not buy the stock (Aroon indicator does not suggest buying opportunity)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = is_stock_worth_buying_with_aroon(stock_symbol)
print(result)

#don't use yet
def is_stock_worth_buying_with_rsi(symbol, period=7, oversold_threshold=30, overbought_threshold=70):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the RSI
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Determine the most recent RSI value
    current_rsi = rsi[-1]
    
    # Check if the RSI value suggests buying the stock
    if current_rsi > overbought_threshold:
        return f"{symbol}: Do not buy the stock (RSI is overbought)"
    elif current_rsi < oversold_threshold:
        return f"{symbol}: Buy the stock (RSI is oversold)"
    else:
        return f"{symbol}: Do not take any action (RSI does not suggest clear buying opportunity)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = is_stock_worth_buying_with_rsi(stock_symbol, period=7)
print(result)

def is_stock_worth_buying_with_vwap(symbol):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate VWAP
    stock_data['Typical Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    stock_data['VWAP'] = (stock_data['Typical Price'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
    
    # Get the most recent VWAP value
    current_vwap = stock_data['VWAP'][-1]
    
    # Check if the stock price is above or below VWAP
    current_price = stock_data['Close'][-1]
    if current_price > current_vwap:
        return f"{symbol}: Buy the stock (Price is above VWAP)"
    else:
        return f"{symbol}: Do not buy the stock (Price is below VWAP)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = is_stock_worth_buying_with_vwap(stock_symbol)
print(result)

def is_stock_worth_buying_with_ema(symbol, short_period=10, long_period=30):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Calculate the exponential moving averages
    stock_data['EMA_short'] = stock_data['Close'].ewm(span=short_period, adjust=False).mean()
    stock_data['EMA_long'] = stock_data['Close'].ewm(span=long_period, adjust=False).mean()
    
    # Get the most recent EMA values
    current_ema_short = stock_data['EMA_short'][-1]
    current_ema_long = stock_data['EMA_long'][-1]
    
    # Check if the short-term EMA is above the long-term EMA
    if current_ema_short > current_ema_long:
        return f"{symbol}: Buy the stock (Short-term EMA is above Long-term EMA)"
    else:
        return f"{symbol}: Do not buy the stock (Short-term EMA is below Long-term EMA)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = is_stock_worth_buying_with_ema(stock_symbol, short_period=10, long_period=30)
print(result)

def get_fibonacci_levels(high, low):
    # Calculate the Fibonacci retracement levels
    fibonacci_levels = {}
    range_high = high - low
    fibonacci_levels[23.6] = high - (range_high * 0.236)
    fibonacci_levels[38.2] = high - (range_high * 0.382)
    fibonacci_levels[61.8] = high - (range_high * 0.618)
    fibonacci_levels[78.6] = high - (range_high * 0.786)
    return fibonacci_levels

def is_stock_worth_buying_with_fibonacci(symbol):
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    # Get the highest and lowest price
    high_price = stock_data['High'].max()
    low_price = stock_data['Low'].min()
    
    # Calculate the Fibonacci retracement levels
    fibonacci_levels = get_fibonacci_levels(high_price, low_price)
    
    # Get the most recent closing price
    current_close = stock_data['Close'][-1]
    
    # Check if the current price is near any Fibonacci level
    for level, value in fibonacci_levels.items():
        if abs(current_close - value) <= (value * 0.02):  # Allowing a 2% tolerance
            if current_close > value:
                return f"{symbol}: Buy the stock (Price is near Fibonacci level {level}%)"
            else:
                return f"{symbol}: Sell the stock (Price is near Fibonacci level {level}%)"
    
    return f"{symbol}: Do not take any action (Price is not near any Fibonacci levels)"

# Example usage
stock_symbol = "AAPL"  # Replace with the desired stock symbol
result = is_stock_worth_buying_with_fibonacci(stock_symbol)
print(result)


# def test_analyze_stock():
#     test_cases = [
#         {"symbol": "AAPL", "expected_result": "AAPL: Buy"},
#         {"symbol": "GOOGL", "expected_result": "GOOGL: Sell"},
#         # Add more test cases here
#     ]
    
#     for test_case in test_cases:
#         symbol = test_case["symbol"]
#         expected_result = test_case["expected_result"]
        
#         result = check_moving_average(symbol)
        
#         # Check if the result matches the expected result
#         if result == expected_result:
#             print(f"Test case passed for symbol {symbol}")
#         else:
#             print(f"Test case failed for symbol {symbol}. Expected: {expected_result}. Got: {result}")

# # Run the test
# test_analyze_stock()
