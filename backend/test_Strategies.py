import yfinance as yf
import datetime as dt
import pandas as pd
import warnings
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')

#modification required
def check_macd(data):
    # Calculate MACD using exponential moving averages
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9).mean()

    # Get the most recent MACD and Signal Line values
    current_macd = data['MACD'][-1]
    current_signal_line = data['Signal Line'][-1]

    # Check if MACD crosses above the Signal Line (buy signal)
    if current_macd > current_signal_line:
        return True
    else:
        return False

def check_moving_average(data, period=10):
    # Calculate the moving average
    data['MA'] = data['Close'].rolling(window=period).mean()

    # Get the last closing price
    current_price = data['Close'].iloc[-1]

    # Get the last moving average value
    last_ma = data['MA'].iloc[-1]

    # Determine if the stock should be bought or sold
    if current_price > last_ma:
        return True
    else:
        return False
    
def check_moving_average2(data, period=5):
    # Calculate the moving average
    data['MA'] = data['Close'].rolling(window=period).mean()

    # Get the last closing price
    current_price = data['Close'].iloc[-1]

    # Get the last moving average value
    last_ma = data['MA'].iloc[-1]

    # Determine if the stock should be bought or sold
    if current_price > last_ma:
        return True
    else:
        return False

def check_bullish_pattern(stock_data, short_period=5, long_period=10):
    
    # Calculate the moving averages
    stock_data['short_MA'] = stock_data['Close'].rolling(window=short_period).mean()
    stock_data['long_MA'] = stock_data['Close'].rolling(window=long_period).mean()
    
    # Get the last values of the moving averages
    last_short_ma = stock_data['short_MA'][-1]
    last_long_ma = stock_data['long_MA'][-1]
    
    # Determine if a bullish pattern exists
    if last_short_ma > last_long_ma:
        return True
    else:
        return False

def check_bullish_pattern2(stock_data, short_period=3, long_period=5):
    
    # Calculate the moving averages
    stock_data['short_MA'] = stock_data['Close'].rolling(window=short_period).mean()
    stock_data['long_MA'] = stock_data['Close'].rolling(window=long_period).mean()
    
    # Get the last values of the moving averages
    last_short_ma = stock_data['short_MA'][-1]
    last_long_ma = stock_data['long_MA'][-1]
    
    # Determine if a bullish pattern exists
    if last_short_ma > last_long_ma:
        return True
    else:
        return False

def check_momentum(stock_data, period=10):
    
    # Calculate the rate of change (ROC) over the specified period
    stock_data['ROC'] = stock_data['Close'].pct_change(period)
    
    # Get the last ROC value
    last_roc = stock_data['ROC'][-1]
    
    # Determine if the momentum indicates upward or downward movement
    if last_roc > 0:
        return True
    elif last_roc < 0:
        return False
    else:
        return False
    
#not using right now
def check_on_balance_volume(stock_data):
    
    # Calculate the On-Balance Volume (OBV)
    stock_data['OBV'] = (stock_data['Close'] - stock_data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1).cumsum()
    
    # Get the most recent OBV value
    current_obv = stock_data['OBV'][-1]
    
    # Get the most recent closing price
    current_close = stock_data['Close'][-1]
    
    # Check if the OBV and price are in agreement
    if current_obv > 0 and current_close > stock_data['Close'].mean():
        return True
    else:
        return False

def check_ema(stock_data, short_period=5, long_period=10):
    
    # Calculate the exponential moving averages
    stock_data['EMA_short'] = stock_data['Close'].ewm(span=short_period, adjust=False).mean()
    stock_data['EMA_long'] = stock_data['Close'].ewm(span=long_period, adjust=False).mean()
    
    # Get the most recent EMA values
    current_ema_short = stock_data['EMA_short'][-1]
    current_ema_long = stock_data['EMA_long'][-1]
    
    # Check if the short-term EMA is above the long-term EMA
    if current_ema_short > current_ema_long:
        return True
    else:
        return False

def check_vwap(stock_data):
    
    # Calculate VWAP
    stock_data['Typical Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    stock_data['VWAP'] = (stock_data['Typical Price'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
    
    # Get the most recent VWAP value
    current_vwap = stock_data['VWAP'][-1]
    
    # Check if the stock price is above or below VWAP
    current_price = stock_data['Close'][-1]
    if current_price > current_vwap:
        return True
    else:
        return f"{symbol}: Do not buy the stock (Price is below VWAP)"

#don't use this
def check_atr(stock_data, atr_period=14, atr_multiplier=2.0):
    # Calculate the Average True Range (ATR)
    # Get historical data for the stock
    stock_data = yf.download(symbol)
    
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High-PrevClose'] = abs(stock_data['High'] - stock_data['Close'].shift())
    stock_data['Low-PrevClose'] = abs(stock_data['Low'] - stock_data['Close'].shift())
    stock_data['TR'] = stock_data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    stock_data['ATR'] = stock_data['TR'].rolling(window=atr_period).mean()

    # Get the most recent ATR value
    current_atr = stock_data['ATR'].iloc[-1]

    # Calculate the buy threshold
    buy_threshold = current_atr * atr_multiplier

    # Get the most recent closing price
    current_close = stock_data['Close'].iloc[-1]

    # Check if the current price is below the buy threshold
    if current_close < buy_threshold:
        return True
    else:
        return False

def calculate_zigzag(data, deviation_percentage):
    close_prices = data['Close'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values

    current_deviation = deviation_percentage / 100.0
    current_high = high_prices[0]
    current_low = low_prices[0]
    zigzag_points = []

    for i in range(1, len(close_prices)):
        if high_prices[i] >= current_high * (1 + current_deviation) or low_prices[i] <= current_low * (1 - current_deviation):
            zigzag_points.append(i - 1)
            current_deviation = -current_deviation
            if current_deviation > 0:
                current_high = high_prices[i]
            else:
                current_low = low_prices[i]

        if current_deviation > 0 and high_prices[i] > current_high:
            current_high = high_prices[i]
        elif current_deviation < 0 and low_prices[i] < current_low:
            current_low = low_prices[i]

    zigzag_points.append(len(close_prices) - 1)

    return zigzag_points

def check_zigzag(stock_data, deviation_percentage=9.0):
    zigzag_points = calculate_zigzag(stock_data, deviation_percentage)

    # Get the most recent ZigZag point
    current_zigzag_point = zigzag_points[-1]

    # Check if the current index is the most recent ZigZag high or low
    if current_zigzag_point == len(stock_data) - 1:
        return True
    else:
        return False

def check_adline(stock_data):
    
    # Calculate the A/D Line
    stock_data['A/D Line'] = ((stock_data['Close'] - stock_data['Low']) - (stock_data['High'] - stock_data['Close'])) / (stock_data['High'] - stock_data['Low'])
    stock_data['A/D Line'] *= stock_data['Volume']
    stock_data['A/D Line'] = stock_data['A/D Line'].cumsum()
    
    # Determine the most recent A/D Line value
    adline_value = stock_data['A/D Line'][-1]
    
    # Check if the A/D Line suggests buying or selling
    if adline_value > 0:
        return True
    elif adline_value < 0:
        return False
    else:
        return False

def check_eps(symbol):
    # Download stock data using yfinance
    stock_data = yf.Ticker(symbol)

    # Get the EPS data
    eps_data = stock_data.earnings

    # Get the most recent EPS value
    current_eps = eps_data['Earnings'].iloc[-1]

    # Define a threshold for positive EPS
    eps_threshold = 0.0  # Adjust this threshold based on your preference

    # Check if the current EPS is above the threshold
    if current_eps > eps_threshold:
        return True  # Buy signal
    else:
        return False  # Do not buy signal

def check_bollinger_bands(stock_data, window=10, deviations=2):

    # Calculate the rolling mean and standard deviation
    stock_data['MA'] = stock_data['Close'].rolling(window).mean()
    stock_data['STD'] = stock_data['Close'].rolling(window).std()

    # Calculate the upper and lower Bollinger Bands
    stock_data['Upper'] = stock_data['MA'] + deviations * stock_data['STD']
    stock_data['Lower'] = stock_data['MA'] - deviations * stock_data['STD']

    # Get the most recent price and Bollinger Bands
    current_price = stock_data['Close'][-1]
    current_upper = stock_data['Upper'][-1]
    current_lower = stock_data['Lower'][-1]

    # Check if the price crosses above the upper band (buy signal)
    if current_price > current_upper:
        return True
    
    # Check if the price crosses below the lower band (sell signal)
    elif current_price < current_lower:
        return False
    
    # No action (hold signal)
    else:
        return True

def check_bollinger_bands2(stock_data, window=5, deviations=2):

    # Calculate the rolling mean and standard deviation
    stock_data['MA'] = stock_data['Close'].rolling(window).mean()
    stock_data['STD'] = stock_data['Close'].rolling(window).std()

    # Calculate the upper and lower Bollinger Bands
    stock_data['Upper'] = stock_data['MA'] + deviations * stock_data['STD']
    stock_data['Lower'] = stock_data['MA'] - deviations * stock_data['STD']

    # Get the most recent price and Bollinger Bands
    current_price = stock_data['Close'][-1]
    current_upper = stock_data['Upper'][-1]
    current_lower = stock_data['Lower'][-1]

    # Check if the price crosses above the upper band (buy signal)
    if current_price > current_upper:
        return True
    
    # Check if the price crosses below the lower band (sell signal)
    elif current_price < current_lower:
        return False
    
    # No action (hold signal)
    else:
        return True

def check_fundamental_indicators(symbol):
    # Download fundamental data for the stock
    stock_info = yf.Ticker(symbol).info

    # Get the PE ratio
    pe_ratio = stock_info.get('trailingPE')
    
    # Get the ROE and ROCE
    roe = stock_info.get('returnOnEquity')
    roce = stock_info.get('returnOnCapitalEmployed')

    # Check the conditions for buying or selling based on the fundamental indicators
    if pe_ratio is not None and roe is not None and roce is not None:
        if pe_ratio < 15 and roe > 0.1 and roce > 0.1:
            return True
        else:
            return False
    else:
        return True

#dont use
def check_stochastic_oscillator(stock_data, period=7, oversold_threshold=20, overbought_threshold=80):
    # Calculate the highest and lowest prices over the specified period
    high_max = stock_data['High'].rolling(window=period).max()
    low_min = stock_data['Low'].rolling(window=period).min()

    # Calculate the stochastic oscillator values
    k = 100 * (stock_data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()  # Using a 3-day simple moving average for %D line

    # Check if the stochastic oscillator values suggest buying the stock
    if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] > overbought_threshold:
        return False
    elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] < oversold_threshold:
        return True
    else:
        return False


# def simulate_trades(symbol, start_date, end_date):
#     # Download historical data for the stock
#     stock_data = yf.download(symbol, start=start_date, end=end_date)

#     # Initialize variables
#     position = None
#     buy_price = 0
#     capital = 10000  # Initial capital
#     shares = 0
#     weekly_pct_profit_loss = []
#     current_week = None
#     stop_loss = 0.001  # Set stop loss threshold at 1%
#     comeback = True
#     print(stock_data.iloc[1])
#     ma_window = 7
#     # Iterate over each trading day
#     for i in range(1, len(stock_data)):
#         # Get the buy/sell signal from the strategy function
#         if comeback:
#             signal = (
#                 check_macd(stock_data.iloc[:i])
#                 and check_moving_average(stock_data.iloc[:i])
#                 and check_bullish_pattern(stock_data.iloc[:i])
#                 and check_momentum(stock_data.iloc[:i])
#                 and check_ema(stock_data.iloc[:i])
#                 and check_vwap(stock_data.iloc[:i])
#                 and check_zigzag(stock_data.iloc[:i])
#                 and check_adline(stock_data.iloc[:i])
#                 and check_bollinger_bands(stock_data.iloc[:i])
#                 and check_fundamental_indicators(symbol)
#             )
        
#             # Check if the signal is a buy signal
#             if signal:
#                 if position != True:  # Avoid buying again if already in a position
#                     position = True
#                     buy_price = stock_data['Close'].iloc[i]
#                     shares = capital / buy_price
#                     capital = 0

#             # Check if the signal is a sell signal
#             elif not signal or (position == True and (stock_data['Close'].iloc[i] / buy_price - 1) <= -stop_loss):
#                 if position == True:  # Only sell if currently in a long position
#                     position = False
#                     capital = shares * stock_data['Close'].iloc[i]
#                     shares = 0

#                     if current_week != stock_data.index[i].week:
#                         # Calculate weekly profit/loss percentage
#                         weekly_pct_change = (capital - 10000) / 10000 * 100
#                         weekly_pct_profit_loss.append((stock_data.index[i], weekly_pct_change))
#                         current_week = stock_data.index[i].week
#                 if (stock_data['Close'].iloc[i] / buy_price - 1) <= -stop_loss:
#                     comeback = False
#         else:
#             if(check_bullish_pattern2(stock_data.iloc[:i]) and check_moving_average2(stock_data.iloc[:i]) and check_bollinger_bands2(stock_data.iloc[:i])):
#                 comeback = True

#     print_weekly_profit_loss(weekly_pct_profit_loss)
#     # Calculate final portfolio value
#     final_value = capital + (shares * stock_data['Close'].iloc[-1])

#     return final_value

def simulate_trades(symbol, start_date, end_date, window=10):
    # Download historical data for the stock
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Initialize variables
    position = None
    buy_price = 0
    capital = 10000  # Initial capital
    shares = 0
    weekly_pct_profit_loss = []
    current_week = None
    stop_loss = 0.01  # Set stop loss threshold at 1%
    
    # Iterate over each trading day
    for i in range(window, len(stock_data)):
        historical_data = stock_data.iloc[i-window+1:i+1]  # Window of historical data
        
        # Get the buy/sell signal from the strategy function
        signal = (
            check_macd(historical_data)
            and check_moving_average(historical_data)
            and check_bullish_pattern(historical_data)
            and check_momentum(historical_data)
            and check_ema(historical_data)
            and check_vwap(historical_data)
            and check_zigzag(historical_data)
            and check_adline(historical_data)
            and check_bollinger_bands(historical_data)
            and check_fundamental_indicators(historical_data)
        )
        
        # Check if the signal is a buy signal
        if signal:
            if position is not True:  # Avoid buying again if already in a position
                position = True
                buy_price = stock_data['Close'].iloc[i]
                shares = capital / buy_price
                capital = 0

        # Check if the signal is a sell signal or stop loss triggered
        elif not signal or (position and stock_data['Close'].iloc[i] <= buy_price * (1 - stop_loss)):
            if position:  # Only sell if currently in a long position
                position = False
                capital = shares * stock_data['Close'].iloc[i]
                shares = 0

                if current_week != stock_data.index[i].week:
                    # Calculate weekly profit/loss percentage
                    weekly_pct_change = (capital - 10000) / 10000 * 100
                    weekly_pct_profit_loss.append((stock_data.index[i], weekly_pct_change))
                    current_week = stock_data.index[i].week
    
    # Calculate final portfolio value
    final_value = capital + (shares * stock_data['Close'].iloc[-1])
    percentage_profit = (final_value - 10000) / 10000 * 100
    
    return final_value


def print_weekly_profit_loss(weekly_pct_profit_loss):
    for week, pct_change in weekly_pct_profit_loss:
        print(f"Week {week}: {pct_change:.2f}%")

def get_all_trades():
    symbol = ['AAPL','ONTO','GOOG']
    start_date = '2023-01-01'
    end_date = '2023-06-01'
    profits = []
    for stock in symbol:
        print(stock)
        profits.append(simulate_trades(stock, start_date, end_date))
    print(profits)
# Example usage with MACD strategy
symbol = 'AAPL'
start_date = '2022-06-01'
end_date = '2023-06-01'
get_all_trades()
# macd_final_value = simulate_trades(symbol, start_date, end_date)
# print(f"Final Portfolio Value: {macd_final_value:.2f}")


# Example usage with Moving Average strategy
# moving_average_final_value = simulate_trades(symbol, start_date, end_date, check_moving_average)
# print(f"Moving Average Final Portfolio Value: {moving_average_final_value:.2f}")

[10655.038418591781, 10791.387337917511, 10052.90578609267]
[10456.824274662853, 11522.813065712877, 10438.881365142293]