import yfinance as yf
import numpy as np
import pandas as pd
import yaml
from datapackage import Package
from support_resistance import check_1_3_6_months
import time
import schedule
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import datetime as dt
# import datetime as dt
# from numpy import arange
# from pandas import read_csv
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import RandomizedSearchCV


with open('app.yaml') as file:
    env_list = yaml.load(file, Loader=yaml.FullLoader)

APCA_API_BASE_URL = env_list['env_variables']['APCA_API_BASE_URL']
trading_client = TradingClient(env_list['env_variables']['API-KEY'], env_list['env_variables']['SECRET-KEY'], paper=True)
account = trading_client.get_account()

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
    return stock_screener(stock_symbols)

def stock_screener(stock_symbols):
    filtered_stocks = []

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

    return filtered_stocks


def check_moving_average(stock_data, long_period=7, short_period=3):
    
    # Calculate the moving average
    stock_data['MA_7'] = stock_data['Close'].rolling(window=long_period).mean()
    stock_data['MA_3'] = stock_data['Close'].rolling(window=short_period).mean()
    
    # Get the last closing price
    current_price = stock_data['Close'][-1]
    
    # Get the last moving average value
    last_ma_7 = stock_data['MA_7'][-1]
    last_ma_3 = stock_data['MA_3'][-1]
    
    # Determine if the stock should be bought or sold
    if current_price > last_ma_7 and current_price > last_ma_3:
        return True
    else:
        return False

def check_bullish_pattern(stock_data, short_period=3, long_period=7):
    
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

def check_momentum(stock_data, period=7):
    
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

def check_on_balance_volume(stock_data):
    
    # Calculate the On-Balance Volume (OBV)
    stock_data['OBV'] = (stock_data['Close'] - stock_data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1).cumsum()
    # copy["OBV"] = (np.sign(copy["Close"].diff()) * copy["Volume"]).fillna(0).cumsum()
    # Get the most recent OBV value
    current_obv = stock_data['OBV'][-1]
    
    # Get the most recent closing price
    current_close = stock_data['Close'][-1]
    
    # Check if the OBV and price are in agreement
    if current_obv > 0 and current_close > stock_data['Close'].mean():
        return True
    else:
        return False

#Accumulation/Distribution line 
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

def is_stock_worth_buying_with_aroon(symbol, period=4):
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
    if aroon_up > 60 and aroon_down < 40:
        return f"{symbol}: Buy the stock (Aroon-Up is above 80 and Aroon-Down is below 20)"
    else:
        return f"{symbol}: Do not buy the stock (Aroon indicator does not suggest buying opportunity)"

#don't use yet
def check_rsi(stock_data, period=14, oversold_threshold=40, overbought_threshold=60):

    # change = stock_data["Close"].diff()
    
    # # Create two copies of the Closing price Series
    # change_up = change.copy()
    # change_down = change.copy()

    # # 
    # change_up[change_up<0] = 0
    # change_down[change_down>0] = 0

    # # Verify that we did not make any mistakes
    # change.equals(change_up+change_down)

    # # Calculate the rolling average of average up and average down
    # avg_up = change_up.rolling(14).mean()
    # avg_down = change_down.rolling(14).mean().abs()

    # rsi = 100 * avg_up / (avg_up + avg_down)

    # Take a look at the 20 oldest datapoints
    
    # Get historical data for the stock
    # stock_data = yf.download(symbol)
    
    # Calculate price change and store as a new column
    stock_data['Price Change'] = stock_data['Close'].diff()
    
    # Calculate the gain and loss for each period
    stock_data['Gain'] = stock_data['Price Change'].apply(lambda x: x if x > 0 else 0)
    stock_data['Loss'] = stock_data['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate the average gain and average loss
    stock_data['Avg Gain'] = stock_data['Gain'].rolling(window=period).mean()
    stock_data['Avg Loss'] = stock_data['Loss'].rolling(window=period).mean()
    
    # Calculate the relative strength (RS)
    stock_data['RS'] = stock_data['Avg Gain'] / stock_data['Avg Loss']
    
    # Calculate the RSI using the RS
    stock_data['RSI'] = 100 - (100 / (1 + stock_data['RS']))
    
    # Get the most recent RSI value
    current_rsi = stock_data['RSI'][-1]
    print(current_rsi)
    # Check if the RSI value suggests buying the stock
    if current_rsi > overbought_threshold:
        return False
    elif current_rsi < oversold_threshold:
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
        return False

def check_ema(stock_data, short_period=3, long_period=7):
    
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

def check_atr(stock_data, atr_period=7, atr_multiplier=2.0):
    # Calculate the Average True Range (ATR)
    
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

def check_zigzag(stock_data, deviation_percentage=5.0):
    zigzag_points = calculate_zigzag(stock_data, deviation_percentage)

    # Get the most recent ZigZag point
    current_zigzag_point = zigzag_points[-1]

    # Check if the current index is the most recent ZigZag high or low
    if current_zigzag_point == len(stock_data) - 1:
        return True
    else:
        return False

def check_macd(stock_data):

    # Calculate MACD using exponential moving averages
    stock_data['EMA12'] = stock_data['Close'].ewm(span=12).mean()
    stock_data['EMA26'] = stock_data['Close'].ewm(span=26).mean()
    stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()

    # Get the most recent MACD and Signal Line values
    current_macd = stock_data['MACD'][-1]
    current_signal_line = stock_data['Signal Line'][-1]

    # Check if MACD crosses above the Signal Line (buy signal)
    if current_macd > current_signal_line:
        return True
    else:
        return False

def check_all_stocks(stocks_to_check):
    #watchlist = [ESTA,]
    #stocks_to_check = ['AAPL']
    #stocks_to_check = ['GS','WH','JNPR','WDC','FI','UFPI','BLCO','AMN','MA','CCK','TXT','HOMB','CI','RWEOY','PRCT','HUBS','TOL','ARES','CCJ','MTZ','DQ','AXTA','INMD','VNOM','ITGR','COHU','KMTUY','KWR','HTHIY','FTV','V','ACGL','GOOG','DBX','AMT','MAURY','KBR','IAC','JLL','CVS','AVY','ETN','FUTU','SLB','SPSC','CEQP','JHX','BKR','WTFC','JCI','SPB','SPOT','DCBO','DINO','TTWO','UBS','LOW','TJX','BR','BSY','ABM','NVDA','SAFE','LEN','IMCR','BRBR','ZI','DOCS','BYD','CPRI','PDCE','CHRD','ROP','SHW','IONS','MIELY','APD','MOH','SVNDY','INSW','KNTK','DOV','WDS','BKNG','WDS','THRM','STAG','QCOM','GLOB','APO','SSB','PEAK','ADI','LIVN','PBA','HES','PNC','HMC','AKZOY','CORT','EDR','RNG','KEX','VTR','COLD','PCTY','PD','DRI','AZEK','INSM','IRM','SMPL','EQT','TEAM','GWRE','CIGI','ULTA','VRDN','EFX','KNX','STE','CNQ','BOX','PWR','MLNK','ANET','PB','MKL','EOG','MS','RRC','CAR','EE','DECK','BERY','BYDDY','BIO','NVEE','CIEN','NVT','DUOL','VMC','BRX','DUI','FSLR','MIDD','EMR','CMG','DVN','SAR','GPI','THC','ASH','HON','SPT','SYIEY','CCRN','TENB','AVTR','POOL','SPGI','CCI','ICE','DEO','GXO','CDRE','AL','FDX','KB','TNL','COST','PCOR','IR','COP','BANR','OFC','COOP','PEN','FITB','DSGX','DNLI','BAM','FSS','ESAB','OTEX','PRVA','NOG','CFIGY','SGRY','NOW','EQIX','EL','WFRD','BPOP','ONEW','BP','MLM','RY','EHC','OEC','PLD','GNTX','ABC','ADRNY','DEN','NVO','MODG','BBWI','MKSI','PXD','HEINY','DRVN','IQV','CW','CBT','BCH','TD','LECO','JBHT','GIB','MSI','NCR','SNPS','CPRT','SNDR','LPLA','HESM','CADE','PRLB','DDOG','RRX','ENTA','CWST','TECK','RYAN','WAL','BAP','TMUS','AAPL','PFSI','AIAGY','EQH','EGLE','CRWD','PBH','TS']
    stocks_to_buy = []
    count = 0
    for stock in stocks_to_check:
        try:
            stock_data = yf.download(stock)
            count = 0

            if check_moving_average(stock_data):
                count += 1
            if check_bullish_pattern(stock_data):
                count += 1
            if check_momentum(stock_data):
                count += 1
            if check_on_balance_volume(stock_data):
                count += 1
            if check_adline(stock_data):
                count += 1
        # if check_rsi(stock_data):
        #     count += 1
            if check_vwap(stock_data):
                count += 1
            if check_ema(stock_data):
                count += 1
            if check_atr(stock_data):
                count += 1
            if check_zigzag(stock_data):
                count += 1
            if check_macd(stock_data):
                count += 1

            if count >= 9:
                stocks_to_buy.append(stock)

        except Exception as e:
            print(f"Error occurred while processing stock '{stock}': {e}")
    stocks_waitlist = []
    positions = trading_client.get_all_positions()
    print('Stage 1 Filtering Done')
    stocks_to_buy, stocks_waitlist = check_1_3_6_months(stocks_to_buy)
    print("Waitlisted stocks: ", stocks_waitlist)
    for position in positions:
        if stocks_to_buy.__contains__(position.symbol):
            stocks_to_buy.remove(position.symbol)
    print('Stage 2 Filtering Done')
    print("Filtered stocks to buy: ", stocks_to_buy)            
    for stock in stocks_to_buy:
        try:
            if trading_client.get_asset(stock).tradable:
                buystock(stock)
        except Exception as e:
            print(f"Error occurred while purchasing stock '{stock}': {e}")
    return stocks_to_buy


def sellstock(stock, quantity):
    market_order_data = MarketOrderRequest(
                    symbol=stock,
                    type="trailing_stop",
                    qty=quantity,
                    side=OrderSide.SELL,
                    trail_percent=1,
                    time_in_force=TimeInForce.DAY
                    )
    # Market order
    market_order = trading_client.submit_order(
                order_data=market_order_data
               )
def sellstock_instant(stock, quantity):
    market_order_data = MarketOrderRequest(
                    symbol=stock,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                    )
    # Market order
    market_order = trading_client.submit_order(
                order_data=market_order_data
               )
    
def buystock(stock):
    market_order_data = MarketOrderRequest(
                    symbol=stock,
                    notional=1000,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )
    # Market order
    market_order = trading_client.submit_order(
                order_data=market_order_data
               )
    print(trading_client.get_orders)

def set_sell_orders(tif):
    #print(trading_client.get_all_positions())
    positions = trading_client.get_all_positions()
    for position in positions:
        print(position.symbol + " " + position.market_value)
        sellstock(position.symbol, position.qty)

def check_positions():
    #print(trading_client.get_all_positions())
    positions = trading_client.get_all_positions()
    print("========================")
    for position in positions:
        print(position.symbol + " " + position.market_value)
        if ((float(position.market_value)/float(position.cost_basis)) > 1.01):
            print("profitted")
            sellstock_instant(position.symbol,position.qty)
        if ((float(position.market_value)/float(position.cost_basis)) < 0.99):
            print("loss")
            sellstock(position.symbol, position.qty)
    print("========================")

stock_symbols = get_stock_listings()

# Schedule the function to run every hour
schedule.every(2).hours.do(check_all_stocks,stock_symbols)
schedule.every(2).minutes.do(check_positions)
check_all_stocks(stock_symbols)
# Run the scheduler continuously
while True:
    schedule.run_pending()
    # time.sleep(1)
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(current_time)
