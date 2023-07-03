import pandas as pd
import yfinance as yf
from strategies import *

# Step 1: Data Collection
# Assuming you have a CSV file containing historical stock data
data = yf.download('GOOG', start='2023-04-03', end='2023-07-03')

#Checks moving average
data['MA'] = data['Close'].rolling(window=7).mean()

#checks for bullish pattern
data['short_MA'] = data['Close'].rolling(window=3).mean()
data['long_MA'] = data['Close'].rolling(window=7).mean()

#check momentum
data['ROC'] = data['Close'].pct_change(7)

#checks OBV (needs work)
data['OBV'] = (data['Close'].diff() > 0).astype(int)
data['OBV'] = (data['OBV'] * 2 - 1) * data['Volume']
data['OBV'] = data['OBV'].cumsum()

#Checks AD line
data['A/D Line'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
data['A/D Line'] *= data['Volume']
data['A/D Line'] = data['A/D Line'].cumsum()

# Calculate MACD using exponential moving averages
data['EMA12'] = data['Close'].ewm(span=12).mean()
data['EMA26'] = data['Close'].ewm(span=26).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9).mean()

#Check EMA
data['EMA_short'] = data['Close'].ewm(span=3).mean()
data['EMA_long'] = data['Close'].ewm(span=7).mean()

data.dropna(inplace=True)

# Step 2: Algorithm Development
# Define your algorithm's buy/sell rules
def buy_signal(data, index):
    count = 0
    current_price = data['Close'][index]
    if current_price > data['MA'][index]:
        count = count+1
    # if data['short_MA'][index] > data['long_MA'][index]:
    #     count = count +1
    if data['ROC'][index] > 0:
        count = count + 1
    if data['OBV'][index] > data['OBV'].shift(1)[index]:
        count = count + 1
    if data['A/D Line'][index] > 0:
        count = count +1
    if data['MACD'][index] > data['Signal Line'][index]:
        count = count +1
    if data['EMA_short'][index] > data['EMA_long'][index]:
        count = count +1
    # Implement your buy criteria here
    if count>=6:
        return True
    return False

def sell_signal(data, index):
    # Implement your sell criteria here
    return True

# Initialize variables
position = None
buy_price = 0
capital = 10000  # Initial capital
shares = 0
weekly_pct_profit_loss = []
current_week = None
stop_loss = 0.01  # Set stop loss threshold at 1%
# Iterate over each trading day
for i in range(1, len(data)):
        
    # Get the buy/sell signal from the strategy function
    signal = buy_signal(data, i)
        
    # Check if the signal is a buy signal
    if signal:
        if position is not True:  # Avoid buying again if already in a position
            position = True
            buy_price = data['Close'].iloc[i]
            print('BUYINGGG: ', buy_price)
            shares_to_buy = capital / buy_price
            shares += shares_to_buy
            capital -= shares_to_buy * buy_price

    # Check if the signal is a sell signal or stop loss triggered
    elif not signal and (position and (data['Close'].iloc[i] <= buy_price * (1 - stop_loss)) or (data['Close'].iloc[i] <= buy_price * (1 + stop_loss))):
        if position:  # Only sell if currently in a long position
            position = False
            print('Sellinggggg: ', data['Close'].iloc[i])
            capital += shares * data['Close'].iloc[i]
            shares = 0

            if current_week != data.index[i].week:
                # Calculate weekly profit/loss percentage
                weekly_pct_change = (capital - 10000) / 10000 * 100
                weekly_pct_profit_loss.append((data.index[i], weekly_pct_change))
                current_week = data.index[i].week
    
# Calculate final portfolio value
final_value = capital + (shares * data['Close'].iloc[-1])
percentage_profit = (final_value - 10000) / 10000 * 100

print(final_value)
print(percentage_profit)