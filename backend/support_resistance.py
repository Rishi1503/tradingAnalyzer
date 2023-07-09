# Importing necessary python libraries for this project
import pandas as pd
import numpy as np
import yfinance as yf
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

# Create two functions to calculate if a level is SUPPORT or a RESISTANCE level through fractal identification
def is_Suppport_Level(df, i):
  support = df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1] and df['Low'][i + 1] < df['Low'][i + 2] and df['Low'][i - 1] < df['Low'][i - 2]
  return support


def is_Resistance_Level(df, i):
  resistance = df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1] and df['High'][i + 1] > df['High'][i + 2] and df['High'][i - 1] > df['High'][i - 2]
  return resistance

def plot_levels(df, ticker_symbol, levels, level_types):
  fig, ax = plt.subplots()
  candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
  date_format = mpl_dates.DateFormatter('%d %b %Y')
  ax.xaxis.set_major_formatter(date_format)
  fig.autofmt_xdate()
  fig.tight_layout()

  for level, level_type in zip(levels, level_types):
    plt.hlines(level[1],
               xmin = df['Date'][level[0]],
               xmax = max(df['Date']),
               colors = 'blue')
    plt.text(df['Date'][level[0]], level[1], (str(level_type) + ': ' + str(level[1]) + ' '), ha='right', va='center', fontweight='bold', fontsize='x-small')
    plt.title('Support and Resistance levels for ' + ticker_symbol, fontsize=24, fontweight='bold')
    fig.show()

# plot_levels()

# This function, given a price value, returns True or False depending on if it is too near to some previously discovered key level.
def distance_from_mean(level, mean, levels):
  return np.sum([abs(level - y) < mean for y in levels]) == 0
      
# plot_levels()

def get_support_resistance_levels(ticker_symbol, start_date, end_date):
  # Obtaining historical stock pricing data
    ticker = yf.Ticker(ticker_symbol)

    df = ticker.history(interval='1d', start=start_date, end=end_date)

    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)

    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

    # Clean noise in data by discarding a level if it is near another
    # (i.e. if distance to the next level is less than the average candle size for any given day - this will give a rough estimate on volatility)
    mean = np.mean(df['High'] - df['Low'])

    # Optimizing the analysis by adjusting the data and eliminating the noise from volatility that is causing multiple levels to show/overlapp
    levels = []
    level_types = []
    for i in range(2, df.shape[0] - 2):

        if is_Suppport_Level(df, i):
            level = df['Low'][i].round(2)

            if distance_from_mean(level, mean, levels):
                levels.append((i, level))
                level_types.append('Support')

        elif is_Resistance_Level(df, i):
            level = df['High'][i].round(2)

            if distance_from_mean(level, mean, levels):
                levels.append((i, level))
                level_types.append('Resistance')

    # print(levels)
    # plot_levels(df, ticker_symbol, levels, level_types)
    # plt.show()

    resistance_levels = []
    support_levels = []

    for i in range (len(levels)):
        if level_types[i] == 'Resistance':
            resistance_levels.append(levels[i])
        else:
            support_levels.append(levels[i])

    # print("Resistance: ", resistance_levels)
    # print("Support: ", support_levels)
    return compare_price_levels(resistance_levels, support_levels, df['Close'][-1], mean)

def compare_price_levels(resistance_levels, support_levels, current_price, mean):
    should_buy = ''
    for level in resistance_levels:
        if abs(current_price - level[1]) < mean * 0.02:
            # print("Current price is near a resistance level:", level[1])
            should_buy = 'No'
            return should_buy

    for level in support_levels:
       if abs(current_price - level[1]) < mean:
            # print("Current price is near a support level:", level[1])
            should_buy = 'Yes'
            return should_buy
    return 'Maybe'

def check_1_3_6_months(ticker_symbols):
    stocks_waitlist = []
    stocks_to_buy = []
    for symbol in ticker_symbols:
        one_month = get_support_resistance_levels(symbol, '2023-06-01', '2023-07-10')
        three_month = get_support_resistance_levels(symbol, '2023-04-01', '2023-07-10')
        six_month = get_support_resistance_levels(symbol, '2023-01-01', '2023-07-10')
        # print(one_month, ' ', three_month, ' ', six_month)
        if one_month == 'Yes':
            if three_month == 'Yes' or six_month == 'Yes':
                stocks_to_buy.append(symbol)
            else:
                stocks_waitlist.append(symbol)
        elif one_month == 'Maybe' and (three_month == 'Yes' and six_month == 'Yes'):
            stocks_to_buy.append(symbol)
        elif one_month == 'Maybe' and three_month == 'Maybe' and six_month == 'Maybe':
            stocks_to_buy.append(symbol)
        else:
            stocks_waitlist.append(symbol)
    return stocks_to_buy, stocks_waitlist
    
    # print('1 month: ')
    # get_support_resistance_levels(ticker_symbol, '2023-06-01', '2023-07-10')
    # print('3 month: ')
    # get_support_resistance_levels(ticker_symbol, '2023-04-01', '2023-07-10')
    # print('6 month: ')
    # get_support_resistance_levels(ticker_symbol, '2023-01-01', '2023-07-10')


# check_1_3_6_months(ticker_symbol)
# ticker_symbol = 'PWR'

#prioritize 1 month, then if either 3 or 6 month say yes then buy