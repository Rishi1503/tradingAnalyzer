import backtrader as bt
import yfinance as yf
import pandas as pd

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.ma7 = bt.indicators.SimpleMovingAverage(self.dataclose, period=7)
        self.ma3 = bt.indicators.SimpleMovingAverage(self.dataclose, period=3)
        self.adx = bt.indicators.AverageDirectionalMovementIndex()
        self.rsi = bt.indicators.RSI(self.dataclose, period=11)
        self.vwap = 0
        self.volume_sum = 0
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.dataclose, period=3)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.dataclose, period=7)
        self.atr = bt.indicators.ATR(self.data, period=7)
        self.macd = bt.indicators.MACD(self.dataclose, period_me1=12, period_me2=26, period_signal=9)

    def next(self):
        #calculate vwap
        typical_price = (self.data.high + self.data.low + self.data.close) / 3
        self.volume_sum += self.data.volume[0]
        self.vwap = (self.vwap * (self.volume_sum - self.data.volume[0]) + typical_price * self.data.volume[0]) / self.volume_sum

        if (self.dataclose[0] > self.ma7[0] and self.dataclose[0] > self.ma3[0] and self.dataclose[0] > self.vwap 
            and self.ema_short[0] > self.ema_long[0] and self.dataclose[0] > self.atr[0] and self.macd.macd[0] > self.macd.signal[0]):
            self.buy()

        elif (self.dataclose[0] < self.ma7[0] and self.dataclose[0] < self.ma3[0] and self.dataclose[0] < self.vwap 
              and self.ema_short[0] < self.ema_long[0] and self.dataclose[0] < self.atr[0] and self.macd.macd[0] < self.macd.signal[0]):
            self.sell()

# Define a list of symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Replace with the desired stock symbols

# Instantiate the Cerebro engine
cerebro = bt.Cerebro()

# Iterate over the symbols
for symbol in symbols:
    # Download historical data using yfinance
    data = yf.download(symbol, start='2022-01-01', end='2023-01-01')  # Replace with the desired start and end dates

    # Convert the DataFrame to a backtrader-compatible data feed
    data_feed = bt.feeds.PandasData(dataname=data)

    # Add the data feed to the engine
    cerebro.adddata(data_feed)

# Add your strategy to the engine
cerebro.addstrategy(MyStrategy)

# Set the initial capital and position size
cerebro.broker.setcash(10000)
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

# Run the backtest
cerebro.run()

# Print the final portfolio value
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())