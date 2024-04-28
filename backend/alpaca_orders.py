import time
import schedule
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import datetime as dt
import yaml

with open('app.yaml') as file:
    env_list = yaml.load(file, Loader=yaml.FullLoader)

APCA_API_BASE_URL = env_list['env_variables']['APCA_API_BASE_URL']
trading_client = TradingClient(env_list['env_variables']['API-KEY'], env_list['env_variables']['SECRET-KEY'], paper=True)
account = trading_client.get_account()

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
                    notional=500,
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
    output_file_positive = 'trade_history_positive.csv'
    output_file_negative = 'trade_history_negative.csv'
    positions = trading_client.get_all_positions()
    print("========================")
    for position in positions:
        print(position.symbol + " " + position.market_value)
        if ((float(position.market_value)/float(position.cost_basis)) > 1.01):
            print("profitted")
            profit_loss = ((float(position.market_value)/float(position.cost_basis)) - 1) * 100
            sellstock_instant(position.symbol,position.qty)
            with open(output_file_positive, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([1])
                file.close()
        if ((float(position.market_value)/float(position.cost_basis)) < 0.99):
            print("loss")
            profit_loss = ((float(position.market_value)/float(position.cost_basis)) - 1) * 100
            sellstock_instant(position.symbol, position.qty)
            with open(output_file_negative, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([1])
                file.close()
    print("========================")

schedule.every(2).minutes.do(check_positions)