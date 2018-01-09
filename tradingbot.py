import gdax, time
import pandas as pd
import numpy as np

import gemini

class TradingBot:
    def __init__(self):
        self.USD = 0
        self.holdings = 0
        self.trades = []
        self.ticks = 0  # 28
        self.position = 0  # 29
        self.df = pd.DataFrame()  # 30
        self.momentum = 12  # 31
        self.units = 100000  # 32

    def buy(self):
        print("WERE BUYING")

    def sell(self):
        print("WERE SELLING")

# class WebsocketClient(gdax.WebsocketClient):
#     def on_open(self):
#         self.url = "wss://ws-feed.gdax.com/"
#         self.products = ["ETH-USD"]

#     def on_message(self, msg):
#         if msg["type"] == 'done' and msg["reason"] == 'filled' and "price" in msg:
#             pass

#     def on_close(self):
#         print("-- Goodbye! --")

# tradingbot = TradingBot()
# wsClient = WebsocketClient()
# wsClient.start()
# print(wsClient.url, wsClient.products)
# while (True):
#     time.sleep(1)
# wsClient.close()

pair = "ETH-USD"    # Use ETH pricing data on the BTC market

# Request data from gdax
import gdax
public_client = gdax.PublicClient()
data = public_client.get_product_historic_rates(pair, granularity=60)

# Convert to Pandas dataframe with datetime format
data = pd.DataFrame(data, columns=['date', 'low', 'high', 'open', 'close', 'volume'])
# data['date'] = pd.to_datetime(data['date'], unit='s')

# Load the data into a backtesting class called Run
r = gemini.Run(data)

import helpers

def Logic(Account, Lookback):
    px = pd.DataFrame([], columns=["26 ema", "12 ema", "MACD", "Signal Line"])
    px['26 ema'] = Lookback["close"].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
    px['12 ema'] = Lookback["close"].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
    px['MACD'] = (px['12 ema'] - px['26 ema'])
    px['Signal Line'] = Lookback["close"].ewm(span=9,min_periods=0,adjust=True,ignore_na=False).mean()
    print(px)


# Start backtesting custom logic with 1000 (BTC) intital capital
r.Start(1000, Logic)

r.Results()
