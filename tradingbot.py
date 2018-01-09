import gdax, time
import pandas as pd
import numpy as np

import gemini
import helpers
import gdax

class TradingBot:
    def __init__(self):
        self.USD = 1000
        self.holdings = 0
        self.trades = []
        self.ticks = 0  # 28
        self.position = 0  # 29
        self.df = pd.DataFrame()  # 30
        self.momentum = 12  # 31

    def buy(self, trade):
        self.holdings += self.USD / trade['close'].item()
        self.USD = 0
        self.trades.append(trade)

    def sell(self, trade):
        self.USD += self.holdings * trade['close'].item()
        self.holdings = 0
        self.trades.append(trade)

# class WebsocketClient(gdax.WebsocketClient):
#     def on_open(self):
#         self.url = "wss://ws-feed.gdax.com/"
#         self.products = ["ETH-USD"]

#     def on_message(self, msg):
#         if msg["type"] == 'done' and msg["reason"] == 'filled' and "price" in msg:
#             pass

#     def on_close(self):
#         print("-- Goodbye! --")

tradingbot = TradingBot()
# wsClient = WebsocketClient()
# wsClient.start()
# print(wsClient.url, wsClient.products)
# while (True):
#     time.sleep(1)
# wsClient.close()

pair = "ETH-USD"    # Use ETH pricing data on the BTC market

# Request data from gdax
public_client = gdax.PublicClient()
data = public_client.get_product_historic_rates(pair, granularity=60)

# Convert to Pandas dataframe with datetime format
data = pd.DataFrame(data, columns=['date', 'low', 'high', 'open', 'close', 'volume'])
# data['date'] = pd.to_datetime(data['date'], unit='s')

# Load the data into a backtesting class called Run
r = gemini.Run(data)

def Logic(Account, DataFrame):
    px = pd.DataFrame([], columns=["time", "9ma", "26ma", "std"])
    px['time'] = pd.to_datetime(DataFrame['date'], unit='s')
    px['9ma'] = DataFrame["close"].rolling(9).mean()
    px['12ma'] = DataFrame["close"].rolling(12).mean()
    px['std'] = DataFrame["close"].rolling(9).std()
    zscores = (DataFrame['close'] - px['9ma']) / px['std']
    if px.shape[0] < 12:
        return
    print(px['9ma'] - px['12ma'])
    # Sell short if the z-score is > 1
    if zscores[-1:].item() > 1:
        ExitPrice = DataFrame[-1:]
        tradingbot.sell(ExitPrice)
    # Buy long if the z-score is < 1
    elif zscores[-1:].item() < -1:
        EntryPrice = DataFrame[-1:]
        tradingbot.buy(EntryPrice)


# Start backtesting custom logic with 1000 (BTC) intital capital
print("Were trading", pair)
r.Start(1000, Logic)
print("Done!")
print("USD: ", tradingbot.USD)
print(pair, ": ", tradingbot.holdings)
# r.Results()
