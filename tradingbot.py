import gdax, time

class TradingBot:
    def __init__(self):
        self.USD = 0
        self.holdings = 0
        self.trades = []


    def analyze(self, data):
        # make a decision based off the data we get
        # make a billion dollars
        pass
    def buy(self):
        pass
    
    def sell(self):
        pass

class WebsocketClient(gdax.WebsocketClient):
    def on_open(self):
        self.url = "wss://ws-feed.gdax.com/"
        self.products = ["ETH-USD"]
        self.message_count = 0
        print("Lets count the messages!")
        
    def on_message(self, msg):
        if msg["type"] == 'done' and msg["reason"] == 'filled' and "price" in msg:
            print(msg["price"])

            analytics = tradingbot.analyze(msg)
            # if (analytics.buy):
            #     tradingbot.buy()
            # if (analytics.sell):
            #     tradingbot.sell()

    def on_close(self):
        print("-- Goodbye! --")


tradingbot = TradingBot()
wsClient = WebsocketClient()
wsClient.start()
print(wsClient.url, wsClient.products)
while (True):
    time.sleep(1)
wsClient.close()