import multiprocessing
from decimal import Decimal
from slacker import Slacker
from pymarketcap import Pymarketcap
from tinymongo import TinyMongoClient
import cryCompare

class ArbitrageBot:

    def __init__(self):
        # getcontext().prec = 15
        # api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
        # api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

        # kucoin_api_key = '5a64f6a46829d247d237e7bf'
        # kucoin_api_secret = '93b85f5c-f164-4bea-bd40-3ffda4c03907'

        self.market_cap = Pymarketcap()

        # connection = TinyMongoClient()
        # db = connection.cryptoAnalytics
        # data = db.arbitrage.find()
        # arbitrage_data = db.arbitrage.find()
        # arbitrage_id = arbitrage_data[0]['_id']
        slack_token = "xoxp-302678850693-302678850805-302556314308-5b70830e08bc3a0f6895d1f8545f537a"
        self.slack = Slacker(slack_token)

        self.exchanges = ["Poloniex", "Kraken",
                        "HitBTC", "Gemini", "Exmo", #"Yobit",
                        "Cryptopia", "Binance", "OKEX"]

        self.to_coins = ["BTC", "ETH", "LTC"]
        self.Price = cryCompare.Price()
        self.lowest_price = 10000000000000000000000
        self.highest_price = 0
        self.exchange1 = None
        self.exchange2 = None
        self.movement = None
        # from_coins = market_cap.symbols

    def scan_for_arbitrage(self, to_coin, targetCoin):
        print("Running!", to_coin)
        # coin = marketCap.ticker(from_coin)
        # exchangeCoin = marketCap.ticker(to_coin)
        # if (coin['market_cap_usd'] and coin['market_cap_usd'] >= 15000000):
        
        for exchange in self.exchanges:
            # if from_coin in exchange_lists[exchange.lower()]:
            prices = self.Price.price(from_curr=targetCoin,
                                    to_curr=to_coin, e=exchange.lower())
            if 'Response' not in prices:
                if prices[to_coin] < self.lowest_price and self.movement == "up":
                    self.lowest_price = prices[to_coin]
                    self.exchange1 = exchange
                if prices[to_coin] > self.highest_price and self.movement == "down":
                    self.highest_price = prices[to_coin]
                    self.exchange2 = exchange
        if (self.highest_price > 0 and self.lowest_price < 10000000000000000000000
                and self.highest_price > Decimal(.0000001) and self.lowest_price > Decimal(.0000001)):
            percent_diff = ((self.highest_price - self.lowest_price) / self.highest_price) * 100
            if percent_diff > 50:
                self.slack.chat.post_message('#signals',
                                        "%s/%s is listed for %f%s on %s and %f%s on %s"
                                        % (targetCoin, to_coin, self.lowest_price, to_coin,
                                            self.exchange1, self.highest_price, to_coin, self.exchange2))

    def checkCoin(self, targetCoin_json):
        if (targetCoin_json["type"] == "up"):
            self.movement = "up"
            self.highest_price = targetCoin_json["price_btc"]
            self.exchange2 = targetCoin_json["exchange"]
        else:
            self.movement = "down"
            self.lowest_price = targetCoin_json["price_btc"]
            self.exchange1 = targetCoin_json["exchange"]
        for coin in self.to_coins:
            p = multiprocessing.Process(target=self.scan_for_arbitrage, args=(coin,targetCoin_json["ticker"]))
            p.start()
