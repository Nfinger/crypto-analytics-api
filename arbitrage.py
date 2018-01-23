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

        market_cap = Pymarketcap()

        connection = TinyMongoClient()
        db = connection.cryptoAnalytics
        # data = db.arbitrage.find()
        # arbitrage_data = db.arbitrage.find()
        # arbitrage_id = arbitrage_data[0]['_id']

        def scan_for_arbitrage(to_coin):
            print("Running!", to_coin)
            coin_data = db[to_coin].find()[0]
            from_idx = 0
            if coin_data["lastCoin"] != "":
                from_idx = from_coins.index(coin_data["lastCoin"])
            for from_coin in from_coins[from_idx:]:
                db[to_coin].update_one({"_id": coin_data["_id"]}, {"$set": {"lastCoin": from_coin}})
                # coin = marketCap.ticker(from_coin)
                # exchangeCoin = marketCap.ticker(to_coin)
                # if (coin['market_cap_usd'] and coin['market_cap_usd'] >= 15000000):
                lowest_price = 10000000000000000000000
                highest_price = 0
                exchange1 = None
                exchange2 = None
                for exchange in exchanges:
                    if from_coin in exchange_lists[exchange.lower()]:
                        prices = Price.price(from_curr=from_coin,
                                             to_curr=to_coin, e=exchange.lower())
                        if 'Response' not in prices:
                            if prices[to_coin] < lowest_price:
                                lowest_price = prices[to_coin]
                                exchange1 = exchange
                            if prices[to_coin] > highest_price:
                                highest_price = prices[to_coin]
                                exchange2 = exchange
                if (highest_price > 0 and lowest_price < 10000000000000000000000
                        and highest_price > Decimal(.0000001) and lowest_price > Decimal(.0000001)):
                    percent_diff = ((highest_price - lowest_price) / highest_price) * 100
                    if percent_diff > 50:
                        slack.chat.post_message('#signals',
                                                "%s/%s is listed for %f%s on %s and %f%s on %s"
                                                % (from_coin, to_coin, lowest_price, to_coin,
                                                   exchange1, highest_price, to_coin, exchange2))
            db[to_coin].update_one({"_id": coin_data["_id"]}, {"$set": {"lastCoin": ""}})
            return ""

        slack_token = "xoxp-302678850693-302678850805-302556314308-5b70830e08bc3a0f6895d1f8545f537a"
        slack = Slacker(slack_token)

        exchanges = ["Poloniex", "Kraken",
                     "HitBTC", "Gemini", "Exmo", #"Yobit",
                     "Cryptopia", "Binance", "OKEX"]

        ## Add exchange info to TinyDb
        # for exchange in exchanges:
        #     exchangeInfo = marketCap.exchange(exchange)
        #     db.exchanges.insert_one({"name":exchange.lower(),
        #        "symbols": [coin["market"].split("-")[0] for coin in exchangeInfo]})

        exchange_lists = {}
        ex_data = db.exchanges.find()
        for exchange in ex_data:
            exchange_lists[exchange["name"]] = exchange["symbols"]

        to_coins = ["BTC", "ETH", "LTC"]

        from_coins = market_cap.symbols

        Price = cryCompare.Price()
        for coin in to_coins:
            p = multiprocessing.Process(target=scan_for_arbitrage, args=(coin,))
            p.start()
