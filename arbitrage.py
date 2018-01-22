import gdax
import twilio
from slackclient import SlackClient
from slacker import Slacker
import asyncio
import cryCompare
from decimal import * 
from pymarketcap import Pymarketcap
from binance.client import Client
from kucoin.client import Client
getcontext().prec = 15
api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

kucoin_api_key = '5a64f6a46829d247d237e7bf'
kucoin_api_secret = '93b85f5c-f164-4bea-bd40-3ffda4c03907'

client = Client(api_key, api_secret)
kuClient = Client(kucoin_api_key, kucoin_api_secret)
client = Client(api_key, api_secret)
public_client = gdax.PublicClient()
marketCap = Pymarketcap()

# from twilio.rest import Client
# account_sid = "AC41f2c538b7d6075d94af6f3b4b8b32cd"
# auth_token = "7a9a687d4207e5aa6386a3a5efbcb8f2"

# toNumbers = ["7139624036", "7742547054"]
# twilioClient = Client(account_sid, auth_token)

slack_token = "xoxp-302678850693-302678850805-302556314308-5b70830e08bc3a0f6895d1f8545f537a"
slack = Slacker(slack_token)

# Get the product ticker for a specific product.
tickerInfo = public_client.get_product_ticker(product_id='ETH-USD')

exchanges = ["Coinbase", "Poloniex", "Kraken", "Bitfinex",
            "HitBTC", "Gemini", "Exmo", "Yobit", 
            "Cryptopia", "Binance", "Gateio", "OKEX"]
toCoins = ["BTC", "ETH", "LTC"]

Price = cryCompare.Price()
for toCoin in toCoins:
    for fromCoin in marketCap.symbols:
        coin = marketCap.ticker(fromCoin)
        exchangeCoin = marketCap.ticker(toCoin)
        if (coin['market_cap_usd'] >= 15000000):
            lowestPrice = 10000000000000000000000
            highestPrice = 0
            exchange1 = None
            exchange2 = None
            for exchange in exchanges:
                prices = Price.price(from_curr=fromCoin, to_curr=toCoin, e=exchange.lower())
                if 'Response' not in prices:
                    if prices[toCoin] < lowestPrice:
                        lowestPrice = prices[toCoin]
                        exchange1 = exchange
                    if prices[toCoin] > highestPrice:
                        highestPrice = prices[toCoin]
                        exchange2 = exchange
            if highestPrice > 0 and lowestPrice < 10000000000000000000000:
                percentDiff = ((highestPrice - lowestPrice) / highestPrice) * 100
                if percentDiff > 50:
                    slack.chat.post_message('#signals', "Arbitrage Opportunity: %s/%s is listed for %f%s on %s and %f%s on %s" % (coin['symbol'], toCoin, lowestPrice, toCoin, exchange1, highestPrice, toCoin, exchange2))
                    # twilioClient.api.account.messages.create(to=number,from_="8566725566",body="Arbitrage Opportunity: %s/%s is listed for %f%s on %s and %f%s on %s" % (coin['symbol'], toCoin, lowestPrice, toCoin, exchange1, highestPrice, toCoin, exchange2))