from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from flask_cors import CORS
from tinymongo import TinyMongoClient
from flask_socketio import SocketIO, emit, disconnect
from binance.enums import *
from binance.websockets import BinanceSocketManager
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import tweepy
import requests
import arrow
# from arbitrage import ArbitrageBot

consumer_key = "BPlzYeWngcK8vluAmNLIoiBgH"
consumer_secret = "nPobDCHkLZKjl8Y1BFm4PiJyCSKB1bM7U9cpLhjjGPqdi6unz4"
access_token = "948764981205569536-BsidADMWaOTFG8qIP58EnD1eppQEX3a"
access_token_secret = "8Rl3J0Mhg8PK6m6sASz3i7Y7VgKUBDHPAEgce7rFyPAr1"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

import urllib.request

# from tradingbot import TradingBot

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

kucoin_api_key = '5a64f6a46829d247d237e7bf'
kucoin_api_secret = '93b85f5c-f164-4bea-bd40-3ffda4c03907'

from binance.client import Client
client = Client(api_key, api_secret)
from kucoin.client import Client
kuClient = Client(kucoin_api_key, kucoin_api_secret)

bm = BinanceSocketManager(client)

app = Flask(__name__)
socketio = SocketIO(app)
connection = TinyMongoClient()
db = connection.cryptoAnalytics
collection = db.arbitrage
CORS(app)

@app.route("/binance")
def binanceKlines():
    res = {}
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    info = client.get_symbol_ticker()
    BTCprice = 0
    
    for ticker in info:
        if ticker['symbol'] == "BTCUSDT":
            BTCprice = ticker['price']
    candles = client.get_klines(symbol=symbol, interval=interval)
    res["data"] = candles
    res["BTC"] = BTCprice
    return jsonify(res)

@app.route("/kucoin")
def kucoinKlines():
    res = {}
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    info = client.get_symbol_ticker()
    BTCprice = 0
    a = arrow.now()
    fromtime = a.shift(weeks=-3)
    from_time = fromtime.timestamp
    to_time = a.timestamp
    for ticker in info:
        if ticker['symbol'] == "BTCUSDT":
            BTCprice = ticker['price']
    if interval == "1min":
        interval = Client.RESOLUTION_1MINUTE
    elif interval == "5min":
        interval = Client.RESOLUTION_5MINUTES
    elif interval == "15min":
        interval = Client.RESOLUTION_15MINUTES
    elif interval == "30min":
        interval = Client.RESOLUTION_30MINUTES
    elif interval == "1hour":
        interval = Client.RESOLUTION_1HOUR
    elif interval == "1day":
        interval = Client.RESOLUTION_1DAY
    candles = kuClient.get_kline_data_tv(
        symbol,
        interval,
        from_time,
        to_time
    )
    res["data"] = candles
    res["BTC"] = BTCprice
    return jsonify(res)

@socketio.on('close')
def closeSocket():
    disconnect()

@socketio.on("exchanges")
def getExchangeInfo():
    results = {}
    exchanges = ['binance', 'kucoin', 'cryptopia']
    for exchange in exchanges:
        coins =  []
        html = requests.get("https://coinmarketcap.com/exchanges/{}".format(exchange))
        soup = BeautifulSoup(html.text, 'html.parser')
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > 0:
                cell = cells[2]
                cell = cell.get_text().split("/")[0]
                coins.append(cell)
        results[exchange] = coins
    socketio.emit('exchangeInfo', results)

@socketio.on("change")
def getTickerChanges():
    keys = ["symbol","price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]
    vals = []
    data = requests.get("https://api.coinmarketcap.com/v1/ticker/?limit=0").json()

    for coin in data:
        vals.append([coin[key] for key in keys])
    df = pd.DataFrame(vals, columns=keys)
    df = df[df["24h_volume_usd"].astype('float32') > 1000]
    # df = df.sort_values("percent_change_1h")
    socketio.emit('stats', {"data":df.to_json(orient='records').replace('},{', '} {')})
    
@socketio.on("candles")
def binanceSocket(info):
    def process_message(msg):
        if msg['e'] == 'error':
            pass
            # close and restart the socket
        else:
            with app.test_request_context('/'):
                socketio.emit('klines', {'data': msg})
    conn_key = bm.start_kline_socket(info['symbol'], process_message, interval=info['interval'])
    bm.start()
    return ""

@app.route("/trades")
def getTrades():
    usd = request.args.get('usd')
    holdings = request.args.get('holdings')
    # trades, usd, holdings, predictions = TradingBot.trade(int(usd), int(holdings))
    # return jsonify({'result' : trades, 'usd':usd, 'holdings': holdings, 'predictions':predictions.tolist()})

@app.route("/twitter")
def getTweets():
    symbol = request.args.get('symbol')
    searchParam = "$%s" % symbol.replace("BTC", "").replace("USDT", "").replace("USD", "")
    tweets = api.search(q=searchParam,lang="en")
    results = []
    for tweet in tweets:
        if "RT" not in tweet.text:
            results.append(tweet)
    data = [s._json for s in results]
    return jsonify({"tweets": data})

@app.route("/cryptoping", methods=["POST"])
def handleCryptoPing():
    print(request.get_json())


if __name__ == '__main__':
    socketio.run(app)
    # arbitrageBot = ArbitrageBot()
    app.run()