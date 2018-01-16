from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask.ext.jsonpify import jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from binance.client import Client
from binance.enums import *
from twitter import *

consumer_key = "BPlzYeWngcK8vluAmNLIoiBgH"
consumer_secret = "nPobDCHkLZKjl8Y1BFm4PiJyCSKB1bM7U9cpLhjjGPqdi6unz4"
token = "948764981205569536-BsidADMWaOTFG8qIP58EnD1eppQEX3a"
token_secret = "8Rl3J0Mhg8PK6m6sASz3i7Y7VgKUBDHPAEgce7rFyPAr1"

t = Twitter(
    auth=OAuth(token, token_secret, consumer_key, consumer_secret))

import urllib.request

from tradingbot import TradingBot

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

app = Flask(__name__)

# app.config['MONGO_DBNAME'] = 'traderdb'
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/traderdb'
# mongo = PyMongo(app)
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

@app.route("/trades")
def getTrades():
    usd = request.args.get('usd')
    holdings = request.args.get('holdings')
    trades, usd, holdings, predictions = TradingBot.trade(int(usd), int(holdings))
    return jsonify({'result' : trades, 'usd':usd, 'holdings': holdings, 'predictions':predictions.tolist()})

@app.route("/twitter")
def getTweets():
    symbol = request.args.get('symbol')
    print("%s" % symbol)
    searchParam = "%s" % symbol
    tweets = t.search.tweets(q=searchParam)
    return jsonify({"tweets": tweets})


if __name__ == '__main__':
     app.run()