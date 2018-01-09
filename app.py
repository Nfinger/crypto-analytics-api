from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask.ext.jsonpify import jsonify
from flask_cors import CORS

from binance.client import Client
from binance.enums import *

import urllib.request

from tradingbot import TradingBot

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

app = Flask(__name__)
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

@app.route("/twitter")
def getTweets():
    return urllib.request.urlopen("https://api.twitter.com/1.1/search/tweets.json?q=BTC").read()


if __name__ == '__main__':
     app.run()