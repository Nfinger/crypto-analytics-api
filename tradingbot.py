import gdax, time
import pandas as pd
import numpy as np
from numpy import concatenate
# import talib as ta
import gdax
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from binance.client import Client
from binance.enums import *

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

class TradingBot:
    def __init__(self):
        self.USD = 0
        self.holdings = 0
        self.trades = []
        self.stopLoss = 0
        self.predictions = np.array([])
        self.ticks = 0  # 28
        self.position = 0  # 29
        self.df = pd.DataFrame()  # 30
        self.momentum = 12  # 31

    def buy(self, trade):
        self.holdings += self.USD / float(trade['close'].item())
        self.USD = 0
        self.trades.append({ 'date': trade['date'].item(), 'type': "buy", 'price': trade['close'].item(), 'low': trade['low'].item(), 'high': trade['high'].item() },)

    def sell(self, trade):
        self.USD += self.holdings * float(trade['close'].item())
        self.holdings = 0
        self.trades.append({ 'date': trade['date'].item(), 'type': "sell",' price': trade['close'].item(), 'low': trade['low'].item(), 'high': trade['high'].item() },)

    def trade(usd, holdings):
        tradingbot = TradingBot()
        tradingbot.USD = usd
        tradingbot.holdings = holdings
        # class WebsocketClient(gdax.WebsocketClient):
        #     def on_open(self):
        #         self.url = "wss://ws-feed.gdax.com/"
        #         self.products = ["ETH-USD"]

        #     def on_message(self, msg):
        #         if msg["type"] == 'done' and msg["reason"] == 'filled' and "price" in msg:
        #             pass

        #     def on_close(self):
        #         print("-- Goodbye! --")

        # wsClient = WebsocketClient()
        # wsClient.start()
        # print(wsClient.url, wsClient.products)
        # while (True):
        #     time.sleep(1)
        # wsClient.close()

        pair = "ETH-USD"    # Use ETH pricing data on the BTC market

        # Request data from gdax
        public_client = gdax.PublicClient()
        data = public_client.get_product_historic_rates(pair, granularity=300)

        # Convert to Pandas dataframe with datetime format
        DataFrame = pd.DataFrame(data, columns=['date', 'low', 'high', 'open', 'close', 'volume'])
        DataFrame['date'] = pd.to_datetime(DataFrame['date'], unit='s')
        DataFrame = DataFrame.sort_index(axis=0, ascending=False)

        ##
        # data = client.get_klines(symbol="ETHUSDT", interval="1m")
        # DataFrame = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'Close time' 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Can be ignored', 'extra'])

        # csv_df = pd.read_csv("%s.csv" % pair)
        # joined_df = pd.concat([csv_df, DataFrame], ignore_index=True)
        # joined_df.to_csv("%s.csv" % pair)

        # Technical Analysis
        RSI_PERIOD = 14
        RSI_AVG_PERIOD = 15
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        STOCH_K = 14
        STOCH_D = 5

        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        def window_transform_series(series, window_size):
            # containers for input/output pairs
            X = []
            y = []
            for i in range(len(series) - window_size):
                X.append(series[i:i+window_size])
                y.append(series[i+window_size])
            # reshape each 
            X = np.asarray(X)
            X.shape = (np.shape(X)[0:2])
            y = np.asarray(y)
            y.shape = (len(y),1)

            return X,y

        model = load_model('classification_model.h5')
        window_size = 5

        for i in range(1, DataFrame.shape[0]):
            ## Uncomment if statement if were calculating stochastic
            # if DataFrame.shape[0] < 22:
            #     return
            df = DataFrame[:i]
            # Graphical approach
            CDL3BLACKCROWS = ta.CDL3BLACKCROWS(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDL3INSIDE = ta.CDL3INSIDE(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDL3LINESTRIKE = ta.CDL3LINESTRIKE(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDL3STARSINSOUTH = ta.CDL3STARSINSOUTH(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLHAMMER = ta.CDLHAMMER(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDL3WHITESOLDIERS = ta.CDL3WHITESOLDIERS(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLADVANCEBLOCK = ta.CDLADVANCEBLOCK(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLCONCEALBABYSWALL = ta.CDLCONCEALBABYSWALL(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLDARKCLOUDCOVER = ta.CDLDARKCLOUDCOVER(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLENGULFING = ta.CDLENGULFING(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLGAPSIDESIDEWHITE = ta.CDLGAPSIDESIDEWHITE(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLHANGINGMAN = ta.CDLHANGINGMAN(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            CDLTHRUSTING = ta.CDLTHRUSTING(df.open.as_matrix(), df.high.as_matrix(), df.low.as_matrix(), df.close.as_matrix())
            
            # print(CDLTHRUSTING)
            macd, macdSignal, macdHist = ta.MACD(np.array([float(x) for x in df.close.as_matrix()]), fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
            # # stoch_k, stoch_d = ta.STOCH(DataFrame.high.as_matrix(), DataFrame.low.as_matrix(), DataFrame.close.as_matrix(), slowk_period=STOCH_K, slowd_period=STOCH_D)
            # # ema = ta.EMA(df.close.as_matrix().astype('float32'), timeperiod=50)
            # # rsi = ta.RSI(df.close.as_matrix().astype('float32'), RSI_PERIOD)
            # # if  rsi[-1:] > 70 and tradingbot.holdings > 0:
            # #     tradingbot.sell(DataFrame[-1:])
            # # if rsi[-1:] < 30 and tradingbot.USD > 0:
            # #     tradingbot.buy(DataFrame[-1:])

            # ### AI approach
            # ## Single variable
            # values = df['close'].values.reshape(-1,1)
            # values = values.astype('float32')
            # scaler = MinMaxScaler(feature_range=(0, 1))
            # dataset_norm = scaler.fit_transform(values)

            # # window the data using your windowing function
            # X,y = window_transform_series(series = dataset_norm,window_size = window_size)

            # # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
            # X = np.asarray(np.reshape(X, (X.shape[0], window_size, 1)))

            # # generate predictions
            # yhat = model.predict(X)

            ## Classification
            df = df.drop(columns=["date", "volume"])
            df = df.fillna(0)
            # One-hot encoding the action
            processed_data = pd.get_dummies(df)
            
            # stoch_k, stoch_d = ta.STOCH(processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix(), slowk_period=STOCH_K, slowd_period=STOCH_D)
            # rsi = ta.RSI(processed_data.close.as_matrix(), RSI_PERIOD)
            # processed_data['rsi'] = rsi
            # processed_data['stoch_k'] = stoch_k
            # processed_data['ULTOSC'] = ta.ULTOSC(processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix(), timeperiod1=7, timeperiod2=14, timeperiod3=28)
            # Normalizing the close and the open scores to be in the interval (0,1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            processed_data["CDL3BLACKCROWS"] = ta.CDL3BLACKCROWS(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDL3INSIDE"] = ta.CDL3INSIDE(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDL3LINESTRIKE"] = ta.CDL3LINESTRIKE(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDL3STARSINSOUTH"] = ta.CDL3STARSINSOUTH(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLHAMMER"] = ta.CDLHAMMER(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDL3WHITESOLDIERS"] = ta.CDL3WHITESOLDIERS(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLADVANCEBLOCK"] = ta.CDLADVANCEBLOCK(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLCONCEALBABYSWALL"] = ta.CDLCONCEALBABYSWALL(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLDARKCLOUDCOVER"] = ta.CDLDARKCLOUDCOVER(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLENGULFING"] = ta.CDLENGULFING(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLGAPSIDESIDEWHITE"] = ta.CDLGAPSIDESIDEWHITE(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLHANGINGMAN"] = ta.CDLHANGINGMAN(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["CDLTHRUSTING"] = ta.CDLTHRUSTING(processed_data.open.as_matrix(), processed_data.high.as_matrix(), processed_data.low.as_matrix(), processed_data.close.as_matrix())
            processed_data["close"] = scaler.fit_transform(processed_data["close"].values.reshape(-1,1))
            processed_data["open"] = scaler.fit_transform(processed_data["open"].values.reshape(-1,1))
            processed_data["high"] = scaler.fit_transform(processed_data["high"].values.reshape(-1,1))
            processed_data["low"] = scaler.fit_transform(processed_data["low"].values.reshape(-1,1))
            # print(processed_data['rsi'])
            # processed_data = processed_data[np.isfinite(processed_data['rsi'])]
            # processed_data["rsi"] = scaler.fit_transform(processed_data["rsi"].values.reshape(-1,1))
            # processed_data["stoch_k"] = scaler.fit_transform(processed_data["stoch_k"].values.reshape(-1,1))
            # processed_data["ULTOSC"] = scaler.fit_transform(processed_data["ULTOSC"].values.reshape(-1,1))
            processed_data["CDL3BLACKCROWS"] = scaler.fit_transform(processed_data["CDL3BLACKCROWS"].values.reshape(-1,1))
            processed_data["CDL3INSIDE"] = scaler.fit_transform(processed_data["CDL3INSIDE"].values.reshape(-1,1))
            processed_data["CDL3LINESTRIKE"] = scaler.fit_transform(processed_data["CDL3LINESTRIKE"].values.reshape(-1,1))
            processed_data["CDL3STARSINSOUTH"] = scaler.fit_transform(processed_data["CDL3STARSINSOUTH"].values.reshape(-1,1))
            processed_data["CDLHAMMER"] = scaler.fit_transform(processed_data["CDLHAMMER"].values.reshape(-1,1))
            processed_data["CDL3WHITESOLDIERS"] = scaler.fit_transform(processed_data["CDL3WHITESOLDIERS"].values.reshape(-1,1))
            processed_data["CDLADVANCEBLOCK"] = scaler.fit_transform(processed_data["CDLADVANCEBLOCK"].values.reshape(-1,1))
            processed_data["CDLCONCEALBABYSWALL"] = scaler.fit_transform(processed_data["CDLCONCEALBABYSWALL"].values.reshape(-1,1))
            processed_data["CDLDARKCLOUDCOVER"] = scaler.fit_transform(processed_data["CDLDARKCLOUDCOVER"].values.reshape(-1,1))
            processed_data["CDLENGULFING"] = scaler.fit_transform(processed_data["CDLENGULFING"].values.reshape(-1,1))
            processed_data["CDLGAPSIDESIDEWHITE"] = scaler.fit_transform(processed_data["CDLGAPSIDESIDEWHITE"].values.reshape(-1,1))
            processed_data["CDLHANGINGMAN"] = scaler.fit_transform(processed_data["CDLHANGINGMAN"].values.reshape(-1,1))
            processed_data["CDLTHRUSTING"] = scaler.fit_transform(processed_data["CDLTHRUSTING"].values.reshape(-1,1))

            # Splitting the data input into X, and the labels y 
            X = np.array(processed_data)[:,1:]
            X = X.astype('float32')

            ## Multi
            # values = df[['close'] + ['open'] + ['high']  + ['low']  + ['volume']].values
            # values = values.astype('float32')

            # scaler = MinMaxScaler(feature_range=(0, 1))
            # scaled = scaler.fit_transform(values)

            # reframed = series_to_supervised(scaled, 1, 1)
            # reframed.drop(reframed.columns[[8,9]], axis=1, inplace=True)
            # ## Split into training and testing
            # values = reframed.values
            # n_train_hours = int(len(values) * 0.7)
            # train = values[:n_train_hours, :]
            # test = values[n_train_hours:, :]
            # # split into input and outputs
            # train_X, train_y = train[:, :-1], train[:, -1]
            # test_X, test_y = test[:, :-1], test[:, -1]
            # # reshape input to be 3D [samples, timesteps, features]
            # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            # ## PREDICT!!
            yhat = model.predict(X)
            
            # loop through perdictions and consolidate
            # final = np.array([])
            # for entry in yhat:
            #     final = np.append(final, entry)
            # # print(final)
            # final = final[-5:]
            # print(final)
            # macd = np.nan_to_num(macd[-5:])
            print(yhat[-1])
            #  or CDL3BLACKCROWS[-1] == -100 or CDLENGULFING[-1] == -100 or CDLTHRUSTING[-1] == -100 or (DataFrame[:i][-1:]['close'].values[0] < tradingbot.stopLoss)
            #   or CDL3BLACKCROWS[-1] == 100 or CDLHAMMER[-1] == 100
                    # or CDL3WHITESOLDIERS[-1] == 100 or CDL3WHITESOLDIERS[-1] == -100 or CDLENGULFING[-1] == 100
                    # or CDLGAPSIDESIDEWHITE[-1] == 100
            if (np.argmax(yhat[-1]) == 0) and tradingbot.holdings > 0:
                tradingbot.stopLoss = 0
                tradingbot.sell(DataFrame[:i][-1:])
            elif (np.argmax(yhat[-1]) == 2) and tradingbot.USD > 0:
                tradingbot.buy(DataFrame[:i][-1:])
                value = float(DataFrame[:i][-1:]['close'].values[0])
                tradingbot.stopLoss = value - (value * 0.1)

        print("Done!")
        return tradingbot.trades, tradingbot.USD, tradingbot.holdings, tradingbot.predictions
