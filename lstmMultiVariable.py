import gdax
import pandas as pd
import numpy as np
from numpy import concatenate
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from matplotlib import pyplot
import keras
# import talib as ta
from binance.client import Client
from binance.enums import *

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

def build_RNN(train_X):
    model = Sequential()
    model.add(LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

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

symbol = "ETH"
## GDAX
# public_client = gdax.PublicClient()
# data = public_client.get_product_historic_rates(symbol, granularity=300)
# # Convert to Pandas dataframe with datetime format
# df = pd.DataFrame(data, columns=['date', 'low', 'high', 'open', 'close', 'volume'])
## Binance
# fetch 1 minute klines for the last day up until now

data = client.get_historical_klines(symbol="ETHUSDT", interval="5m", start_str="1 Nov, 2017")
df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'Close time' 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Can be ignored', 'extra'])

# df['date'] = pd.to_datetime(df['date'], unit='s')
df.set_index('date')
df = df.sort_index(axis=0, ascending=False)

values = df['close'].values.reshape(-1,1)
values = values.astype('float32')
print(values)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_norm = scaler.fit_transform(values)

### Lets add more features to analyze

values = df[['close'] + ['open'] + ['high']  + ['low']  + ['volume']].values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
print(reframed.head())
## Split into training and testing
values = reframed.values
n_train_hours = int(len(values) * 0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

## Build new model
multi_model = build_RNN(train_X)
# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
multi_model.compile(loss='mean_squared_error', optimizer=optimizer)
multi_history = multi_model.fit(train_X, train_y, epochs=200, batch_size=500, validation_data=(test_X, test_y), verbose=1, shuffle=False)
multi_model.save('ETH_multi_model.h5')  # creates a HDF5 file 'my_model.h5'
## PREDICT!!
yhat = multi_model.predict(test_X)

## Scaler Inverse Y back to normal value

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

pyplot.plot(inv_y, label='actual')
pyplot.plot(inv_yhat, label='perdict', alpha=0.5)
pyplot.legend()
pyplot.show()