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

# Request data from gdax
public_client = gdax.PublicClient()
window_size = 7

def build_RNN():
    model = Sequential()
    model.add(LSTM(512, input_shape = (window_size,1), activation="softsign", init='glorot_normal', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(512, activation="softsign"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

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
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_norm = scaler.fit_transform(values)

# window the data using your windowing function
X,y = window_transform_series(series = dataset_norm,window_size = window_size)

# split our dataset into training / testing sets
train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

# partition the training set
X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

# keep the last chunk for testing
X_test = X[train_test_split:,:]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

model = build_RNN()

# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=0.01, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

# run your model!
model.fit(X_train, y_train, epochs=300, batch_size=500, verbose=1)
model.save('%s_model.h5' % symbol)

# returns a compiled model
model = load_model('%s_model.h5' % symbol)
# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))


### Plot everything - the original series as well as predictions on training and testing sets
import matplotlib.pyplot as plt
# %matplotlib inline
inv_dataset = scaler.inverse_transform(dataset_norm)
inv_train = scaler.inverse_transform(train_predict)
inv_test = scaler.inverse_transform(test_predict)
# plot original series
plt.plot(inv_dataset,color = 'k')

# plot training set prediction
split_pt = train_test_split + window_size 
plt.plot(np.arange(window_size,split_pt,1),inv_train,color = 'b')

# plot testing set prediction
plt.plot(np.arange(split_pt,split_pt + len(inv_test),1),inv_test,color = 'r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('(normalized) price')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()