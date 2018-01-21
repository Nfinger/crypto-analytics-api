import util
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from   sys import argv,exit
from   keras.models import Sequential
from   keras.layers import Dense,Dropout,GRU,Reshape
from   keras.layers.normalization import BatchNormalization
from binance.client import Client
from binance.enums import *

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

def buildNet(w_init="glorot_uniform",act="tanh"):
    global net
    print("Building net..",end="")
    net = Sequential()
    net.add(Dense(12,kernel_initializer=w_init,input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(BatchNormalization())
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.3))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=False))
    net.add(Dropout(0.4))
    net.add(Dense(1,kernel_initializer=w_init,activation='linear'))
    net.compile(optimizer='nadam',loss='mse')
    print("done!")

def chart(real,predicted,show=True):
    plt.plot(real,color='g')
    plt.plot(predicted,color='r')
    plt.ylabel('BTC/USD')
    plt.xlabel("9Minutes")
    plt.savefig("chart.png")
    if show:plt.show()

def predictFuture(m1,m2,old_pred,writeToFile=False):
    actual,latest_p = util.getCurrentData(label=True)
    actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
    pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    pred = float(int(pred[0]*100)/100)
    if writeToFile:
        f = open("results","a")
        f.write("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$\n".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
        f.close()

    print("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
    return latest_p,pred

window_size = 7

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

#Assembling Net:
buildNet()
epochs = args.iterations
#Training dnn
print("training...")
el = len(data)-10     #Last ten elements are for testing
net.fit(data[:el],labels[:el],epochs=epochs,batch_size=300)
print("trained!\nSaving...",end="")
net.save_weights("model.h5")
print("saved!")

### Predict all over the dataset to build the chart
reals,preds = [],[]
for i in range(len(data)-40,len(data)):
    x = np.array(data[i]).reshape(1,12)
    predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
    real = util.augmentValue(labels[i],m1,m2)
    preds.append(predicted)
    reals.append(real)

### Predict Price the next 9m price (magic)
real,hip = predictFuture(m1,m2,0)
reals.append(real)
preds.append(hip)

### PLOTTING
chart(reals,preds)