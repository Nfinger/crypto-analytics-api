from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstmHelper
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
from binance.enums import *

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

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

data = client.get_klines(symbol="ETHUSDT", interval="5m")
df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'Close time' 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Can be ignored', 'extra'])

# df['date'] = pd.to_datetime(df['date'], unit='s')
df.set_index('date')
df = df.sort_index(axis=0, ascending=False)

values = df['close'].values.reshape(-1,1)
values = values.astype('float32')
#Step 1 Load Data
X_train, y_train, X_test, y_test = lstmHelper.load_data(values, 1, True)

model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=1,
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(500))

model.add(Dropout(0.2))

model.add(Dense(output_dim=1))

model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05
)

predictions = lstmHelper.predict_sequences_multiple(model, X_test, 1, 1)
lstmHelper.plot_results_multiple(predictions, y_test, 1)