import gdax
import pandas as pd
import numpy as np
from numpy import concatenate
import keras
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
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
import talib as ta
from catboost import Pool, CatBoostClassifier
from binance.client import Client
from binance.enums import *

api_key = 'EcBv9wqxfdWNMhtOI8WbkGb9XwOuITAPxBdljcxv8RYX1H7u2ucC0qokDp2KOWmr'
api_secret = 'i5Y57Gwu8sH9qUE5TbB7zLotm7deTa1D9S8K458LWLXZZzNq5wNAZOHlGJmyjq1s'

client = Client(api_key, api_secret)

# Request data from gdax
public_client = gdax.PublicClient()

def build_RNN():
    model = Sequential()
    model.add(LSTM(100, input_shape = (window_size,1), return_sequences=True, activation="relu"))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(3, activation="softmax"))
    model.summary()
    return model

def window_transform_series(series, labels, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(labels[i+window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)

    return X,y

window_size = 5
symbol = "ETH"
data = pd.read_csv("trade.csv")
# data['date'] = pd.to_datetime(data['date'], unit='s')
data.set_index('date')
data = data.sort_index(axis=0, ascending=False)

# # integer encode
labels = data['action'].values
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)

data = data.drop(columns=["date", "volume", "action"])
print(data.head())
# remove NaNs
data = data.fillna(0)

# One-hot encoding the action
processed_data = pd.get_dummies(data)

# Normalizing the close and the open scores to be in the interval (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
processed_data["close"] = scaler.fit_transform(processed_data["close"].values.reshape(-1,1))
processed_data["open"] = scaler.fit_transform(processed_data["open"].values.reshape(-1,1))
processed_data["high"] = scaler.fit_transform(processed_data["high"].values.reshape(-1,1))
processed_data["low"] = scaler.fit_transform(processed_data["low"].values.reshape(-1,1))

# Splitting the data input into X, and the labels y 
X = np.array(processed_data)[:,1:]
X = X.astype('float32')


# Checking that the input and output look correct
print("Shape of X:", X.shape)
print("\nShape of y:", y.shape)
print("\nFirst 10 rows of X")
print(X[:10])
print("\nFirst 10 rows of y")
print(y[:10])

# break training set into training and validation sets
(X_train, X_test) = X[50:], X[(X.shape[0] - 50):]
(y_train, y_test) = y[50:], y[(y.shape[0] - 50):]

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Building the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(3,)))
model.add(Dropout(.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(3, activation='sigmoid'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# Training the model
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=1, callbacks=[early_stopping_monitor])
model.save('classification_model.h5')

# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(X_test, y_test)
print("\n Testing Accuracy:", score[1])


# returns a compiled model
model = load_model('classification_model.h5')

## CatBoost
# model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass')
# # Fit model
# model.fit(X=X_train, y=y_train)
# # Get predicted classes
# preds_class = model.predict(X_test)
# # Get predicted probabilities for each class
# preds_proba = model.predict_proba(X_test)
# # Get predicted RawFormulaVal
# preds_raw = model.predict(X_test, prediction_type='RawFormulaVal') 



# generate predictions for training
print(X_test.shape)
for i in range(X_test.shape[0]):
    test_predict = model.predict(X_test[:i])
    if (len(test_predict) > 0):
        print(np.argmax(test_predict[len(test_predict) - 1]))







# # Convert to Pandas dataframe with datetime format
# values = df['close'].values.reshape(-1,1)
# values = values.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset_norm = scaler.fit_transform(values)

# # integer encode
# labels = df['action'].values
# print(labels)
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(labels)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# # invert first example
# # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# # print(inverted)

# # window the data using your windowing function
# X,y = window_transform_series(series = dataset_norm, labels=onehot_encoded,window_size = window_size)

# # split our dataset into training / testing sets
# train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

# # partition the training set
# X_train = X[:train_test_split,:]
# y_train = y[:train_test_split]

# # keep the last chunk for testing
# X_test = X[train_test_split:,:]
# y_test = y[train_test_split:]

# # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
# X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
# X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

# model = build_RNN()

# # build model using keras documentation recommended optimizer initialization
# # optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# # compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # run your model!
# model.fit(X_train, y_train, epochs=500, batch_size=500, verbose=1)
# model.save('classification_model.h5')

# # returns a compiled model
# model = load_model('classification_model.h5')


# # print out training and testing errors
# training_error = model.evaluate(X_train, y_train, verbose=0)
# print('training error = ' + str(training_error))

# testing_error = model.evaluate(X_test, y_test, verbose=0)
# print('testing error = ' + str(testing_error))

# # generate predictions for training
# for i in range(X_test.shape[0]):
#     test_predict = model.predict(X_test[:i])
#     if (len(test_predict) > 0):
#         print(np.argmax(test_predict[len(test_predict) - 1]))