import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout, GRU
from keras.callbacks import LambdaCallback
from keras.optimizers import SGD, adam

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

import matplotlib.pyplot as plt

wandb.init()
config = wandb.config
config.repeated_predictions = False
config.look_back = 20
config.opt = SGD(lr=0.001, momentum=0.9)


def data_normalizer(df, inplace = False):
    if inplace:
        for column in df.columns:
            data = df[f'{column}'].values
            max_val = max(data)
            min_val = min(data)
            dif = max_val - min_val
            df[f'{column}'] = df[f'{column}'].map(lambda x: (x - min_val) / dif)
    if not inplace:
        for column in df.columns:
            temp_df = df
            data = df[f'{column}'].values
            max_val = max(data)
            min_val = min(data)
            dif = max_val - min_val
            temp_df[f'{column}'] = df[f'{column}'].map(lambda x: (x - min_val) / dif)
            return temp_df

def load_data(data_type="airline"):
    data = pd.DataFrame
    if data_type == "flu":
        df = pd.read_csv('flusearches.csv')
        data = df.flu.astype('float32').values
    elif data_type == "airline":
        data = pd.read_csv('datasets/international-airline-passengers.csv', index_col=0)
    elif data_type == "sin":
        data = pd.read_csv('datasets/sin.csv', index_col=0)
    elif data_type == "plpl_2000":
        df = pd.read_csv('plpl_2000.csv')
        data = df['aver2000'].astype('float32').values
    elif data_type == "plpl_60":
        df = pd.read_csv('plpl_60.csv')
        data = df['aver60'].astype('float32').values
    elif data_type == "tsla":
        data = pd.read_csv('datasets/tsla.csv', index_col=0)
    elif data_type == "line":
        data = np.array([i for i in range(1000)]).astype('float32')
    elif data_type == "con":
        data = np.array([0.5 for i in range(1000)]).astype('float32')
    elif data_type == 'sig':
        data = pd.read_csv('datasets/sig.csv')
    elif data_type == 'gauss':
        data = pd.read_csv('datasets/gauss.csv')
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - config.look_back):
        a = dataset[i:(i + config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)

def test_size_create(data, alpha=0.15):
    if int(len(data) * alpha) > config.look_back + 10:
        test_size = int(len(data) * alpha)
    else:
        test_size = config.look_back + 10
    return test_size

temp_data = load_data('gauss')
data = data_normalizer(temp_data[['gauss']].astype('float32'))
data = data['gauss'].values
'''temp_data.drop('Volume', axis=1, inplace=True)
temp_data['aver'] = (temp_data['Open'] + temp_data['High'] + temp_data['Low'] + temp_data['Close'] + temp_data['Adj Close'])/5
data = data_normalizer(temp_data[['aver']].astype('float32'))
data = data['aver'].values'''

# split into train and test sets
test_size = test_size_create(data, alpha=0.5)
X, Y = create_dataset(data)
trainX = X[:-(config.look_back + 1 + test_size)]
testX = X[-(config.look_back + 1 + test_size):]
trainY = Y[:-(config.look_back + 1 + test_size)]
testY = Y[-(config.look_back + 1 + test_size):]

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]


model = Sequential()
model.add(GRU(50, input_shape=(config.look_back, 1), activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
model.fit(trainX, trainY, epochs=1000, batch_size=32, validation_data=(testX, testY), shuffle=False, verbose=2,
          callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])
