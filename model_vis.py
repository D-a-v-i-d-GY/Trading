import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout, GRU
from keras.callbacks import LambdaCallback

import sklearn
from sklearn import svm, preprocessing

import pickle

window_size = 20
model = load_model('wandb/run-20200413_104420-apxmhm45/model-best.h5')

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
def lazy_switch(X, test=True, alpha=0.15):
    if test:
        return test_size_create(X, alpha=alpha)
    if not test:
        return len(X)
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size):
        a = dataset[i:(i + window_size)]
        dataX.append(a)
        dataY.append(dataset[i + window_size])
    return np.array(dataX), np.array(dataY)
def test_size_create(data, alpha=0.15):
    if int(len(data) * alpha) > window_size + 10 + 1:
        test_size = int(len(data) * alpha) + window_size + 1
    else:
        test_size = window_size + 1 + 10
    return test_size
def create_dataset_by_pred(data, model, N, look_back, pure_pred=True):
    if pure_pred:
        temp = N
        pred_data = data
        while temp != 0:
            next_input = pred_data[-look_back:][np.newaxis, :, np.newaxis]
            next_data = model.predict(next_input)
            pred_data = np.append(pred_data, next_data)
            pred_data = pred_data
            temp -= 1
        return pred_data
    else:
        return model.predict(data[:, :, np.newaxis])


temp_data = load_data('gauss')
data_normalizer(temp_data, inplace=True)
data = temp_data['gauss'].astype('float32').values

'''temp_data.drop('Volume', axis=1, inplace=True)
temp_data['aver'] = (temp_data['Open'] + temp_data['High'] + temp_data['Low'] + temp_data['Close'] + temp_data['Adj Close'])/5
data = data_normalizer(temp_data[['aver']].astype('float32'))
data = data['aver'].values'''
'''max_val = max(data)
min_val = min(data)
data = (data - min_val) / (max_val - min_val)'''

#=======================================================================================================================
X, Y = create_dataset(data)
test_size = lazy_switch(X, test=True, alpha=0.5)
trainX = X[:-test_size]
testX = X[-test_size:]
trainY = Y[:-test_size]
testY = Y[-test_size:]

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]
#=======================================================================================================================
pred = create_dataset_by_pred(X[:], model, test_size - window_size, window_size, pure_pred=False)
#mean = abs(pred-testY[window_size + 1:-window_size])

#for i in range(len(pred)):
#    mean[i] /= testY[i]
#print(mean.mean())

ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1), sharex=ax1, sharey=ax1)

ax1.plot([i/len(pred[-test_size:]) for i in range(len(pred[-test_size:]))], pred[-test_size:], color='red')
ax1.plot([i/len(Y[-test_size:]) for i in range(len(Y[-test_size:]))], Y[-test_size:], color='blue')

ax2.scatter(Y[-test_size:], pred[-test_size:], s=0.2, color='red')
ax2.plot([i/len(Y[-test_size:]) for i in range(len(Y[-test_size:]))], [i/len(Y[-test_size:]) for i in range(len(Y[-test_size:]))], color='blue')

plt.xlim((0, 1.05))
plt.ylim((0, 1.05))

plt.show()

