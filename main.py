import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScalar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data

tickerSymbol = 'AAPL'
financeAPI = 'yahoo'

startDate = dt.datetime(2012, 1, 1)
endDate = dt.datetime(2020, 1, 1)

data = web.DataReader(tickerSymbol, financeAPI, startDate, endDate)

# Prepare Data

scalar = MinMaxScalar(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))

# days in the past to base prediction on
prediction_days = 1825

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.array(x_train, (x_train.shape[0], x_train.shape[1], 1))
