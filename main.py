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

