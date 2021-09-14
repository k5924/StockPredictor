import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from constants import *


def LoadData(startdate, enddate):
    return web.DataReader(kTickerSymbol, kFinanceAPI, startdate, enddate)


def ReshapeData(array):
    return np.reshape(array, (array.shape[0], array.shape[1], 1))

# Load Data


data = LoadData(kStartDate, kEndDate)

# Prepare Data

scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))

x_train = []
y_train = []

for x in range(kPredictionDays, len(scaled_data)):
    x_train.append(scaled_data[x - kPredictionDays: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = ReshapeData(x_train)

# Build the Model

model = Sequential()

model.add(LSTM(units=kLayers, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=kLayers, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=kLayers))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Test the models accuracy on existing data

# Load test data

test_data = LoadData(kTestStartDate, kTestEndDate)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(
    total_dataset) - len(test_data) - kPredictionDays:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scalar.transform(model_inputs)

# Make predictions on test data

x_test = []
for x in range(kPredictionDays, len(model_inputs)):
    x_test.append(model_inputs[x-kPredictionDays:x, 0])

x_test = np.array(x_test)
x_test = ReshapeData(x_test)

predicted_prices = model.predict(x_test)
predicted_prices = scalar.inverse_transform(predicted_prices)

# Plot test data predictions
plt.plot(actual_prices, color="black", label=f"Actual {kTickerSymbol} Price")
plt.plot(predicted_prices, color="green",
         label=f"Predicted {kTickerSymbol} Price")
plt.title(f"{kTickerSymbol} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{kTickerSymbol} Share Price")
plt.legend()
plt.show()

# predict one day into the future
real_data = [
    model_inputs[len(model_inputs) + 1 - kPredictionDays: len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = ReshapeData(real_data)

prediction = model.predict(real_data)
prediction = scalar.inverse_transform(prediction)
print(f"PredictionL {prediction}")
