import datetime as dt

kTickerSymbol = 'AAPL'
kFinanceAPI = 'yahoo'

kStartDate = dt.datetime(2012, 1, 1)
kEndDate = dt.datetime(2020, 1, 1)

# days in the past to base prediction on
kPredictionDays = 1825

# how sophisticated you want the model to be
kLayers = 50

kTestStartDate = dt.datetime(2020, 1, 1)
kTestEndDate = dt.datetime.now()