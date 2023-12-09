# Importing modules
import pandas as pd
import requests
import alpaca
from datetime import datetime, date, timedelta
import numpy as np
# Importing alpaca trading modules (i.e to execute trades)
from alpaca.trading.client import TradingClient
import alpaca_trade_api as tradeapi
# Importing alpaca historicacl data module
from alpaca.data.historical import StockHistoricalDataClient
# Import alpaca's time frame module
from alpaca.data.timeframe import TimeFrame
# Import alpaca's stock data module
from alpaca.data.requests import StockBarsRequest

# Passing in our api key
api_key = 'PKF7DLHP2UGIVNVQ2GVS'
# Passing in our password
api_secret = 'tYZRksZMPvKXpMia5UNBGaCjqn86a5IKwJkkCcXP'

# Create an alpaca trading client object using the TradingClient module we imported and our api key and api secret key 
trading_client = TradingClient(api_key, api_secret, paper=True)
# Create 'TradeAccount' object
account = trading_client.get_account()
# Get our account number
print(account.account_number) 
# Create alpaca_trade_api.REST object using apaca_trade_api module 
api = tradeapi.REST(api_key, api_secret, "https://paper-api.alpaca.markets/")
# List our current positions 
api.list_positions()

# Create StockHistoricalDataClient object 
historical_client = StockHistoricalDataClient(api_key, api_secret)

# st.date_input(label,
# Input data retreival parameters 
request_params = StockBarsRequest(
                    symbol_or_symbols=["SPXL","SPXS"],
                    timeframe=TimeFrame.Minute,
                    start="2019-12-01 00:00:00",
                    end = "2023-12-08 23:59:00")

# Retreive the data object 
bars = historical_client.get_stock_bars(request_params)
# Create a dataframe with the data object
bars_df = bars.df

# Matching SPXL and SPXS by date/time and dropping observations at times for which there is no data for SPXL OR SPXS (emphasis on the OR. I didn't say AND)
new_bars = bars_df.stack().unstack(level=1).dropna(axis=1).T.reset_index()
# Converting timestamp column to pandas datetime object
new_bars.timestamp = pd.to_datetime(new_bars.timestamp)
# Creating a subset of the dataframe to include only relavent information and dropping multi-level columns
data = pd.DataFrame([new_bars.timestamp,new_bars.SPXL.open,new_bars.SPXS.open]).T
# Drop column that has duplicated column name
df = data.loc[:, ~data.columns.duplicated()].copy()
# Renaming first instance of 'open' (renaming SPXL opening price column from open to 'spxl')
df.rename(columns={'open':'SPXL'},inplace=True)
# Renaming second instance of 'open' (Renaming SPXS opening price column from open to 'spxs')
df['SPXS'] = data.loc[:,data.columns.duplicated()].copy()
#create data csv
df.to_csv('data.csv')
