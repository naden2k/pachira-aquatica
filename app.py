# Importing modules
import pandas as pd
import requests
import alpaca
import plotly.express as px
from plotly import graph_objs as go
from datetime import datetime, date, timedelta
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', figsize=(12, 10))
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

# Input data retreival parameters 
request_params = StockBarsRequest(
                    symbol_or_symbols=["SPXL","SPXS"],
                    timeframe=TimeFrame.Minute,
                    start="2022-09-22 00:00:00",
                    end = "2023-09-22 23:59:00")

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
df.rename(columns={'open':'spxl'},inplace=True)
# Renaming second instance of 'open' (Renaming SPXS opening price column from open to 'spxs')
df['spxs'] = data.loc[:,data.columns.duplicated()].copy()
# Adding percent growth of SPXL and SPXS over each minute
df['spxl_growth'] = df.spxl.pct_change() * 100
df['spxs_growth'] = df.spxs.pct_change() * 100

# Calculating SMA and Bollinger Bands for SPXL
df['spxl_upper2'] = df.spxl_growth.rolling(20).mean() + (df.spxl_growth.rolling(20).std() * 2)
df['spxl_upper1'] = df.spxl_growth.rolling(20).mean() + df.spxl_growth.rolling(20).std()
df['spxl_sma'] = df.spxl_growth.rolling(20).mean()

# Calculating SMA and Bollinger Bands for SPXS
df['spxs_upper2'] = df.spxs_growth.rolling(20).mean() + (df.spxs_growth.rolling(20).std() * 2)
df['spxs_upper1'] = df.spxs_growth.rolling(20).mean() + df.spxs_growth.rolling(20).std()
df['spxs_sma'] = df.spxs_growth.rolling(20).mean()

# STATS
print("\n~~ SPXL ~~")
print("--------------------")
print("\nMax SPXL Growth: ", df.spxl_growth.max().round(3))
print("Min SPXL Growth: ", df.spxl_growth.min().round(3))
print("\nMean SPXL Growth: ",df.spxl_growth.mean().round(3))
print("Median SPXL Growth: ",df.spxl_growth.median().round(3))

print('\n~~ SPXS ~~')
print("--------------------")
print("\nMax SPXS Growth: ", df.spxs_growth.max().round(3))
print("Min SPXS Growth: ", df.spxs_growth.min().round(3))
print("\nMean SPXS Growth: ",df.spxs_growth.mean().round(3))
print("Median SPXS Growth: ",df.spxs_growth.median().round(3))

# Starting shares
df['spxs_shares'] = 1.000
df['spxl_shares'] = 1.000

df['tot_spxs'] = df.spxs_shares * df.spxs
df['tot_spxl'] = df.spxl_shares * df.spxl

numb_of_trades = 0

# For every row in range of dataframe rows (idx1 = starts at first row)
for idx1 in range(len(df)-1):
    # Get the row directly after idx1
    idx2 = idx1 + 1
    # Price of spxl at time 1
    spxl1 = df.spxl[idx1]
    # Price of spxl at time 2
    spxl2 = df.spxl[idx2]

    # Price of spxl at time 1
    spxs1 = df.spxs[idx1]
    # Price of spxl at time 2
    spxs2 = df.spxs[idx2]
    if df.spxl_growth[idx2] >= df.spxl_upper1[idx2]:
    # if df.spxl_growth[idx2] >= 0.00119:
    #if df.spxl_growth[idx2] >= 0.006916492916150138:
        # Sell the difference between spxl price at time 2 and spxl price at time 1
        sell = spxl2 - spxl1
        
        # New spxl share count at time 2 is (spxl price at time 1 divided by spxl price at time 2) multiplied by our spxl share count at time 1
        # Example: If spxl is $5 per share and goes to $8 per share and we own 1 share, our new shares after selling $3 or 0.37 shares is:
        # new shares owned of spxl $5/$8 * 1 share = 0.62 * 1 share = 0.62 shares
        df.loc[idx2:, 'spxl_shares'] = (spxl1 / spxl2) * df.spxl_shares[idx1]
        
        # Total spxl at time 2 is spxl shares owned at time 2 multiplied by spxl price at time 2
        df.loc[idx2:,'tot_spxl'] = df.spxl_shares[idx2] * spxl2
        
        #df7.tot_spxl[idx1] + (df7.spxl_shares[idx2] * spxl2)
        
        plus_shares = sell / spxs2
        
        df.loc[idx2:,'spxs_shares'] = df.spxs_shares[idx1] + plus_shares

        #df7.loc[idx2:,'tot_spxl'] = df7.tot_spxl[idx1] + (df7.spxl_shares[idx2] * spxl2)

        df.loc[idx2:,'tot_spxs'] = df.spxs_shares[idx2] * spxs2
        
        #df7.tot_spxs[idx1] + (spxs2 * plus_shares)

        numb_of_trades = numb_of_trades + 1

        df = df.copy()
    elif df.spxs_growth[idx2] >= df.spxs_upper2[idx2]:
        # if df.spxs_growth[idx2] >= 0.033362023346761444:
            # Sell the difference between spxl price at time 2 and spxl price at time 1
            sell = spxs2 - spxs1
            
            df.loc[idx2:,'spxs_shares'] = (spxs1 / spxs2) * df.spxs_shares[idx1]
            
            df.loc[idx2:,'tot_spxs'] = df.spxs_shares[idx2] * spxs2
            
            plus_shares = sell / spxl2
            #I dont know why it works but it works
            # plus_shares = sell / spxs2
            
            df.loc[idx2:,'spxl_shares'] = df.spxl_shares[idx1] + plus_shares

            df.loc[idx2:,'tot_spxl'] = df.spxl_shares[idx2] * spxl2

            numb_of_trades = numb_of_trades + 1

            df = df.copy() 

# # Remove hashtag to import the ouput dataframe to excel   
# df.to_excel('pachira_aquatica.xlsx')

beg_investment = df.tot_spxl.head(1).values + df.tot_spxs.head(1).values
end_indvestment = df.tot_spxs.tail(1).values + df.tot_spxl.tail(1).values
total_profit = end_indvestment - beg_investment
profit_margin = (total_profit / end_indvestment) * 100

print("\n\nRESULTS")
print('----------------')
print("Total Number of Trades: ", numb_of_trades)
print("\nEnding Investment Value")
print(round(end_indvestment[0],3))
print("\nBeginning Investment Value")
print(round(beg_investment[0],3))
print("\nTotal Profit")
print(round(total_profit[0],3)),
print("\nProfit Margin")
print(round(profit_margin[0],3),"%")

# Show df on GUI
st.dataframe(data = df)

df_plt = df.tail(100)
#.set_index('timestamp')
df_plt.spxl_upper2.plot(label='SPXL 2 STD Upper Bound')
df_plt.spxl_upper1.plot(label='SPXL 1 STD Upper Bound')
df_plt.spxl_sma.plot(label='SPXL SMA')
df_plt.spxl_growth.plot(label='SPXL Growth')

df_plt.spxs_upper2.plot(label='SPXS 2 STD Upper Bound')
df_plt.spxs_upper1.plot(label='SPXS 1 STD Upper Bound')
df_plt.spxs_sma.plot(label='SPXS SMA')
df_plt.spxs_growth.plot(label='SPXS Growth')


plt.legend()