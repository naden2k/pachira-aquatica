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
#mpl.rc('figure', figsize=(12, 10))

# Creates header in GUI
st.header('Pachira Aquatica ~ Quant Trading Dashboard', divider='grey')

# Reads in df of all historical prices 
data = pd.read_csv('/Users/audreyanaya/Documents/vscode/Repositories/pachira-aquatica/data.csv')

# Function that creates a specific subset df of specified date inputs
def subset(date1,date2):
    data['date'] = pd.to_datetime(data['timestamp']).dt.date
    # print(date1,type(date1))
    # date1 = pd.to_datetime(date1).dt.date
    # date2 = pd.to_dateime(date2).dt.date
    mask = (data['date'] > date1) & (data['date'] <= date2)
    #df = data.loc[data.timestamp==date1:data.timestamp==date2,:]
    df = data.loc[mask]
    print(df.columns)
    df.reset_index(inplace=True)
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

    df.loc[:,'num_of_trades'] = 0

    # COMPUTATION
    # For every row in range of dataframe rows (idx1 = starts at first row)
    for idx1 in range(len(df)-1):
        #trades = 0
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
            df.loc[idx2:,'num_of_trades'] = df.num_of_trades[idx1] + 1
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
                df.loc[idx2:,'num_of_trades'] = df.num_of_trades[idx1] + 1
                df = df.copy() 
    df = df.drop(columns = {'index','Unnamed: 0'})
    return df
    # # Remove hashtag to import the ouput dataframe to excel   
    # df.to_excel('pachira_aquatica.xlsx')

def stats(df):
    beg_investment = df.tot_spxl.head(1).values + df.tot_spxs.head(1).values
    end_indvestment = df.tot_spxs.tail(1).values + df.tot_spxl.tail(1).values
    total_profit = end_indvestment - beg_investment
    profit_margin = (total_profit / end_indvestment) * 100
    tot_num_trades = df.num_of_trades.tail(1).values 
    
    trades = "Total Number of Trades: " + str(round(tot_num_trades[0],3)) 
    end_inv_value = "Ending Investment Value: " + str(round(end_indvestment[0],3)) 
    beg_inv_value = "Beginning Investment Value: " + str(round(beg_investment[0],3)) 
    tot_prof = "Total Profit: " + str(round(total_profit[0],3)) 
    pm = "Profit Margin: " + str(round(profit_margin[0],3))
    return st.markdown(trades),st.markdown(end_inv_value),st.markdown(beg_inv_value),st.markdown(tot_prof),st.markdown(pm)

def plot(df2):
    df2['inv_value'] = df2.tot_spxl.values + df2.tot_spxs.values
    fig = px.line(df, x="timestamp", y="inv_value", title="Total Investment Value over time")
    return fig
#Create excel file
#excel_df = df.to_excel("pachira_aquatica.xlsx")

# Show df on GUI
date1 = st.date_input("pick a date to start investing")
date2 = st.date_input("pick a date to end investing")
if (date1 != None) & (date2 != None) & (date1 != date2):
    df = subset(date1,date2)
    st.dataframe(df)
    st.download_button('Download to excel',pd.to_excel("SPXL SPXS Trade Data.xlsx"))
    st.subheader('Results')
    stats(df)
    plot(df)

st.download_button("Download data to excel", excel_df)

# df_plt = df.tail(100)
# #.set_index('timestamp')
# df_plt.spxl_upper2.plot(label='SPXL 2 STD Upper Bound')
# df_plt.spxl_upper1.plot(label='SPXL 1 STD Upper Bound')
# df_plt.spxl_sma.plot(label='SPXL SMA')
# df_plt.spxl_growth.plot(label='SPXL Growth')

# df_plt.spxs_upper2.plot(label='SPXS 2 STD Upper Bound')
# df_plt.spxs_upper1.plot(label='SPXS 1 STD Upper Bound')
# df_plt.spxs_sma.plot(label='SPXS SMA')
# df_plt.spxs_growth.plot(label='SPXS Growth')

# plt.legend()