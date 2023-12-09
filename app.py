# Importing modules
import pandas as pd
import requests
import alpaca
import plotly.express as px
from plotly import graph_objs as go
from datetime import datetime, date, timedelta
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#mpl.rc('figure', figsize=(12, 10))

# Creates header in GUI
st.header('Pachira Aquatica ~ Quant Trading Dashboard', divider='grey')

# Reads in df of all historical prices 
data = pd.read_csv('/Users/audreyanaya/Documents/vscode/Repositories/pachira-aquatica/data.csv')

# Function that creates a specific subset df of specified date inputs
def subset(date1,date2):
    data.rename(columns={'timestamp':'Time Stamp'}, inplace=True)
    data['Time Stamp'] = pd.to_datetime(data['Time Stamp'])
    data['date'] = pd.to_datetime(data['Time Stamp']).dt.date
    # print(date1,type(date1))
    # date1 = pd.to_datetime(date1).dt.date
    # date2 = pd.to_dateime(date2).dt.date
    mask = (data['date'] > date1) & (data['date'] <= date2)
    #df = data.loc[data.timestamp==date1:data.timestamp==date2,:]
    df = data.loc[mask]
    print(df.columns)
    df.reset_index(inplace=True)
    # Adding percent growth of SPXL and SPXS over each minute
    df['SPXL_growth'] = df.SPXL.pct_change() * 100
    df['SPXS_growth'] = df.SPXS.pct_change() * 100

    # Calculating SMA and Bollinger Bands for SPXL
    df['SPXL_upper2'] = df.SPXL_growth.rolling(20).mean() + (df.SPXL_growth.rolling(20).std() * 2)
    df['SPXL_upper1'] = df.SPXL_growth.rolling(20).mean() + df.SPXL_growth.rolling(20).std()
    df['SPXL_sma'] = df.SPXL_growth.rolling(20).mean()

    # Calculating SMA and Bollinger Bands for SPXS
    df['SPXS_upper2'] = df.SPXS_growth.rolling(20).mean() + (df.SPXS_growth.rolling(20).std() * 2)
    df['SPXS_upper1'] = df.SPXS_growth.rolling(20).mean() + df.SPXS_growth.rolling(20).std()
    df['SPXS_sma'] = df.SPXS_growth.rolling(20).mean()

    # STATS
    print("\n~~ SPXL ~~")
    print("--------------------")
    print("\nMax SPXL Growth: ", df.SPXL_growth.max().round(3))
    print("Min SPXL Growth: ", df.SPXL_growth.min().round(3))
    print("\nMean SPXL Growth: ",df.SPXL_growth.mean().round(3))
    print("Median SPXL Growth: ",df.SPXL_growth.median().round(3))

    print('\n~~ SPXS ~~')
    print("--------------------")
    print("\nMax SPXS Growth: ", df.SPXS_growth.max().round(3))
    print("Min SPXS Growth: ", df.SPXS_growth.min().round(3))
    print("\nMean SPXS Growth: ",df.SPXS_growth.mean().round(3))
    print("Median SPXS Growth: ",df.SPXS_growth.median().round(3))

    # Starting shares
    df['SPXS_shares'] = 1.000
    df['SPXL_shares'] = 1.000

    df['SPXS MV'] = df.SPXS_shares * df.SPXS
    df['SPXL MV'] = df.SPXL_shares * df.SPXL

    df.loc[:,'num_of_trades'] = 0

    # COMPUTATION
    # For every row in range of dataframe rows (idx1 = starts at first row)
    for idx1 in range(len(df)-1):
        #trades = 0
        # Get the row directly after idx1
        idx2 = idx1 + 1
        # Price of SPXL at time 1
        SPXL1 = df.SPXL[idx1]
        # Price of SPXL at time 2
        SPXL2 = df.SPXL[idx2]

        # Price of SPXL at time 1
        SPXS1 = df.SPXS[idx1]
        # Price of SPXL at time 2
        SPXS2 = df.SPXS[idx2]

        # If SPXL % growth is greater or equal to the upper bound of SPXL % growth then sell SPXL and buy SPXS
        if df.SPXL_growth[idx2] >= df.SPXL_upper1[idx2]:
            # Sell the difference between SPXL price at time 2 and SPXL price at time 1
            sell = SPXL2 - SPXL1
            # New SPXL share count at time 2 is (SPXL price at time 1 divided by SPXL price at time 2) multiplied by our SPXL share count at time 1
            # Example: If SPXL is $5 per share and goes to $8 per share and we own 1 share, our new shares after selling $3 or 0.37 shares is:
            # new shares owned of SPXL $5/$8 * 1 share = 0.62 * 1 share = 0.62 shares
            df.loc[idx2:, 'SPXL_shares'] = (SPXL1 / SPXL2) * df.SPXL_shares[idx1]
            # Total SPXL at time 2 is SPXL shares owned at time 2 multiplied by SPXL price at time 2
            df.loc[idx2:,'SPXL MV'] = df.SPXL_shares[idx2] * SPXL2
            #df7['SPXL MV'][idx1] + (df7.SPXL_shares[idx2] * SPXL2)
            plus_shares = sell / SPXS2
            df.loc[idx2:,'SPXS_shares'] = df.SPXS_shares[idx1] + plus_shares
            #df7.loc[idx2:,'SPXL MV'] = df7['SPXL MV'][idx1] + (df7.SPXL_shares[idx2] * SPXL2)
            df.loc[idx2:,'SPXS MV'] = df.SPXS_shares[idx2] * SPXS2
            #df7['SPXS MV'][idx1] + (SPXS2 * plus_shares)
            df.loc[idx2:,'num_of_trades'] = df.num_of_trades[idx1] + 1
            df = df.copy()
        elif df.SPXS_growth[idx2] >= df.SPXS_upper2[idx2]:
            # Sell the difference between SPXL price at time 2 and SPXL price at time 1
            sell = SPXS2 - SPXS1
            df.loc[idx2:,'SPXS_shares'] = (SPXS1 / SPXS2) * df.SPXS_shares[idx1]
            df.loc[idx2:,'SPXS MV'] = df.SPXS_shares[idx2] * SPXS2
            plus_shares = sell / SPXL2
            #I dont know why it works but it works
            # plus_shares = sell / SPXS2
            df.loc[idx2:,'SPXL_shares'] = df.SPXL_shares[idx1] + plus_shares
            df.loc[idx2:,'SPXL MV'] = df.SPXL_shares[idx2] * SPXL2
            df.loc[idx2:,'num_of_trades'] = df.num_of_trades[idx1] + 1
            df = df.copy() 
    df = df.drop(columns = {'index','Unnamed: 0'})
    df['Total MV'] = df['SPXL MV'].values + df['SPXS MV'].values
    return df

def stats(df):
    # Beginning investment value is beginning spxl mv + beginning spxs mv
    beg_investment = df['SPXL MV'].head(1).values + df['SPXS MV'].head(1).values
    # Ending investment value is ending spxl mv + ending spxs mv
    end_indvestment = df['SPXS MV'].tail(1).values + df['SPXL MV'].tail(1).values
    # total profit is ending investment value - beginning investment value
    total_profit = end_indvestment - beg_investment
    # Profit margin is total profit divided by final sale of shares at mv
    profit_margin = (total_profit / end_indvestment) * 100
    # finds the total number of trades by looking at the last value in the 'num_of_trades' column of df
    tot_num_trades = df.num_of_trades.tail(1).values 
    
    # Displaying results on gui
    trades = "Total Number of Trades: " + str(round(tot_num_trades[0],2)) 
    end_inv_value = "Ending MV: $" + str(round(end_indvestment[0],2)) 
    beg_inv_value = "Beginning MV: $" + str(round(beg_investment[0],2)) 
    tot_prof = "Total Profit: $" + str(round(total_profit[0],2)) 
    pm = "Profit Margin: " + str(round(profit_margin[0],2)) + "%"
    return st.markdown(trades),st.markdown(end_inv_value),st.markdown(beg_inv_value),st.markdown(tot_prof),st.markdown(pm)

def plot(df2, options,t):
    # Create an empty figure instance
    fig = go.Figure()
    # If the multi-select option is a single option, it won't be in a list type so it doesn't need to go into a forloop
    if type(options) == str:
        fig.add_trace(go.Scatter(x = df2['Time Stamp'], y = df2.loc[:,options], name = options))
    # If the multi-select option is multiple options, it will be a list type object so it does need to go into a forloop
    else:
        for option in options:
            fig.add_trace(go.Scatter(x = df2['Time Stamp'], y = df2.loc[:,option], name = option))
    fig.update_layout(title = t)
    return st.plotly_chart(fig,use_container_width=True)
    
@st.cache_data
def convert_df(data):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return data.to_csv().encode('utf-8')

# Show df on GUI
with st.sidebar:
    date1 = st.date_input("Pick a date to start investing")
    date2 = st.date_input("Pick a date to end investing")

if (date1 != None) & (date2 != None) & (date1 != date2):
    # Create subset of data lake with the input dates
    df = subset(date1,date2)
    # Create a streamlit df object on gui
    st.dataframe(df)
    # Convert df to csv by calling convert_df function
    csv = convert_df(df)
    # Create download button on gui
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='trade_data.csv',
        mime='text/csv',
    )
    # Add results to sidebar
    with st.sidebar:
        st.subheader('Results')
        stats(df)

    #Get price graph
    plot(df,['SPXL','SPXS'],'Price Per Share')

    # Get shares graph
    plot(df,['SPXL_shares','SPXS_shares'],"Shares Over Time")

    # Multi-select button
    iv_options = st.multiselect("Graph MV",['Total MV', 'SPXL MV', 'SPXS MV'],default = 'Total MV')
    
    # Get MV graph
    plot(df,iv_options, "Total MV Over Time")

    

# df_plt = df.tail(100)
# #.set_index('timestamp')
# df_plt.SPXL_upper2.plot(label='SPXL 2 STD Upper Bound')
# df_plt.SPXL_upper1.plot(label='SPXL 1 STD Upper Bound')
# df_plt.SPXL_sma.plot(label='SPXL SMA')
# df_plt.SPXL_growth.plot(label='SPXL Growth')

# df_plt.SPXS_upper2.plot(label='SPXS 2 STD Upper Bound')
# df_plt.SPXS_upper1.plot(label='SPXS 1 STD Upper Bound')
# df_plt.SPXS_sma.plot(label='SPXS SMA')
# df_plt.SPXS_growth.plot(label='SPXS Growth')

# plt.legend()