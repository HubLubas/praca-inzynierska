import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
import seaborn as sns

import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc

from scipy import stats
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

import datetime
from datetime import date, timedelta

import warnings
warnings.filterwarnings('ignore')

ftse100_stocks = yf.download("AZN.L", start=datetime.datetime(2010, 1, 1), 
                                     end=datetime.datetime(2019, 12, 31), group_by='tickers')
#print(ftse100_stocks["Adj Close"].rolling(window=20).mean())

#Simple Moving Average
def sma(win_wide, start_date, end_date):
    sma = ftse100_stocks["Adj Close"].loc[start_date:end_date].rolling(window=win_wide).mean()
    return sma;

#print('SMA')
#print(sma(20, '2019-01-01','2019-12-31'))

#Expotential Moving Average 
def esma(win_wide, start_date, end_date):
    sma = ftse100_stocks["Adj Close"].loc[start_date:end_date].ewm(win_wide).mean()
    return sma;

#print('ESMA')
#print(esma(20, '2019-01-01','2019-12-31'))

#Triple Moving Average Crossover Strategy

def triple_macs(start_date, end_date):
    azn_adj_6mo = ftse100_stocks[['Adj Close']][start_date:end_date]
    ShortEMA = azn_adj_6mo['Adj Close'].ewm(span=5, adjust=False).mean()
    MiddleEMA = azn_adj_6mo['Adj Close'].ewm(span=21, adjust=False).mean()
    LongEMA = azn_adj_6mo['Adj Close'].ewm(span=63, adjust=False).mean() 
    
    return (ShortEMA, MiddleEMA, LongEMA, azn_adj_6mo)

def buy_sell_triple_macs(ShortEMA, MiddleEMA, LongEMA, data):
    data['Short'] = ShortEMA
    data['Middle'] = MiddleEMA
    data['Long'] = LongEMA  
    
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Adj Close'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell_list.append(data['Adj Close'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)
    
    return (buy_list, sell_list)

#print('Buy and sale')
(ShortEMA, MiddleEMA, LongEMA, azn_adj_6mo) = triple_macs('2019-05-01','2019-10-31')
#print(buy_sell_triple_macs(ShortEMA, MiddleEMA, LongEMA, azn_adj_6mo))
azn_adj_6mo['Buy'] = buy_sell_triple_macs(ShortEMA, MiddleEMA, LongEMA, azn_adj_6mo)[0]
azn_adj_6mo['Sell'] = buy_sell_triple_macs(ShortEMA, MiddleEMA, LongEMA, azn_adj_6mo)[1]
#print(azn_adj_6mo['Buy'])
#print(azn_adj_6mo['Sell'])

def macd(start_date, end_date):
    azn_adj_3mo = ftse100_stocks[['Adj Close']][start_date:end_date]
    ShortEMA = azn_adj_3mo['Adj Close'].ewm(span=12, adjust=False).mean()
    LongEMA = azn_adj_3mo['Adj Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    azn_adj_3mo['MACD'] = MACD
    azn_adj_3mo['Signal Line'] = signal
    azn_adj_3mo
    return azn_adj_3mo

def buy_sell_macd(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Adj Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Adj Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return (Buy, Sell)

azn_adj_3mo = macd('2019-08-01','2019-10-31');   
a = buy_sell_macd(azn_adj_3mo) 
#print('Buy and sell MACD')
#print(a)

#RSI 

def rsi(start_date, end_date):
    azn_adj_12mo = ftse100_stocks[['Adj Close']][start_date:end_date]
    delta = azn_adj_12mo['Adj Close'].diff(1)
    delta = delta.dropna()
    
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    period = 14
    # Calculate average gain and average loss
    AVG_Gain = up.rolling(window=period).mean()
    #AVG_Loss = abs(down.rolling(window=period).mean())
    AVG_Loss = down.abs().rolling(window=period).mean()
    # Calculate Relative Strength (RS)
    RS = AVG_Gain / AVG_Loss
    # Calculate RSI
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    
    # Calculate the EWMA average gain and average loss
    AVG_Gain2 = up.ewm(span=period).mean()
    AVG_Loss2 = down.abs().ewm(span=period).mean()

    # Calculate the RSI based on EWMA
    RS2 = AVG_Gain2 / AVG_Loss2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    
    return (RSI, RSI2)
    
(RSI, RSI2)  = rsi('2019-01-01','2019-12-31')
#print('RSI')
#print(RSI)
#print('RSI2')
#print(RSI2)

#Rate of change
def roc(start_date, end_date):
    azn_roc_12mo = ftse100_stocks[start_date:end_date]
    azn_roc_12mo['ROC'] = ( azn_roc_12mo['Adj Close'] / azn_roc_12mo['Adj Close'].shift(9) -1 ) * 100
    
    azn_roc_100d = azn_roc_12mo[-100:]
    dates = azn_roc_100d.index
    price = azn_roc_100d['Adj Close']
    roc = azn_roc_100d['ROC']

    return roc

#print('ROC')
#print(roc('2019-01-01','2019-12-31'))

#Bollinder Bands
def bollinder_bands(start_date, end_date):
    azn_12mo_bb = ftse100_stocks[start_date:end_date]
    #Get the time period (20 days)
    period = 20
    # Calculate the 20 Day Simple Moving Average, Std Deviation, Upper Band and Lower Band
    #Calculating the Simple Moving Average
    azn_12mo_bb['SMA'] = azn_12mo_bb['Close'].rolling(window=period).mean()
    # Get the standard deviation
    azn_12mo_bb['STD'] = azn_12mo_bb['Close'].rolling(window=period).std()
    #Calculate the Upper Bollinger Band
    azn_12mo_bb['Upper'] = azn_12mo_bb['SMA'] + (azn_12mo_bb['STD'] * 2)
    #Calculate the Lower Bollinger Band
    azn_12mo_bb['Lower'] = azn_12mo_bb['SMA'] - (azn_12mo_bb['STD'] * 2)
    #Create a list of columns to keep
    #column_list = ['Close', 'SMA', 'Upper', 'Lower']
    return azn_12mo_bb

def buy_sell_bb(data):
    buy_signal = [] #buy list
    sell_signal = [] #sell list

    for i in range(len(data['Close'])):
      if data['Close'][i] > data['Upper'][i]: #Then you should sell 
        buy_signal.append(np.nan)
        sell_signal.append(data['Close'][i])
      elif data['Close'][i] < data['Lower'][i]: #Then you should buy
        sell_signal.append(np.nan)
        buy_signal.append(data['Close'][i])
      else:
        buy_signal.append(np.nan)
        sell_signal.append(np.nan)
    return (buy_signal, sell_signal)

data_bb = bollinder_bands('2019-01-01','2019-12-31')
(buy_bb, sell_bb) = buy_sell_bb(data_bb)
#print(buy_bb)
#print(sell_bb)

def so(start_date, end_date):
    azn_so = ftse100_stocks[start_date:end_date]
    azn_so['L14'] = azn_so['Low'].rolling(window=14).min()
    azn_so['H14'] = azn_so['High'].rolling(window=14).max()

    azn_so['%K'] = 100*((azn_so['Close'] - azn_so['L14']) / (azn_so['H14'] - azn_so['L14']) )
    azn_so['%D'] = azn_so['%K'].rolling(window=3).mean()

    azn_so['Sell Entry'] = ((azn_so['%K'] < azn_so['%D']) & (azn_so['%K'].shift(1) > azn_so['%D'].shift(1))) & (azn_so['%D'] > 80)
    azn_so['Buy Entry'] = ((azn_so['%K'] > azn_so['%D']) & (azn_so['%K'].shift(1) < azn_so['%D'].shift(1))) & (azn_so['%D'] < 20)
    return (azn_so)
