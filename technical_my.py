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
print('Buy and sell MACD')
print(a)