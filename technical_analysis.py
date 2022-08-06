from urllib import response
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


def sma(ticker, start_date, end_date, wide):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    sma = stocks["Adj Close"].loc[start_date:end_date].rolling(window=wide).mean()
    response = stocks[["Adj Close"]]
    response['SMA'] = sma
    return response.to_json(orient="table")

#print(sma('AZN.L', '2018-05-01','2019-10-31', 20))

def esma(ticker, start_date, end_date, wide):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    esma = stocks["Adj Close"].loc[start_date:end_date].ewm(wide).mean()
    response = stocks[["Adj Close"]]
    response['ESMA'] = esma
    return response.to_json(orient="table")

def triple_macs(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[['Adj Close']][start_date:end_date]
    ShortEMA = response['Adj Close'].ewm(span=5, adjust=False).mean()
    MiddleEMA = response['Adj Close'].ewm(span=21, adjust=False).mean()
    LongEMA = response['Adj Close'].ewm(span=63, adjust=False).mean() 
    
    response['Short'] = ShortEMA
    response['Middle'] = MiddleEMA
    response['Long'] = LongEMA 
    return response.to_json(orient="table")

#print(triple_macs('AZN.L', '2018-05-01','2019-10-31'))

def buy_sell_triple_macs(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[['Adj Close']][start_date:end_date]
    ShortEMA = response['Adj Close'].ewm(span=5, adjust=False).mean()
    MiddleEMA = response['Adj Close'].ewm(span=21, adjust=False).mean()
    LongEMA = response['Adj Close'].ewm(span=63, adjust=False).mean() 
    
    response['Short'] = ShortEMA
    response['Middle'] = MiddleEMA
    response['Long'] = LongEMA 
    
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False
    
    for i in range(0, len(response)):
        if response['Middle'][i] < response['Long'][i] and response['Short'][i] < response['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(response['Adj Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and response['Short'][i] > response['Middle'][i]:
            sell_list.append(response['Adj Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif response['Middle'][i] > response['Long'][i] and response['Short'][i] > response['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(response['Adj Close'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and response['Short'][i] < response['Middle'][i]:
            sell_list.append(response['Adj Close'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)
    
    response['Buy'] = buy_list
    response['Sell'] = sell_list
    
    return response.to_json(orient="table")    

#print(buy_sell_triple_macs('AZN.L', '2018-05-01','2019-10-31'))

def macd(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[['Adj Close']][start_date:end_date]
    ShortEMA = response['Adj Close'].ewm(span=12, adjust=False).mean()
    LongEMA = response['Adj Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    response['MACD'] = MACD
    response['ShortEMA'] = ShortEMA
    response['LongEMA'] = LongEMA
    response['Signal Line'] = signal
    
    return response.to_json(orient="table")   

#print(macd('AZN.L', '2018-05-01','2019-10-31'))

def buy_sell_macd(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[['Adj Close']][start_date:end_date]
    ShortEMA = response['Adj Close'].ewm(span=12, adjust=False).mean()
    LongEMA = response['Adj Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    response['MACD'] = MACD
    response['ShortEMA'] = ShortEMA
    response['LongEMA'] = LongEMA
    response['Signal Line'] = signal
    
    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(response)):
        if response['MACD'][i] > response['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(response['Adj Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif response['MACD'][i] < response['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(response['Adj Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
            
    response['Buy'] = Buy
    response['Sell'] = Sell
    
    return response.to_json(orient="table")    

#print(buy_sell_macd('AZN.L', '2018-05-01','2019-10-31'))   

def rsi(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[['Adj Close']][start_date:end_date]
    delta = response['Adj Close'].diff(1)
    delta = delta.dropna()
    
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    period = 14
    AVG_Gain = up.rolling(window=period).mean()
    AVG_Loss = down.abs().rolling(window=period).mean()
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    AVG_Gain2 = up.ewm(span=period).mean()
    AVG_Loss2 = down.abs().ewm(span=period).mean()

    # Calculate the RSI based on EWMA
    RS2 = AVG_Gain2 / AVG_Loss2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))     
    
    response['RS'] = RS
    response['RSI'] = RSI
    response['RS_EWMA'] = RS2
    response['RSI_EWMA'] = RSI2
    
    return response.to_json(orient="table")  

#print(rsi('AZN.L', '2018-05-01','2019-10-31'))

def roc(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')    
    response = stocks[start_date:end_date]
    response['ROC'] = ( response['Adj Close'] / response['Adj Close'].shift(9) -1 ) * 100
    
    roc_100d = response[-100:]
    dates = roc_100d.index
    price = roc_100d['Adj Close']
    roc = roc_100d['ROC']
    
    response['ROC'] = roc
    return response.to_json(orient="table")  

#print(roc('AZN.L', '2018-05-01','2019-10-31'))

def bollinder_bands(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[start_date:end_date]
    period = 20

    response['SMA'] = response['Close'].rolling(window=period).mean()
    response['STD'] = response['Close'].rolling(window=period).std()
    response['Upper'] = response['SMA'] + (response['STD'] * 2)
    response['Lower'] = response['SMA'] - (response['STD'] * 2)
    
    return response.to_json(orient="table")

#print(bollinder_bands('AZN.L', '2018-05-01','2019-10-31'))

def buy_sell_bollinder_bands(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[start_date:end_date]
    period = 20

    response['SMA'] = response['Close'].rolling(window=period).mean()
    response['STD'] = response['Close'].rolling(window=period).std()
    response['Upper'] = response['SMA'] + (response['STD'] * 2)
    response['Lower'] = response['SMA'] - (response['STD'] * 2)
    
    buy_signal = [] #buy list
    sell_signal = [] #sell list

    for i in range(len(response['Close'])):
      if response['Close'][i] > response['Upper'][i]: #Then you should sell 
        buy_signal.append(np.nan)
        sell_signal.append(response['Close'][i])
      elif response['Close'][i] < response['Lower'][i]: #Then you should buy
        sell_signal.append(np.nan)
        buy_signal.append(response['Close'][i])
      else:
        buy_signal.append(np.nan)
        sell_signal.append(np.nan)    

    response['Buy'] = buy_signal
    response['Sell'] = sell_signal
    
    return response.to_json(orient="table")

#print(buy_sell_bollinder_bands('AZN.L', '2018-05-01','2019-10-31'))

def so(ticker, start_date, end_date):
    stocks = yf.download(ticker, start=start_date, 
                                     end=end_date, group_by='tickers')
    response = stocks[start_date:end_date]
    response['L14'] = response['Low'].rolling(window=14).min()
    response['H14'] = response['High'].rolling(window=14).max()

    response['%K'] = 100*((response['Close'] - response['L14']) / (response['H14'] - response['L14']) )
    response['%D'] = response['%K'].rolling(window=3).mean()

    response['Sell Entry'] = ((response['%K'] < response['%D']) & (response['%K'].shift(1) > response['%D'].shift(1))) & (response['%D'] > 80)
    response['Buy Entry'] = ((response['%K'] > response['%D']) & (response['%K'].shift(1) < response['%D'].shift(1))) & (response['%D'] < 20)
    
    return response.to_json(orient="table")

#print(so('AZN.L', '2018-05-01','2019-10-31'))