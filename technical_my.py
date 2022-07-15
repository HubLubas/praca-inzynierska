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
print(ftse100_stocks["Adj Close"].rolling(window=20).mean())

#Simple Moving Average
def sma(win_wide, start_date, end_date):
    sma = ftse100_stocks["Adj Close"].loc[start_date:end_date].rolling(window=win_wide).mean()
    return sma;

print('SMA')
print(sma(20, '2019-01-01','2019-12-31'))

#Expotential Moving Average 
def esma(win_wide, start_date, end_date):
    sma = ftse100_stocks["Adj Close"].loc[start_date:end_date].ewm(win_wide).mean()
    return sma;

print('ESMA')
print(esma(20, '2019-01-01','2019-12-31'))
