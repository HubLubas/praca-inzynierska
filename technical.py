import yfinance as yf

import numpy as np
import pandas as pd

import datetime
from datetime import date, timedelta

import warnings


def get_prices():
    tse100_stocks = yf.download("AZN.L", start=datetime.datetime(2010, 1, 1), 
                                     end=datetime.datetime(2019, 12, 31), group_by='tickers')
    return tse100_stocks.to_json(orient="records")