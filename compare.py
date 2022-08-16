import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import vader_predictor as vp
import bert_predictor as bp
import torch

def data_handler(date, period):
    new_date = (date + timedelta(days=period)).strftime('%Y-%m-%d')
    return new_date

def predict_profits(filename_vader, filename_bert, decision_date):
    date_desicion = datetime.strptime(decision_date, '%Y-%m-%d').date()
    
    data_end = yf.download('AZN.L', start=decision_date, end=datetime.today().strftime('%Y-%m-%d'))
    data_end = data_end['Close']
    
    
    is_trading_day = 0
    i = 1
    
    while(is_trading_day == 0):
        long_date = data_handler(date_desicion, 360 + i)
        #print(i)
        value = yf.download('AZN.L', start=long_date, end=long_date)['Close']
        #print(len(value))
        if(len(value) == 0):
            i = i + 1
        else: 
            is_trading_day = 1
            
    is_trading_day2 = 0
    j = 1
    while(is_trading_day2 == 0):
        medium_date = data_handler(date_desicion, 180 + j)
        #print(i)
        mvalue = yf.download('AZN.L', start=medium_date, end=medium_date)['Close']
        #print(len(value))
        if(len(mvalue) == 0):
            j = j + 1
        else: 
            is_trading_day2 = 1
            
    is_trading_day3 = 0
    k = 1
    while(is_trading_day3 == 0):
        short_date = data_handler(date_desicion, 30 + k)
        #print(short_date)
        svalue = yf.download('AZN.L', start=short_date, end=short_date)['Close']
        #print(len(svalue))
        if(len(svalue) == 0):
            k = k + 1
        else: 
            is_trading_day3 = 1
    

    long_profit =   value - data_end.loc[date_desicion.strftime('%Y-%m-%d')]
    medium_profit = mvalue  - data_end.loc[date_desicion.strftime('%Y-%m-%d')]
    short_profit = svalue  - data_end.loc[date_desicion.strftime('%Y-%m-%d')]
    
    model = torch.load('bert_model')
    bert_data = pd.read_pickle(filename_bert)
    vader_data = pd.read_pickle(filename_vader)
    
    bert_data = bert_data.loc[[decision_date]]
    vader_data = vader_data.loc[[decision_date]]
    
    vader_prediction = vp.predict_vader_v2(vader_data,'model.pkl')
    bert_prediction = bp.bert_prediction(bert_data, model, ['astrazeneca', 'pfizer'])

    return  (short_profit, medium_profit, long_profit, vader_prediction, bert_prediction)
(short_profit, medium_profit, long_profit, vader_prediction, bert_prediction) = predict_profits('vader_data5.pkl', 'bert_data_v3.pkl', '2018-05-16')
#print('#######################')
#print(bert_prediction)
#print('#######################')

#dv = pd.read_pickle('vader_data6.pkl')
#db = pd.read_pickle('bert_data_v3.pkl')
#dv = db.loc['2017-05-12':'2022-05-26']
#dv.to_pickle('vader_data6.pkl')
#print(dv)
#print(db)

