import compare  as com
import pandas as pd
from datetime import datetime, timedelta


def check_series(filename_vader, filename_bert):
    bert_data = pd.read_pickle(filename_bert)
    #vader_data = pd.read_pickle(filename_vader) 
    
    copy_bert = bert_data 
    copy_bert = copy_bert.reset_index()
    
    #skócenie dla testu
    dates = copy_bert["Date"]
    dates = dates.iloc[0:5]
    #print(dates)
    
    for date in dates:
        (short_profit, medium_profit, long_profit, vader_prediction, bert_prediction) = com.predict_profits(filename_vader, filename_bert, date.strftime('%Y-%m-%d'))
        #print('############')
        #print(vader_prediction)
        #print('############')
        
        
        
    
check_series('vader_data5.pkl', 'bert_data_v3.pkl')