import yfinance as yf
import pandas as pd
from textblob import TextBlob
import numpy as np
import re

def download_finance_data(ticker, start_date, end_date):
    finance_data = yf.download(ticker, start=start_date, end=end_date)
    return finance_data;

def financial_article_cleaner(file_name):
    article_sentiments = pd.read_pickle(file_name) 
    article_sentiments_company = article_sentiments.copy()
    article_sentiments_company['body_text'] = article_sentiments_company['body_text'].astype(str) + '---newarticle---'
    company_bodytext = article_sentiments_company['body_text']
    pd.set_option("display.max_colwidth", -1)
    
    with open('company_bodytext.txt', 'w', encoding="utf-8") as f:
        f.write(
            company_bodytext.to_string(header = False, index = False)
        )
    
    with open('company_bodytext.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        
    lines = [line.replace(' ', '') for line in lines]
    
    with open('company_bodytext.txt', 'w', encoding="utf-8") as f:
        f.writelines(lines)
    
    a_file = open("company_bodytext.txt", "r", encoding="utf-8")
    
    string_without_line_breaks = ""
    
    for line in a_file:
        stripped_line = line.rstrip() 
        string_without_line_breaks += stripped_line
    a_file.close()
    
    with open('company_bodytext.txt', 'w', encoding="utf-8") as f:
        f.writelines(string_without_line_breaks)
        
    
def financial_article_cleaner_v2(file_name, ticker, start_date, end_date):
    news_df = pd.read_pickle(file_name)
    news_df_new = news_df.copy()
    news_df_new = news_df_new.replace(to_replace='None', value=np.nan).dropna()
    news_df_new.drop_duplicates(subset ="title", 
                        keep = 'first', inplace = True)
    news_df_new['Date'] = pd.to_datetime(news_df_new.publish_date)
    news_df_new.set_index('Date', inplace=True)
    news_df_new = news_df_new.sort_index()

    news_df_combined = news_df_new.copy()
    news_df_combined['news_combined'] = news_df_combined.groupby(['publish_date'])['body_text'].transform(lambda x: ' '.join(x))
    news_df_combined.drop_duplicates(subset ="publish_date", 
                        keep = 'first', inplace = True)
    news_df_combined['Date'] = pd.to_datetime(news_df_combined.publish_date)
    news_df_combined.set_index('Date', inplace=True)

    stock_df = download_finance_data(ticker, start_date, end_date)

    merge = stock_df.merge(news_df_combined, how='inner', left_index=True, right_index=True)

    clean_news = []

    for i in range(0, len(merge["news_combined"])): 
        clean_news.append(re.sub("\n", ' ', merge["news_combined"][i])) 
        clean_news[i] = re.sub(r'[^\w\d\s\']+', '', clean_news[i]) 

    merge['news_cleaned'] = clean_news
    merge['news_cleaned'][0]
    merge['subjectivity'] = merge['news_cleaned'].apply(getSubjectivity)
    merge['polarity'] = merge['news_cleaned'].apply(getPolarity)

    stock_df_label = stock_df.copy()
    stock_df_label['Adj Close Next'] = stock_df_label['Adj Close'].shift(-1)
    stock_df_label['Label'] = stock_df_label.apply(lambda x: 1 if (x['Adj Close Next']>= x['Adj Close']) else 0, axis =1)
    stock_df_label[['Adj Close', 'Adj Close Next', 'Label']].head(5)
    
    stock_df_label_adj_nxt = stock_df_label[['Adj Close Next', 'Label']]
    stock_df_label_adj_nxt = stock_df_label_adj_nxt.dropna()

    merge2 = stock_df.merge(stock_df_label_adj_nxt, how='inner', left_index=True, right_index=True)
    merge2 = merge2.dropna()

    merge3 = stock_df_label_adj_nxt.merge(merge, how='inner', left_index=True, right_index=True)

    keep_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'subjectivity', 'polarity', 'compound', 'neg',	'neu',	'pos', 'Label']
    df =  merge3[keep_columns]
    return df
    
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df = financial_article_cleaner_v2('azn_article_sentiments_20220602.pkl', 'AZN.L', "2017-05-02", "2022-01-05")
print(df.head())