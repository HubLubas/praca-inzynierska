from regex import E
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import numpy as np
import re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

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
    #merge['news_cleaned'][0]
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

def financial_article_cleaner_v3(file_name, ticker, start_date, end_date):
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
    #merge['news_cleaned'][0]
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
    return merge3 
    
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def bert_data_tokenizer(article_sentiments, tokenizer):
    articles = article_sentiments.news_cleaned
    labels = article_sentiments.Label
    tokens = [word_tokenize(article) for article in articles]
    text = nltk.Text(tokens)
    stop_words = set(stopwords.words('english'))
    
    filtered_articles = [[word for word in article if not word in stop_words if word.isalpha()] for article in tokens]
    filtered_articles_joined = [','.join(article).replace(',', ' ') for article in filtered_articles]
    
    article_sentiments['filtered_articles_joined'] = filtered_articles_joined
    
    articles = article_sentiments.filtered_articles_joined
    labels = article_sentiments.Label
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.add_tokens(tokens)
    max_len = 0
    
    for article in article_sentiments.filtered_articles_joined:
    
        input_ids = tokenizer.encode(article, add_special_tokens=True)

        max_len = max(max_len, len(input_ids))
        
    token_lens = []

    for txt in article_sentiments.filtered_articles_joined:
        tokens = tokenizer.encode(txt, max_length=7792)
        token_lens.append(len(tokens))
        
    article_sentiments['filtered_articles_split'] = article_sentiments['filtered_articles_joined']
    keep_columns = ['filtered_articles_split', 'Label']
    df =  article_sentiments[keep_columns]
    
    articles = df.filtered_articles_split
    labels = df.Label

    input_ids = []
    attention_masks = []
    for article in articles:
        encoded_dict = tokenizer.encode_plus(
                            article,                  
                            add_special_tokens = True, 
                            max_length = 200,           
                            return_token_type_ids=False,
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',    
                    )
        
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    batch_size = 16

    data_loader = DataLoader(df, batch_size, sampler=RandomSampler(df), num_workers=4)
    
    return data_loader   

def bert_data_v2(article_sentiments):
    articles = article_sentiments.news_cleaned.values
    labels = article_sentiments.Label.values
    
    tokens = [word_tokenize(article) for article in articles]
    text = nltk.Text(tokens)
    
    stop_words = set(stopwords.words('english'))
    filtered_articles = [[word for word in article if not word in stop_words if word.isalpha()] for article in tokens]
    filtered_articles_joined = [','.join(article).replace(',', ' ') for article in filtered_articles]
    article_sentiments['filtered_articles_joined'] = filtered_articles_joined
    
    return article_sentiments
    
def get_split(text1):
    l_total = []
    l_partial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text1.split()[:200]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_partial))
    return l_total  

data = pd.read_pickle('azn_prices_labels_news_20210107.pkl')  
end = bert_data_tokenizer(data, ['astrazeneca', 'pfizer'])

print('###################');