from urllib import response
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
#import data_manager as dm 
import torch
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

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

def bert_prediction(article_sentiments, model, word_tokens):  
    
    articles = article_sentiments.filtered_articles_joined.values
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 
    tokenizer.add_tokens(word_tokens)
    
    max_len = 0

    for article in article_sentiments.filtered_articles_joined:

        input_ids = tokenizer.encode(article, add_special_tokens=True)

        max_len = max(max_len, len(input_ids))

    token_lens = []

    for txt in article_sentiments.filtered_articles_joined:
        tokens = tokenizer.encode(txt, max_length=7792)
        token_lens.append(len(tokens))
        
    article_sentiments['filtered_articles_split'] = article_sentiments['filtered_articles_joined'].apply(get_split)
    
    keep_columns = ['filtered_articles_split']
    df =  article_sentiments[keep_columns]
    
    articles = df.filtered_articles_split.values

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

    batch_size = 20  

    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_data_loader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    model.eval()

    predictions= []

    for batch in prediction_data_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)
        flat_predictions = []
        response = []
        for i in range(len(predictions)):
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
            flat_predictions.append(pred_labels_i)
        
        response = np.concatenate(flat_predictions, axis=0)
    return response

#article_sentimets = pd.read_pickle('azn_article_sentiments_20220602.pkl')
#data_vader = dm.financial_article_cleaner_v3('azn_article_sentiments_20220602.pkl', 'AZN.L', '2021-05-15', '2022-06-02')
#bert_data = dm.bert_data_v2(data_vader);
#bert_data = bert_data
#print(bert_data)
#bert_data.drop("Label", axis=1, inplace=True)
#print('######################')
#print(bert_data)
#bert_data.to_pickle('bert_data2.pkl')
bert_data = pd.read_pickle('bert_data2.pkl')
model = torch.load('bert_model')
print(bert_prediction(bert_data, model, ['astrazeneca', 'pfizer']))