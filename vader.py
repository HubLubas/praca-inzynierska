import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
import seaborn as sns
import math
import datetime
import re
import yfinance as yf
import nltk
import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
from textblob import TextBlob

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale


article_sentiments = pd.read_pickle('azn_article_sentiments_20220602.pkl') 

article_sentiments_azn = article_sentiments.copy()

article_sentiments_azn['body_text'] = article_sentiments_azn['body_text'].astype(str) + '---newarticle---'

azn_bodytext = article_sentiments_azn['body_text']

pd.set_option("display.max_colwidth", -1)

with open('azn_bodytext_20210105.txt', 'w', encoding="utf-8") as f:
    f.write(
        azn_bodytext.to_string(header = False, index = False)
    )
    
# first get all lines from file
with open('azn_bodytext_20210105.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()

# remove spaces
lines = [line.replace(' ', '') for line in lines]

# finally, write lines in the file
with open('azn_bodytext_20210105.txt', 'w', encoding="utf-8") as f:
    f.writelines(lines)

"""### Remove end line breaks from text file"""

# first get all lines from file
a_file = open("azn_bodytext_20210105.txt", "r", encoding="utf-8")

# create variable for string without line breaks
string_without_line_breaks = ""

# iterate over strings 
for line in a_file:
  stripped_line = line.rstrip() # rstrip() method removes any trailing characters - space is the default trailing character to remove
  string_without_line_breaks += stripped_line
a_file.close()

# finally, write lines in the file
with open('azn_bodytext_20210105.txt', 'w', encoding="utf-8") as f:
    f.writelines(string_without_line_breaks)

# Read article sentiments data into DataFrame 

azn_news_df = pd.read_pickle('azn_article_sentiments_20220602.pkl')

azn_news_df_new = azn_news_df.copy()

azn_news_df_new = azn_news_df_new.replace(to_replace='None', value=np.nan).dropna()

azn_news_df_new.drop_duplicates(subset ="title", 
                     keep = 'first', inplace = True)

azn_news_df_new['Date'] = pd.to_datetime(azn_news_df_new.publish_date)
azn_news_df_new.set_index('Date', inplace=True)

azn_news_df_new = azn_news_df_new.sort_index()

azn_news_df_new.to_pickle("azn_news_df_new_20210106.pkl")
azn_news_df_new.to_csv("azn_news_df_new_20210106.csv", sep=',', encoding='utf-8', header=True)

azn_news_df_combined = azn_news_df_new.copy()

azn_news_df_combined['news_combined'] = azn_news_df_combined.groupby(['publish_date'])['body_text'].transform(lambda x: ' '.join(x))

azn_news_df_combined.drop_duplicates(subset ="publish_date", 
                     keep = 'first', inplace = True)

azn_news_df_combined.to_csv("azn_news_df_combined_20210106.csv", sep=',', encoding='utf-8', header=True)

azn_news_df_combined = pd.read_csv("azn_news_df_combined_20210106.csv")

azn_news_df_combined['Date'] = pd.to_datetime(azn_news_df_combined.publish_date)
azn_news_df_combined.set_index('Date', inplace=True)

azn_stock_df = yf.download("AZN.L", start="2017-05-02", end="2022-01-05")

"""## 6. Merge Stock and Sentiment Dataframes on Date"""

# Merge data sets on date
merge = azn_stock_df.merge(azn_news_df_combined, how='inner', left_index=True, right_index=True)

# Save merged DataFrame

merge.to_csv("azn_news_stock_merge_20210107.csv", sep=',', encoding='utf-8', header=True)

# Show first row in combined news column

merge['news_combined'].iloc[0]

# Iterate over rows in combined news column

#for index, row in merge.iterrows(): 
    #print (row["news_combined"])

"""### Clean data in combined news column

Strip newline escape sequence (\n), unwanted punctuation and backslashes.  
"""

# Create empty list to append cleaned data from combined news column

clean_news = []

for i in range(0, len(merge["news_combined"])): 
    clean_news.append(re.sub("\n", ' ', merge["news_combined"][i]))  # replace n\ with ' '
    clean_news[i] = re.sub(r'[^\w\d\s\']+', '', clean_news[i]) # remove unwanted punctuation and \'


# Add cleaned news column to merged data set

merge['news_cleaned'] = clean_news

merge['news_cleaned'][0]

# Save merged DataFrame

merge.to_csv("azn__merge_cleaned_20210107.csv", sep=',', encoding='utf-8', header=True)


# Create function to get subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Create function to get polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# Create new columns
merge['subjectivity'] = merge['news_cleaned'].apply(getSubjectivity)
merge['polarity'] = merge['news_cleaned'].apply(getPolarity)



# Save DataFrame with subjectivity and polarity scores
merge.to_csv("azn__merge_cleaned_subj_pol_20210107.csv", sep=',', encoding='utf-8', header=True)


# Create copy of stock data

azn_stock_df_label = azn_stock_df.copy()

# "1" when AZN Adj Close value rose or stayed as the same;
# "0" when AZN Adj Close value decreased.

azn_stock_df_label['Adj Close Next'] = azn_stock_df_label['Adj Close'].shift(-1)
azn_stock_df_label['Label'] = azn_stock_df_label.apply(lambda x: 1 if (x['Adj Close Next']>= x['Adj Close']) else 0, axis =1)

azn_stock_df_label[['Adj Close', 'Adj Close Next', 'Label']].head(5)

# Save DataFrame

azn_stock_df_label.to_pickle("azn_stock_df_labels_20210107.pkl")

azn_stock_df_label.to_csv("azn_stock_df_label_20210107.csv", sep=',', encoding='utf-8', header=True)

# Show Adj Close Next and Label with Date

azn_stock_df_label_adj_nxt = azn_stock_df_label[['Adj Close Next', 'Label']]

# Drop NaN row

azn_stock_df_label_adj_nxt = azn_stock_df_label_adj_nxt.dropna()

# Merge DataFrames on date
merge2 = azn_stock_df.merge(azn_stock_df_label_adj_nxt, how='inner', left_index=True, right_index=True)

# Drop NaN row and show merged DataFrame
merge2 = merge2.dropna()

# Save DataFrame
merge2.to_csv("azn_prices_labels_20210107.csv", sep=',', encoding='utf-8', header=True)

merge2.to_pickle("azn_prices_labels_20210107.pkl")

# Merge next day Adjusted Close price and Label with combined stock data and sentiment DataFrame

merge3 = azn_stock_df_label_adj_nxt.merge(merge, how='inner', left_index=True, right_index=True)

# Save merged DataFrame

merge3.to_csv("azn_prices_labels_news_20210107.csv", sep=',', encoding='utf-8', header=True)

merge3.to_pickle("azn_prices_labels_news_20210107.pkl")

merge3 = pd.read_pickle("azn_prices_labels_news_20210107.pkl")

# Collapse data set to keep relevant stock price and sentiment score columns only

keep_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'subjectivity', 'polarity', 'compound', 'neg',	'neu',	'pos', 'Label']
df =  merge3[keep_columns]



# Create feature data set
X = df
X = np.array(X.drop(['Label'], 1))

# Create target data set
y = np.array(df['Label'])

"""We will split the data into train and test sets to verify predictions. Time series data cannot be split randomly as this would introduce look-ahead bias so the first 80% will be the training set and the last 20% the test set."""

# Split data into 80% training and 20% testing data sets

split = int(0.8*len(df))

X_train = X[0:split]
y_train = y[0:split]

X_test = X[split:]
y_test = y[split:]



# Create and train the model
model = LinearDiscriminantAnalysis().fit(X_train, y_train)

# Show model's predictions
predictions = model.predict(X_test)

"""### Feature scaling

We will standardise the data using scikit-learn's preprocessing.scale() algorithm so that it is all on one scale.
"""

# Standardise X's
X_train = scale(X_train)
X_test = scale(X_test)

"""### Create function for confusion matrix to visualise performance"""

# Function for confusion matrix

def plot_confusion_matrix(y_true, y_pred, labels=["Decrease", "Increase"], 
                          normalize=False, title=None, cmap=plt.cm.coolwarm):

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='ACTUAL',
           xlabel='PREDICTED')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="snow" if cm[i, j] > thresh else "orange",
                    size=26)
    ax.grid(False)
    fig.tight_layout()
    return ax

"""### Create dictionary of classifiers to train and predict on"""

# test models
models = {  'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis(),
            'SVM Classification': SVC(),
            'SGDClassifier': SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
            'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=10),
            'GaussianProcessClassifier': GaussianProcessClassifier(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
            }

for model_name in models.keys():

    model = models[model_name]
    print('\n'+'--------------',model_name,'---------------'+'\n')
    model.fit(X_train,y_train)
    print(model.predict(X_test))
    # Final Classification Report
    #print(classification_report(model.predict(X_test),y_test, target_names=['Decrease', 'Increase']))    