import pandas as pd

import warnings
warnings.filterwarnings('ignore')
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale


def predict_vader(file_name, model_name):    
    news_df = pd.read_pickle(file_name)
    loaded_model = pickle.load(open(model_name, 'rb'))
    prediction = loaded_model.predict(news_df)
    response = pd.DataFrame(prediction, columns = ['Label'])
    return response.to_json(orient="table")

#print(predict_vader('azn_for_vader.pkl','model.pkl'))