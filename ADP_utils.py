
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,confusion_matrix

import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
from imblearn.pipeline import make_pipeline
import pickle
import plotly.graph_objects as go

import plotly.express as px
from collections import Counter
np.random.seed(23)
SEED=23


#======================    Functions For Cleaning    ==============================
def row_missing_value(df,value=100):
    print('Total ROW numbers Originally...........................:',df.shape[0])
    print('Total ROW numbers After dropping missing rows under {}:'.format(value),
          df[df.isnull().sum(axis=1)<=value].shape[0])
    a = df[df.isnull().sum(axis=1)>value].index
    df.drop(a, inplace=True)
    


def N_unique(df):
    # The number of Unique values of columns
    N_unique = {}
    for col in df.columns.tolist():
        N_unique[col] = df[col].nunique()
    return N_unique


def report_missing_values(df, value = 27):
    # Missing and Type of columns
    table = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'}) 
    table = table.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    table = table.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0: 'null values (%)'}))
    table = table.T
    table['level'] = table['null values (%)'].apply(lambda x:'Zero' if x==0 else 'High' if x>=value else 'Low' )   
    table['N_unique'] = N_unique(df).values()
    table.sort_values(by=['level','null values (nb)'], ascending=[True,False],inplace=True)
    return table 


    
def drop(df,drop_list):
    df.drop(drop_list, axis=1, inplace=True)
    

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    

    
def Best_GS(model):
    '''
    Print bestscore_ and best_param_ after GridSearchCV
    Example:
    >>>printS(clf_Train)
    ============================================================
    For LGBMClassifier() model Best Score: 0.809740807398293
    ============================================================
    Best Params: 
     {'class_weight': 'balanced', 'learn_rate': 0.01, 
     ...'max_depth': 8, 'max_features': 25}
    
    '''
   
    print('='*60)
    print("For {} model \nBest Score: {}".format(model.estimator, model.best_score_))
    print('='*60)
    print("Best Params: \n", model.best_params_)
    
def Eval_Results(model, X_train, y_train, X_test, y_test):
    # Fitting
    model.fit(X_train, y_train)
    
    # Evaluating
    roc = roc_auc_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))
    print('ROC AUC Score:', roc)
    print('Confusion Matrix:\n', cm)
    return (roc, cm)

def map_out(df,col):
    
    df.loc[df[col] > 1, [col]]  = 1
    
# Encodes values to 1 that are greater than 1 for given columns
def mat_convert_pages(df,col_3,col_5):
    '''
    Input: 
        {df} : Dataframe to be modified
        {col_1,col_3,col_5} : Columns to encode to value 1
    Output:
        None
    '''
    df.loc[df[col_5] > 0, [col_3]]  = 0
    df.rename(columns={'Session_3plus_pages': 'Session_3_to_5_pages'}, inplace=True)
    
    
def mat_convert_minute(df,col_1,col_3,col_5):
    '''
    Input: 
        {df} : Dataframe to be modified
        {col_1,col_3,col_5} : Calculation to adjust Session Minutes columns
    Output:
        None
    '''
    df.loc[df[col_5] > 0, [col_1,col_3]]  = 0
    df.loc[df[col_3] > 0, [col_1]]       = 0
    df.rename(columns={'Session_1plus_minute': 'Session_1_to_3_minute',
                       'Session_3plus_minutes': 'Session_3_to_5_minutes'}, inplace=True)
    
    
def corr_drop(df,val=0.8):
    # correlation matrix    
    corr = df.corr().abs()
    upper_limit = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # find features to remove 
    drop_columns = [column for column in upper_limit.columns if any(upper_limit[column] > val)]
    return drop_columns

def reduce_component(df, n=3):
    pca = PCA(n_components=n)
    pca.fit(df)
    reduced_data = pca.transform(df)
    reduced_data = pd.DataFrame(reduced_data)
    print(pca.explained_variance_ratio_.sum())
    return reduced_data

def Eval_Results2(model, X_train, y_train, X_test, y_test):
    # Fitting
    model.fit(X_train, y_train)
    
    # Evaluating
    roc = roc_auc_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    return (roc, cm)
    