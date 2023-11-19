import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

import xgboost as xgb
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook

tqdm.pandas()

#reading data
path = 'data'

mcc_codes = pd.read_csv(os.path.join(path, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
trans_types = pd.read_csv(os.path.join(path, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions = pd.read_csv(os.path.join(path, 'transactions.csv'), index_col='client_id')
gender_train = pd.read_csv(os.path.join(path, 'train.csv'), index_col='client_id')
gender_test = pd.read_csv(os.path.join(path, 'test.csv'), index_col='client_id')
transactions_train = transactions.join(gender_train, how='inner')
transactions_test = transactions.join(gender_test, how='inner')

params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    "device": "cuda",
    
    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,
    
    'eval_metric': 'auc',
    'objective': 'binary:logistic' ,
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx',
}

def basic_features(df): 
    features = []
    features.append(pd.Series(df[df['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('positive_transactions_')))
    features.append(pd.Series(df[df['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('negative_transactions_')))
 
    return pd.concat(features)

data_train = transactions_train.groupby(transactions_train.index).progress_apply(basic_features)
data_test = transactions_test.groupby(transactions_test.index).progress_apply(basic_features)

for df in tqdm([transactions_train, transactions_test]):
    df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = ~df['hour'].between(6, 22).astype(int)

def big_ballz_features(x): 
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix('negative_transactions_')))
    features.append(pd.Series(x['trans_type'].value_counts(normalize=True).add_prefix('trans_type_')))
    features.append(pd.Series(x['mcc_code'].value_counts(normalize=True).add_prefix('mcc_code_')))
 
    return pd.concat(features)

data_train = transactions_train.groupby(transactions_train.index)\
                               .progress_apply(big_ballz_features).unstack(-1)
data_test = transactions_test.groupby(transactions_test.index)\
                             .progress_apply(big_ballz_features).unstack(-1)

names = list(set(data_train.columns) - set(data_test.columns))

data_test = data_test.reindex(columns=data_test.columns.tolist() + list(names))

target = data_train.join(gender_train, how='inner')['gender']

res = xgb.cv(params, xgb.DMatrix(data_train, target),
                  early_stopping_rounds=10, maximize=True, 
                  num_boost_round=10000, nfold=5, stratified=True)

russian_most_wanted = res['test-auc-mean'].argmax()
print('ROC_AUC: ', res.loc[russian_most_wanted]['test-auc-mean'], '+-', 
      res.loc[russian_most_wanted]['test-auc-std'], 'trees: ', russian_most_wanted)

model = xgb.train(params, xgb.DMatrix(data_train.values, target, feature_names=list(data_train.columns)), 
                    num_boost_round=russian_most_wanted, maximize=True)

y_pred = model.predict(xgb.DMatrix(data_test.values, feature_names=list(data_train.columns)))

y_pred = list(map(round, y_pred))

submission = pd.DataFrame(index=data_test.index, data=y_pred, columns=['probability'])

submission.to_csv('result.csv')