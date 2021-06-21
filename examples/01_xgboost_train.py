import xgboost as xgb 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import warnings
from xgboost import XGBClassifier
import pickle

# dump to pickle
def write2file(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# max_depth = 3, 학습률은 0.1, 예제가 이진분류이므로 목적함수(objective)는 binary:logistic(이진 로지스틱)
# 부스팅 반복횟수는 400
xgb_wrapper = XGBClassifier(n_estimators = 400, learning_rate = 0.1 , max_depth = 3)
evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds = 100, 
                eval_metric="logloss", eval_set = evals, verbose=True)
w_preds = xgb_wrapper.predict(X_test)

# dump mode data to pickle
write2file(xgb_wrapper, 'best_xgboost_model.pkl')
