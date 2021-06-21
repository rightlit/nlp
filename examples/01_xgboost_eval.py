from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import warnings
from xgboost import XGBClassifier
import pickle

# 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)

    #print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    #print('정밀도: {:.4f}'.format(precision))
    #print('재현율: {:.4f}'.format(recall))
    #print('F1: {:.4f}'.format(F1))
    #print('AUC: {:.4f}'.format(AUC))

# load to data variable
def load2var(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data

X_train = load2var('X_train.pkl')
X_test = load2var('X_test.pkl')
y_train = load2var('y_train.pkl')
y_test = load2var('y_test.pkl')

xgb_model = load2var('best_xgboost_model.pkl')
w_preds = xgb_wrapper.predict(X_test)
# 예측 결과 확인
get_clf_eval(y_test, w_preds)
