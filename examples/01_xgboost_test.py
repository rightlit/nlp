from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def get_clf_eval(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print('\n정확도: {:.4f}'.format(accuracy))

# load to data variable
def load2var(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data

def sentiment_predict(sentence):
  new_sentence = okt.morphs(sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  #print(pad_new)
  #score = float(loaded_model.predict(pad_new)) # 예측
  p_label = xgb_wrapper.predict(pad_new)
  score = float(p_label) # 예측
  if(score > 0.5):
    print("{} : 긍정 리뷰입니다.{}\n".format(sentence, p_label))
  else:
    print("{} : 부정 리뷰입니다.{}\n".format(sentence, p_label))

#vocab_size = 19416
max_len = 30
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

print('loading tokenizer...')
tokenizer = load2var('tokenizer_nsmc.pkl')

#print('loading LSTM model...')
#loaded_model = load_model('best_model.h5')

print('loading xgboost model...')
xgb_wrapper = load2var('best_xgboost_model.pkl')
#w_preds = xgb_wrapper.predict(X_test)
# 예측 결과 확인
#get_clf_eval(y_test, w_preds)

# input your words
input_data = ['이 영화 개꿀잼 ㅋㅋㅋ', 
              '이 영화 핵노잼 ㅠㅠ', 
              '이딴게 영화냐 ㅉㅉ', 
              '감독 뭐하는 놈이냐?', 
              '와 개쩐다 정말 세계관 최강자들의 영화다']

for s in input_data:
    sentiment_predict(s)
