from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

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
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{} : {:.2f}% 확률로 긍정 리뷰입니다.\n".format(sentence, score * 100))
  else:
    print("{} : {:.2f}% 확률로 부정 리뷰입니다.\n".format(sentence, (1 - score) * 100))

#vocab_size = 19416
max_len = 30
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


print('loading tokenizer...')
tokenizer = load2var('tokenizer_nsmc.pkl')

print('loading CNN model...')
loaded_model = load_model('best_cnn_model.h5')

# input your words
input_data = ['이 영화 개꿀잼 ㅋㅋㅋ', 
              '이 영화 핵노잼 ㅠㅠ', 
              '이딴게 영화냐 ㅉㅉ', 
              '감독 뭐하는 놈이냐?', 
              '와 개쩐다 정말 세계관 최강자들의 영화다']

for s in input_data:
    sentiment_predict(s)
