# NLP(Natural Language Procesing) examples

본 예제는 자연어처리의 기본 지식을 학습하는 예제로서,    
자연어처리 중 분류(Classification)를 주제로 구성하였으며,    
샘플 데이터셋은 네이버 영화감성 분석 데이터셋(Naver Movie Review Sentiment Analysis)을 활용하였습니다.   

- 데이터셋 : [NSMC 데이터셋(https://github.com/e9t/nsmc/)](https://github.com/e9t/nsmc/)
- 소스 참고 
    - [네이버 영화 리뷰 감성 분류하기(Naver Movie Review Sentiment Analysis)](https://wikidocs.net/44249) <br>
    - [1D CNN으로 IMDB 리뷰 분류하기](https://wikidocs.net/80783) <br>
    - [사전 훈련된 워드 임베딩을 이용한 의도 분류(Intent Classification using Pre-trained Word Embedding)](https://wikidocs.net/86083) <br>

예제는 아래와 같이 구성되어 있습니다.

### 예제 구성
1. RNN 알고리즘을 이용한 분류 
    - 모델 훈련 : [01_lstm_train.py](https://github.com/rightlit/nlp/blob/main/examples/01_lstm_train.py)
    - 모델 평가 : [01_lstm_eval.py](https://github.com/rightlit/nlp/blob/main/examples/01_lstm_eval.py)
2. CNN 알고리즘을 이용한 분류 
    - 모델 훈련 : [02_cnn_train.py](https://github.com/rightlit/nlp/blob/main/examples/02_cnn_train.py)
    - 모델 평가 : [02_cnn_eval.py](https://github.com/rightlit/nlp/blob/main/examples/02_cnn_eval.py)
3. 사전훈련된 Word2Vec 을 이용한 분류


