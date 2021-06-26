from tensorflow.keras.models import load_model
import pickle

# load to data variable
def load2var(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data

X_train = load2var('X_train.pkl')
X_test = load2var('X_test.pkl')
y_train = load2var('y_train.pkl')
y_test = load2var('y_test.pkl')

print('loading LSTM model...')
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
