import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(0)
tf.random.set_seed(0)

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# 데이터를 불러오고 전처리하는 함수입니다.

def load_data(n_of_training_ex, n_of_testing_ex, max_review_length):
    
    PATH = "./data/"
    
    X_train = np.load(PATH + "X_train.npy")[:n_of_training_ex]
    y_train = np.load(PATH + "y_train.npy")[:n_of_training_ex]
    X_test = np.load(PATH + "X_test.npy")[:n_of_testing_ex]
    y_test = np.load(PATH + "y_test.npy")[:n_of_testing_ex]
    
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    
    return X_train, y_train, X_test, y_test
    
'''
1. SimpleRNN을 적용할 하나의 모델을 자유롭게 생성합니다.
'''
    
def SimpleRNN(embedding_vector_length, max_review_length):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = 1000, output_dim = embedding_vector_length, input_length = max_review_length))
    model.add(tf.keras.layers.SimpleRNN(5))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    
    return model

'''
2. LSTM을 적용할 하나의 모델을 자유롭게 생성합니다.
'''

def LSTM(embedding_vector_length, max_review_length):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = 1000, output_dim = embedding_vector_length, input_length = max_review_length))
    model.add(tf.keras.layers.LSTM(5))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    
    return model

'''
3. GRU를 적용할 하나의 모델을 자유롭게 생성합니다.
'''

def GRU(embedding_vector_length, max_review_length):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = 1000, output_dim = embedding_vector_length, input_length = max_review_length))
    model.add(tf.keras.layers.GRU(5))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    
    return model

'''
4. 세 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. SimpleRNN, LSTM, GRU 함수를 이용해 세 모델을 불러옵니다.
   
   Step02. 세 모델의 손실 함수, 최적화 알고리즘, 평가 방법을 설정합니다.
   
   Step03. 세 모델의 구조를 확인하는 코드를 작성합니다.
   
   Step04. 세 모델을 각각 학습시킵니다. 검증용 데이터는 설정하지 않습니다.
           세 모델 모두 'epochs'는 3, 'batch_size'는 256으로 설정합니다.
   
   Step05. 세 모델을 테스트하고 각각의 Test Accuracy 값을 출력합니다. 
           셋 중 어느 모델의 성능이 가장 좋은지 확인해보세요.
'''

def main():
    
    max_review_length = 300
    embedding_vector_length = 32
    
    n_of_training_ex = 25000
    n_of_testing_ex = 3000
    
    X_train, y_train, X_test, y_test = load_data(n_of_training_ex, n_of_testing_ex, max_review_length)
    
    model_simple_rnn = SimpleRNN(embedding_vector_length, max_review_length)
    model_lstm = LSTM(embedding_vector_length, max_review_length)
    model_gru = GRU(embedding_vector_length, max_review_length)
    
    model_simple_rnn.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model_gru.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model_simple_rnn.summary()
    model_lstm.summary()
    model_gru.summary()
    
    model_simple_rnn_history = model_simple_rnn.fit(X_train, y_train, epochs = 3, batch_size = 256, verbose = 0)
    print('\n')
    model_lstm_history = model_lstm.fit(X_train, y_train, epochs = 3, batch_size = 256,  verbose = 0)
    print('\n')
    model_gru_history = model_gru.fit(X_train, y_train, epochs = 3, batch_size = 256, verbose = 0)
    
    scores_simple_rnn = model_simple_rnn.evaluate(X_test, y_test, verbose = 0)
    scores_lstm = model_lstm.evaluate(X_test, y_test, verbose = 0)
    scores_gru = model_gru.evaluate(X_test, y_test, verbose = 0)
    
    print('\nTest Accuracy_simple rnn: ', scores_simple_rnn[-1])
    print('Test Accuracy_lstm: ', scores_lstm[-1])
    print('Test Accuracy_gru: ', scores_gru[-1])
    
    return model_simple_rnn_history, model_lstm_history, model_gru_history

if __name__ == '__main__':
    main()