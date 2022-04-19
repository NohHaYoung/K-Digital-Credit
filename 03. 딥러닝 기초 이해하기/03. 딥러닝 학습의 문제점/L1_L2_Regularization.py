import numpy as np
import tensorflow as tf
from visual import *

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 데이터를 전처리하는 함수

def sequences_shaping(sequences, dimension):
    
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0 
    
    return results

'''
1. L1, L2 정규화를 적용한 모델과 비교하기 위한
   하나의 기본 모델을 자유롭게 생성합니다.
'''

def Basic(word_num):
    
    basic_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape = (word_num, ), activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    return basic_model

'''
2. 기본 모델에 L1 정규화를 적용합니다.
   입력층과 히든층에만 적용하세요.
'''

def L1(word_num):
    
    l1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape = (word_num, ), activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.001)),
        tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.001)),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    return l1_model

'''
3. 기본 모델에 L2 정규화를 적용합니다.
   입력층과 히든층에만 적용하세요.
'''

def L2(word_num):
    
    l2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape = (word_num, ), activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    return l2_model


'''
4. 세 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. Basic, L1, L2 함수를 이용해 세 모델을 불러옵니다.
   
   Step02. 세 모델의 손실 함수, 최적화 알고리즘, 
           평가 방법을 설정합니다.
   
   Step03. 세 모델의 구조를 확인하는 코드를 작성합니다.
   
   Step04. 세 모델을 학습시킵니다. 
           세 모델 모두 'epochs'는 20,
           'batch_size'는 500으로 설정합니다. 
           검증용 데이터도 설정해주세요.
   
   Step05. 세 모델을 테스트하고 
           binary crossentropy 값을 출력합니다. 
           셋 중 어느 모델의 성능이 가장 좋은지 확인해보세요.
'''

def main():
    
    word_num = 100
    data_num = 25000
    
    # Keras에 내장되어 있는 imdb 데이터 세트를 불러오고 전처리합니다.
    
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words = word_num)
    
    train_data = sequences_shaping(train_data, dimension = word_num)
    test_data = sequences_shaping(test_data, dimension = word_num)
    
    basic_model = Basic(word_num)  # 기본 모델입니다.
    l1_model = L1(word_num)    # L1 정규화를 적용할 모델입니다.
    l2_model = L2(word_num)     # L2 정규화를 적용할 모델입니다.
    
    basic_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'binary_crossentropy'])
    l1_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'binary_crossentropy'])
    l2_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'binary_crossentropy'])
    
    basic_model.summary()
    l1_model.summary()
    l2_model.summary()
    
    basic_history = basic_model.fit(train_data, train_labels, epochs = 20, batch_size = 500, validation_data = (test_data, test_labels), verbose = 0)
    print('\n')
    l1_history = l1_model.fit(train_data, train_labels, epochs = 20, batch_size = 500, validation_data = (test_data, test_labels), verbose = 0)
    print('\n')
    l2_history = l2_model.fit(train_data, train_labels, epochs = 20, batch_size = 500, validation_data = (test_data, test_labels), verbose = 0)
    
    scores_basic = basic_model.evaluate(test_data, test_labels, verbose = 0)
    scores_l1 = l1_model.evaluate(test_data, test_labels, verbose = 0)
    scores_l2 = l2_model.evaluate(test_data, test_labels, verbose = 0)
    
    print('\nscores_basic: ', scores_basic[-1])
    print('scores_l1: ', scores_l1[-1])
    print('scores_l2: ', scores_l2[-1])
    
    Visulaize([('Basic', basic_history),('L1 Regularization', l1_history), ('L2 Regularization', l2_history)])
    
    return basic_history, l1_history, l2_history

if __name__ == "__main__":
    main()              