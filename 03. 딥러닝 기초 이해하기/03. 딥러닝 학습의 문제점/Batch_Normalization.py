import numpy as np
import tensorflow as tf
from visual import *

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(200)
tf.random.set_seed(200)

# 배치 정규화를 적용할 모델과 비교하기 위한 기본 모델입니다.

def Basic():
    
    basic_model = tf.keras.Sequential([
                  tf.keras.layers.Flatten(input_shape=(28, 28)),
                  tf.keras.layers.Dense(256),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(128),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(512),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(64),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(128),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(256),
                  tf.keras.layers.Activation('relu'),
                  tf.keras.layers.Dense(10, activation='softmax')
                  ])
    
    return basic_model

'''
1. 기본 모델에 배치 정규화 레이어를 적용한 
   모델을 생성합니다. 입력층과 출력층은 그대로 사용합니다.
'''

def BN():
    
    bn_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return bn_model

'''
2. 두 개의 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. Basic, BN 함수를 이용해 두 모델을 불러옵니다.
   
   Step02. 두 모델의 손실 함수, 최적화 알고리즘, 
           평가 방법을 설정합니다.
   
   Step03. 두 모델의 구조를 확인하는 코드를 작성합니다.
   
   Step04. 두 모델을 학습시킵니다. 
           두 모델 모두 'epochs'는 20,
           'batch_size'는 500으로 설정합니다. 
           검증용 데이터도 설정해주세요.
   
   Step05. 두 모델을 테스트하고 accuracy 값을 출력합니다. 
           둘 중 어느 모델의 성능이 더 좋은지 확인해보세요.
'''

def main():
    
    # MNIST 데이터를 불러오고 전처리합니다.
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data, test_data = train_data / 255.0, test_data / 255.0
    
    basic_model = Basic()  # 기본 모델입니다.
    bn_model = BN()     # 배치 정규화를 적용할 모델입니다.
    
    basic_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    bn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    basic_model.summary()
    bn_model.summary()
    
    basic_history = basic_model.fit(train_data, train_labels, epochs = 20, batch_size = 500, validation_data = (test_data, test_labels), verbose = 0)
    print('\n')
    bn_history = bn_model.fit(train_data, train_labels, epochs = 20, batch_size = 500, validation_data = (test_data, test_labels), verbose = 0)
    
    scores_basic = basic_model.evaluate(test_data, test_labels, verbose = 0)
    scores_bn = bn_model.evaluate(test_data, test_labels, verbose = 0)
    
    print('\naccuracy_basic: ', scores_basic[-1])
    print('accuracy_bn: ', scores_bn[-1])
    
    Visulaize([('Basic', basic_history),('Batch Normalization', bn_history)])
    
    return basic_history, bn_history

if __name__ == "__main__":
    main()