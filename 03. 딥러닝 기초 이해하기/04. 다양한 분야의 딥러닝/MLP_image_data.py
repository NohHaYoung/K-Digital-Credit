import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from visual import *

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

'''
1. MNIST 데이테 셋을 전처리하는 'preprocess' 함수를 완성합니다.

   Step01. MNIST 데이터 이미지를 
           0~1 사이 값으로 정규화해줍니다.
           원본은 0~255 사이의 값입니다.
           
   Step02. 0~9 사이 값인 label을 클래스화 하기 위해 
           원-핫 인코딩을 진행합니다.
'''

def preprocess():
    
    # MNIST 데이터 세트를 불러옵니다.
    mnist = tf.keras.datasets.mnist
    
    # MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()    
    
    # 이미지 정규화
    train_images = train_images / 255.
    test_images = test_images / 255.
    
    # label One-hot / 10개의 카테고리
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    return train_images, test_images, train_labels, test_labels

'''
2. 다층 퍼셉트론(MLP) 모델을 생성합니다.
'''
def MLP():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    
    return model
    
'''
3. 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. MLP 함수를 통해 모델을 불러옵니다.
   
   Step02. 모델의 손실 함수, 최적화 알고리즘, 
          평가 방법을 설정합니다.
   
   Step03. 모델의 구조를 확인하는 코드를 작성합니다.
   
   Step04. 모델을 학습시킵니다. 검증용 데이터도 설정하세요.
           'epochs'와 'batch_size'도 자유롭게 설정하세요.
              
   Step05. 모델을 테스트하고 손실(loss)값과 
           Test Accuracy 값 및 예측 클래스, 
           손실 함수값 그래프를 출력합니다. 
           
           모델의 성능을 확인해보고,
           목표값을 달성해보세요.
'''

def main():
    
    train_images, test_images, train_labels, test_labels = preprocess()
    
    model = MLP()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.summary() # 모델 구조 확인
    
    history = model.fit(train_images, train_labels, epochs = 20, batch_size = 128, validation_data = (test_images, test_labels), verbose = 2)
    
    loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0)
    
    print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
    print('예측한 Test Data 클래스 : ',model.predict_classes(test_images))
    
    Visulaize([('MLP', history)], 'loss')
    
    return history
    
if __name__ == "__main__":
    main()