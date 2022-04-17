import numpy as np
import random

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

from elice_utils import EliceUtils
elice_utils = EliceUtils()

import warnings
warnings.filterwarnings(action='ignore')

np.random.seed(100)

'''
1. 손글씨 데이터를 X, Y로 읽어온 후 
   학습 데이터, 테스트 데이터로 나눕니다. 
   
   Step01. 학습 데이터는 앞의 1600개를 사용하고, 
           테스트 데이터는 학습 데이터를 제외한 나머지를 사용합니다.
           X, Y 데이터의 타입은 Numpy array라는 것을 참고하세요.
'''

def load_data(X, Y):
    
    X_train = X[ : 1600]
    Y_train = Y[ : 1600]
    
    X_test = X[1600 : ]
    Y_test = Y[1600 : ]
    
    return X_train, Y_train, X_test, Y_test
    
'''
2. MLPClassifier를 정의하고 hidden_layer_sizes를
   조정해 hidden layer의 크기 및 레이어의 개수를
   바꿔본 후, 학습을 시킵니다.
'''

def train_MLP_classifier(X, Y):
    
    clf = MLPClassifier(hidden_layer_sizes = (4, 4, 3))
    
    clf.fit(X, Y)
    
    return clf

'''
3. 정확도를 출력하는 함수를 완성합니다. 
   이전 실습에서 작성한 'score'를 그대로 
   사용할 수 있습니다.
'''

def report_clf_stats(clf, X, Y):
    
    hit = 0
    miss = 0
    
    for x, y in zip(X, Y):
        if clf.predict([x])[0] == y:
            hit += 1
        else:
            miss += 1
    
    score = (hit / miss) * 100
    
    print("Accuracy: %.1lf%% (%d hit / %d miss)" % (score, hit, miss))

'''
4. main 함수를 완성합니다.

   Step01. 훈련용 데이터와 테스트용 데이터를
           앞에서 완성한 함수를 이용해 불러옵니다.
           
   Step02. 앞에서 학습시킨 다층 퍼셉트론 분류 
           모델을 'clf'로 정의합니다.
           
   Step03. 앞에서 완성한 정확도 출력 함수를
           'score'로 정의합니다.
   
'''

def main():
    
    digits = load_digits()
    
    X = digits.data
    Y = digits.target

    X_train, Y_train, X_test, Y_test = load_data(X, Y)
    
    clf = train_MLP_classifier(X_train, Y_train)
    
    score = report_clf_stats(clf, X_test, Y_test)
    
    return score

if __name__ == "__main__":
    main()