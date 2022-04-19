import numpy as np
from visual import *

np.random.seed(100)

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def relu(x):
    result = np.maximum(0,x)
    return result

'''
1. 입력 데이터를 정의하세요.

2. 가중치 초깃값 설정 부분을 왼쪽 설명에 맞게 바꿔보세요.
   Numpy의 연산 메서드를 사용할 수 있습니다.
   
3. sigmoid와 relu를 통과할 값인 'a_sig', 'a_relu'를 정의하세요.
'''

def main():
    
    x_sig = np.random.randn(1000, 100)
    x_relu = np.random.randn(1000, 100)
    
    node_num = 100
    hidden_layer_size = 5
    
    activations_sig = {}
    activations_relu = {}
    
    for i in range(hidden_layer_size):
        if i != 0:
            x_sig = activations_sig[i-1]
            x_relu = activations_relu[i-1]
        
        w_sig =  np.random.randn(100, 100) / np.sqrt(node_num)
        w_relu =  np.random.randn(100, 100) / np.sqrt(node_num)
        
        a_sig = np.dot(x_sig, w_sig)
        a_relu = np.dot(x_relu, w_relu)
        
        z_sig = sigmoid(a_sig)
        z_relu = relu(a_relu)
        
        activations_sig[i] = z_sig
        activations_relu[i] = z_relu
        
    Visual(activations_sig, activations_relu)
    
    return activations_sig, activations_relu

if __name__ == "__main__":
    main()