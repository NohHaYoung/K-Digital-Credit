'''
1. 가중치와 Bias 값을 
   임의로 설정해줍니다.

   Step01. 0이상 1미만의 임의의 값으로 정의된 
           4개의 가중치 값이 들어가있는 
           1차원 리스트를 정의해줍니다.
           
   Step02. Bias 값을 임의의 값으로 설정해줍니다.
'''

def main():
    
    x = [1,2,3,4]
    
    w = [0.3, 0.5, 0.1, 0.7]
    b = -1
    
    output, y = perceptron(w,x,b)
    
    print('output: ', output)
    print('y: ', y)

'''
2. 신호의 총합과 그에 따른 결과 0 또는 1을
   반환하는 함수 perceptron을 완성합니다.
   
   Step01. 입력 받은 값과 Bias 값을 이용하여
           신호의 총합을 구합니다.
           
   Step02. 신호의 총합이 0 이상이면 1을, 
           그렇지 않으면 0을 반환하는 활성화 
           함수를 작성합니다.
'''

def perceptron(w, x, b):
    
    #output = x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + x[3] * w[3] + b
    output = sum(x_i * w_i for x_i, w_i in zip(x, w)) + b
        
    y = 1 if output > 0 else 0
    
    return output, y

if __name__ == "__main__":
    main()