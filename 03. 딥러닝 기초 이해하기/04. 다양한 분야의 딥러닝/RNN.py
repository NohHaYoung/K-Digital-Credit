import numpy as np

np.random.seed(100)

'''
1. 간단한 RNN 모델을 만들어봅시다.

   Step01. 0의 값을 갖는 (output_size,) 모양의 'state' 벡터를 만들어 봅니다.
   
   Step02. 1의 값을 갖는 (output_size, input_size) 모양의 'w' 벡터를 만들어 봅니다.
   
   Step03. 1의 값을 갖는 (output_size, output_size) 모양의 'u' 벡터를 만들어 봅니다.
   
   Step04. 임의의 값을 갖는 (output_size,) 모양의 'b' 벡터를 만들어 봅니다.
   
   Step05. bias 가 False 이면 b를 (output_size,) 모양의 영벡터로 만듭니다.
   
   Step06. Numpy를 사용해서 w와 _input을 내적하고, u 와 state를 내적한 후
           b를 더한 다음 tanh 함수를 적용합니다.
'''

def rnn(inputs, input_size, output_size, bias = False):
    
    input_size = len(inputs[0])
    
    state = np.zeros((output_size, ))
    
    w = np.ones((output_size, input_size))
    
    u = np.ones((output_size, output_size))
    
    b = np.random.random((output_size, ))
    
    if not bias:
        b = np.zeros((output_size, ))
    
    outputs = []
    
    for _input in inputs:
        
        _output = np.tanh(np.dot(w, _input) + np.dot(u, state) + b)
        outputs.append(_output)
        state=_output
        
    return np.stack(outputs, axis=0)

# 케이스에 따라 RNN 모델의 결과가 어떻게 바뀌는지 확인해보세요.

def main():
    
    print("-----------------CASE 1-----------------")
    _input1 = [[0], [0], [0], [0], [0]]
    
    # 입력이 모두 0이고 출력 벡터의 크기가 1일 때 값의 추세가 어떠한지 확인해보세요.
    case1_a = rnn(_input1, input_size=1, output_size=1)
    print('\nCASE 1_a:', case1_a)
    # Bias가 있으면 값이 어떻게 변화하는지 확인해보세요.
    case1_b = rnn(_input1, input_size=1, output_size=1, bias = True)
    print('\nCASE 1_b:', case1_b)
    
    
    print("\n-----------------CASE 2-----------------")
    _input2 = [[1], [1], [1], [1], [1]]
    
    # 입력이 모두 1이고 출력 벡터의 크기가 1일 때 값의 추세가 어떠한지 확인해보세요.
    case2_a = rnn(_input2, input_size=1, output_size=1)
    print('\nCASE 2_a:', case2_a)
    # Bias가 있으면 값이 어떻게 변화하는지 확인해보세요.
    case2_b = rnn(_input2, input_size=1, output_size=1, bias = True)
    print('\nCASE 2_b:', case2_b)
    
    
    print("\n-----------------CASE 3-----------------")
    _input3 = [[1], [2], [3], [4], [5]]
    
    # 입력값이 증가하고 출력 벡터의 크기가 2일 때 값의 추세가 어떠한지 확인해보세요.
    case3_a = rnn(_input3, input_size=1, output_size=2)
    print('\nCASE 3_a:', case3_a)
    # Bias가 있으면 값이 어떻게 변화하는지 확인해보세요.
    case3_b = rnn(_input3, input_size=1, output_size=2, bias = True)
    print('\nCASE 3_b:', case3_b)
    
    return case1_a, case1_b, case2_a, case2_b, case3_a, case3_b

if __name__ == '__main__':
    main()