import numpy as np
from konlpy.tag import Twitter 

def load_data(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

'''
1. '문서'를 '문단'의 리스트로 변환하는 함수 doc2para를 완성합니다.

   Step01. 문장의 마침표(.) 뒤에 있는 개행 표시(\n)를 기준으로 
           문서 내 글들을 리스트 요소 즉, 문단으로 나눕니다.
           
   Step02. 나누어진 글들 중 마지막 글자가 "."인 경우만 
           문단이 나누어진 것으로 보고 그 외의 문장들은 서로 
           병합하여 줍니다.
'''

def doc2para(writing):
    
    paragraphs = []
    
    splited = writing.split('\n')
    splited = list(filter(lambda x : len(x) > 0, splited))
    para = ""
    
    for sentence in splited:
        if sentence[-1] != '.':
            para = para + sentence
        else:
            paragraphs.append(para)
            para = ""
    return paragraphs

'''
2. '문단'을 '문장'의 리스트로 변환하는 함수 para2sen을 완성합니다.

   Step01. 문단을 "."으로 나누어 리스트로 만들고, 
           변수 sentences에 저장합니다.
           
   Step02. sentences 내 문장들에 대해서 "?"로 재분할 한 후, 
           ndarray.flatten()을 활용하여 재분할된 문장이 합쳐질 
           수 있도록 리스트로 만들어 줍니다. 
           ("!"에 대해서도 마찬가지로 적용합니다.)
'''

def para2sen(paragraph):
    
    sentences = []
    
    # Step01.
    sentences = paragraph.split('.')
    
    # Step02. 
    sentences = [sentence.split('?') for sentence in sentences]
    sentences = np.array(sentences).flatten()
    sentences = [sentence.split('!') for sentence in sentences]
    sentences = np.array(sentences).flatten()
    sentences = [ sentence.replace('"','') for sentence in sentences]
    
    return sentences

# 띄어쓰기로 문장을 구분하는 함수

def sen2words_byspace(sentence):
    
    words = []
    words = sentence.strip().split(" ")
    
    return words

'''
3. 'Twitter()'로 선언된 Tokenizer인 'analyzer'를 이용해 형태소에 따라 
    분할된 문장의 리스트를 변수 'morphs'에 저장하는 sen2morph
    함수를 완성합니다. Twitter.morphs 메소드를 사용하세요.
'''

def sen2morph(sentence):
    
    morphs = []
    
    analyzer = Twitter()
    morphs = analyzer.morphs(sentence)
    
    return morphs

'''
4. 3번과 같이 'Twitter()'로 선언된 Tokenizer인 'analyzer'를 이용해
   형태소와 그에 따른 품사를 분할하는 analyzing_morphs 함수를 완성합니다.
   Twitter.pos 메소드를 사용하세요.
'''

def analyzing_morphs(sentence):
    
    analyzer = Twitter()
    
    return analyzer.pos(sentence)
    
# 위에서 정의한 함수들을 바탕으로 문서를 토큰화를 진행합니다.

def main():
    
    DATA_PATH = "./data/blood_rain.txt"
    
    blood_rain = load_data(DATA_PATH)
    paragraphs = doc2para(blood_rain)
    # print(paragraphs)
    sentences = para2sen(paragraphs[4])
    words_byspace = sen2words_byspace(sentences[3])
    words_bymorphs = sen2morph(sentences[3])
    morphs_analyzed = analyzing_morphs(sentences[3])
    
    # 출력을 통해 토큰화가 잘 되었는지 확인합니다.
    
    print("문장으로 구분된 5번째 문단: ", sentences)
    print("\n띄어쓰기로 구분된 문장 (5번째 문단의 4번째 문장): ", words_byspace)
    print("\n형태소 별로 구분된 문장 (5번째 문단의 4번째 문장): ", words_bymorphs)
    print("\n형태소와 그에 따른 품사로 분류된 문장 (5번째 문단의 4번째 문장): ", morphs_analyzed)
    
    return words_byspace, words_bymorphs, morphs_analyzed
    
if __name__=='__main__':
    main()