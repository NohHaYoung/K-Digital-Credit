import numpy as np
from konlpy.tag import Twitter

# 텍스트 파일을 읽어서 저장하는 함수입니다.

def read_txt(path):
    
    file = open(path, 'r')
    output = str(file.read())
    
    return output

'''
1. BoW를 딕셔너리 형태로 출력하는 함수를 만들어봅니다.

   Step01. 기존 word_dict 에 그 값이 있으면 1을 더해주고, 없으면 1을 부여합니다.
'''

def bag_of_words(tokenized_sentences):
    
    word_dict = dict()
    for sentence in tokenized_sentences:
        for word in sentence:
            try:
                word_dict[word] += 1
            except:
                word_dict[word] = 1
    
    return word_dict

'''
2. 읽어온 텍스트 파일을 형태소 단위로 분석해서 저장하는 
   함수를 완성하세요. 이전 실습을 참고해도 좋습니다.
'''

def get_splited_doc(path):
    
    output = []
    
    text = read_txt(path)
    analyzer = Twitter()
    output = analyzer.morphs(text)
    
    return text, output

def main():
    
    PATH = "./data/text.txt"
    
    origin, splitted = get_splited_doc(PATH)
    
    print('형법 제2장 원본: ', origin)
    
    bow_criminal_law = bag_of_words([splitted])
    
    print('\n형법 제2장의 BoW: ', bow_criminal_law)
    
    return bow_criminal_law

if __name__ == "__main__":
    main()