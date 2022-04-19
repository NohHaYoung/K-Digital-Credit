import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

'''
1. embedding 함수를 완성합니다.

   Step01. 입력된 리스트 'sentence1+sentence2'에 존재하는
           요소마다 고유 인덱스를 붙입니다.
           
   Step02. 요소와 인덱스를 짝지은 딕셔너리 'word_dict'를 정의합니다.
   
   Step03. 'sentence1', 'sentence2'를 정수값으로 변환하고 이를
           각각 리스트 변수 'sen1', 'sen2'로 정의합니다.
'''

def embedding(sentence1, sentence2):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence1+sentence2)
    word_dict = tokenizer.word_index
    
    # word_dict의 value가 0부터 시작하게 바꿔줍니다.
    for k, v in word_dict.items():
        word_dict[k] = v - 1
    
    sen1 = tokenizer.texts_to_sequences(sentence1)
    sen2 = tokenizer.texts_to_sequences(sentence2)
    
    sen1 = [token[0] for token in sen1]
    sen2 = [token[0] for token in sen2]
    
    return word_dict, sen1, sen2

'''
2. 텐서플로우를 사용하여 원-핫 인코딩을 실행합니다.
'''   

def one_hot(sen1, sen2, word_dict):
    
    oh_sen1 = sum(tf.one_hot(sen1, len(word_dict)))
    oh_sen2 = sum(tf.one_hot(sen2, len(word_dict)))
    
    return oh_sen1, oh_sen2

def main():
    
    sentence1 = ['나','는','오늘','저녁','에','치킨','을','먹','을','예정','입니다']
    sentence2 = ['나','는','어제', '맥주','와', '함께', '치킨','을', '먹었', '습니다']
    
    word_dict, seq_1, seq_2 = efrom gensim.models import word2vec
import list_file

'''
1. CBOW 방식의 Word2Vec 모델을
   반환하는 CBOW 함수를 완성하세요.
'''

def CBOW(sentences):
    
    model_cbow = word2vec.Word2Vec(sentences, size = 300, min_count = 1, window = 10, sg = 0)
    
    return model_cbow

'''
2. Skip-Gram 방식의 Word2Vec 모델을
   반환하는 Skip_Gram 함수를 완성하세요.
'''

def Skip_Gram(sentences):
    
    model_skipgram = word2vec.Word2Vec(sentences, size = 300, min_count = 1, window = 10, sg = 1)
    
    return model_skipgram

'''
3. 각 모델의 결괏값을 정의하세요.
'''

def main():
    
    sen1, sen2 = list_file.sen1(), list_file.sen2()
    
    sentences = [sen1, sen2]
    
    cbow = CBOW(sentences)
    skipgram = Skip_Gram(sentences)
    
    idx2word_set_cbow = cbow.wv.index2word
    idx2word_set_skipgram = skipgram.wv.index2word
    
    print('CBOW: ', idx2word_set_cbow)
    print('\nSkip-Gram: ', idx2word_set_skipgram)
    
    return idx2word_set_cbow, idx2word_set_skipgram

if __name__ == '__main__':
    main()mbedding(sentence1, sentence2)
    onehot_sen1, onehot_sen2 = one_hot(seq_1, seq_2, word_dict)
        
    print('리스트 요소-인덱스 딕셔너리: ', word_dict)
    
    print('\n정수값으로 변환된 sentence1:', seq_1)
    print('\n정수값으로 변환된 sentence2:', seq_2)
    
    print('\n원-핫 인코딩된 문장1:', onehot_sen1.numpy())
    print('\n원-핫 인코딩된 문장2:', onehot_sen2.numpy())
    
    return onehot_sen1, onehot_sen2

if __name__ == '__main__':
    main()