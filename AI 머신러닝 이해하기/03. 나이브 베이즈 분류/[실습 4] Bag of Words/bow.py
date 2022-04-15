import re

special_chars_remover = re.compile("[^\w'|_]")

def main():
    sentence = input()
    bow = create_BOW(sentence)

    print(bow)


def create_BOW(sentence):
    bow = {}
    #1. 소문자로 치환
    lowered = sentence.lower()
    
    #2. 특수문자 제거
    without_special_characters = remove_special_characters(lowered)
    
    #3. space 기준으로 잘라내기
    splitted = without_special_characters.split()
    
    #4. 단어의 길이 체크
    splitted_filter = [
        token
        for token in splitted
        if len(token) >= 1
    ]
    
    for token in splitted_filter:
        bow.setdefault(token, 0)
        bow[token] += 1
    '''
        if token not in bow:
            bow[token] = 1
        else:
            bow[token] += 1
      '''      
    return bow


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()
