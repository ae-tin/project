import re
import os
import pandas as pd


## 추가
from tqdm import tqdm
import re

# 정규표현식 패턴
pattern0 = r'[가-힣]+\*'
pattern00 = r'\+'
#pattern1 = r'\(([\w\s]+)\)/\(([\w\s]+)\)'
pattern1 = r'\(([^)]+)\)/\(([^)]+)\)'
pattern11 = r'([^)]+)\)/\(([^)]+)\)'
pattern111 = r'\(([^)]+)/\(([^)]+)\)'
pattern1111 = r'\(([^)]+)\)/([^)]+)\)'
pattern11111 = r'\(([^)]+)\)/\(([^)]+)'
pattern111111 = r'\(([^)]+)\)/([^)]+)\(([^)]+)\)'
pattern2 = r'[가-힣]+/'
pattern3 = r'[A-Za-z]+/'
pattern4 = r'@[가-힣]+'
pattern5 = r'\s{2,}'


def replace_pattern0(match):   # 한글* 의 패턴 -> 한글 반환
    x = match[0][:-1]
    return x

def replace_pattern00(match):   # 한글+ 의 패턴 -> 한글 반환
    x = match[0][:-1]
    return x


def replace_pattern1(match):   #()/()의 패턴
    x = match.group(1)  
    y = match.group(2)  
    return y
def replace_pattern111111(match):   #()/()의 패턴
    y = match.group(3)  
    return y

def replace_pattern2(match):   # 한글/의 패턴 -> 한글 반환
    x = match[0][:-1]
    return x

def replace_pattern3(match):   # 엉어/의 패턴 -> 뒤에 공백제거 할거라서 공백으로 반환
    return ' '

def replace_pattern4(match):   # @이름 의 패턴 -> 이름 반환
    x = match[0][1:]
    return x

def replace_pattern5(match):   # '  '공백 두개 이상의 패턴 -> 없앰
    return ''

## 까지

def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]


def generate_character_script(data_df, labels_dest):
    print('[INFO] create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, "labels.csv"))

    with open(os.path.join(labels_dest,"transcripts.txt"), "w+") as f:
        for audio_path, transcript in tqdm(data_df.values):
            
            ## 추가 
            match0 = re.findall(pattern0, transcript)
            match00 = re.findall(pattern00, transcript)
            match1 = re.findall(pattern1, transcript)
            match11 = re.findall(pattern11, transcript)
            match111 = re.findall(pattern111, transcript)
            match1111 = re.findall(pattern1111, transcript)
            match11111 = re.findall(pattern11111, transcript)
            match111111 = re.findall(pattern111111, transcript)
            match2 = re.findall(pattern2, transcript)
            match3 = re.findall(pattern3, transcript)
            match4 = re.findall(pattern4, transcript)
            match5 = re.findall(pattern5, transcript)

            if match0 or match00 or match1 or match11 or match111 or match1111 or match11111 or match111111 or match2 or match3 or match4 or match5:
                result0 = re.sub(pattern0, replace_pattern0, transcript)
                result00 = re.sub(pattern00, replace_pattern00, result0)
                result1 = re.sub(pattern1, replace_pattern1, result00)
                result11 = re.sub(pattern111, replace_pattern1, result1)
                result111 = re.sub(pattern111, replace_pattern1, result11)
                result1111 = re.sub(pattern1111, replace_pattern1, result111)
                result11111 = re.sub(pattern11111, replace_pattern1, result1111)
                result111111 = re.sub(pattern111111, replace_pattern111111, result11111)
                result2 = re.sub(pattern2, replace_pattern2, result111111)
                result3 = re.sub(pattern3, replace_pattern3, result2)
                result4 = re.sub(pattern4, replace_pattern4, result3)
                transcript = re.sub(pattern5, replace_pattern5, result4)
            ## 까지

            char_id_transcript = sentence_to_target(transcript, char2id)
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')


def preprocessing(transcripts_dest, labels_dest):
    transcript_df = pd.read_csv(transcripts_dest)
    ## 바꾼거
    transcript_df.dropna(axis=0, how='any', subset=None, inplace=True)
    
    
    transcript_df = transcript_df#[:200000]
    ## 까지
    generate_character_script(transcript_df, labels_dest)

    print('[INFO] Preprocessing is Done')
    
    
############################ 밑에 내가 추가 #######################
    
    






