import os
import librosa

import numpy as np
import json



with open('C:\data\한국어 방언 발화 데이터(경상도)\Training\[라벨]경상도_학습데이터_1\DKCI20000002.json', 'r',encoding='utf-8') as f:

    json_data = json.load(f)

print(json.dumps(json_data, indent="\t") )
'''
def trim_audio_data(audio_file, save_file):
    sr = 16000
    sec = 30

    y, sr = librosa.load(audio_file, sr=sr)

    ny = y[:sr*sec]

    librosa.output.write_wav(save_file + '.wav', ny, sr)

base_path = 'dataset/'

audio_path = base_path + '/audio'
save_path = base_path + '/save'

audio_list = os.listdir(audio_path)

for audio_name in audio_list:
    if audio_name.find('wav') is not -1:
        audio_file = audio_path + '/' + audio_name
        save_file = save_path + '/' + audio_name[:-4]

        trim_audio_data(audio_file, save_file)
'''
























