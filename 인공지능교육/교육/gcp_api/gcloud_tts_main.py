import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/test/gcloud_key.json"
from google.cloud import texttospeech # pip install --upgrade google-cloud-texttospeech
from playsound import playsound # 음성파일 재생하는 라이브러리 : pip install playsound

import requests    # pip install requests
from datetime import datetime
import json
import time
"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
'''
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

# 합성할 text 지정
synthesis_input = texttospeech.SynthesisInput(text="Hello, World!")

# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

# The response's audio_content is binary.
with open("output.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
'''

# 텍스트 입력하면 음성파일로 저장하는 함수 만들기
def TextToSpeech(text) :

    client = texttospeech.TextToSpeechClient()

    # 합성할 text 지정
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name = 'ko-KR-Wavenet-D',
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("c:\\test\\output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')




# 기상예보 크롤링
def weather_crawling():
    key = 'GtbNVjtp2jvReVCmZ3V%2F4mPHVV02KkS3%2B86s%2BApiM4dnx8KoVGUWqmRZj7rH%2FJ96%2Ff9ZkBPMvTBTjNGuLoLaGA%3D%3D'
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService/getUltraSrtFcst?serviceKey=' + key + '&numOfRows=100&pageNo=1&base_date=' + datetime.today().strftime("%Y%m%d") + '&base_time=' + str(int(datetime.today().strftime("%H%M"))-100) + '&nx=62&ny=122&dataType=JSON'

    req = requests.get(url)
    html = req.content

    result = json.loads(html)

    for i in result['response']['body']['items']['item']:
        if i['category'] == 'T1H':
            print('현재 기온은 ', i['fcstValue'], '도 입니다.')
            return '현재 기온은 ', i['fcstValue'], '도 입니다.'
        '''
    for i in result['response']['body']['items']['item']:
        if i['category'] == 'POP':
            print('현재 강수 확률은 ', i['fcstValue'], '퍼센트 입니다.')
            '''


TextToSpeech(str(weather_crawling()))
time.sleep(2)
playsound('C:\\Users\\slinfo\\PycharmProjects\\0901_gcloud_TextToSpeech\\output.mp3', True)


# 시간 알려주는 함수
def timenow() :
    time = datetime.today().strftime("%H%M")
    if int(time[:2]) < 12 :
        return '현재 시간은 오전 ',time[:2],'시 ',time[2:],'분 입니다.'
    else :
        return '현재 시간은 오후 ',time[:2],'시 ',time[2:],'분 입니다.'




















