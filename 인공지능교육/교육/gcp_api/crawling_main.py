
# pip install requests
import requests
from datetime import datetime
import json

key = 'GtbNVjtp2jvReVCmZ3V%2F4mPHVV02KkS3%2B86s%2BApiM4dnx8KoVGUWqmRZj7rH%2FJ96%2Ff9ZkBPMvTBTjNGuLoLaGA%3D%3D'
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService/getUltraSrtFcst?serviceKey='+key+'&numOfRows=100&pageNo=1&base_date='+datetime.today().strftime("%Y%m%d")+'&base_time='+datetime.today().strftime("%H%M")+'&nx=62&ny=122&dataType=JSON'

req = requests.get(url)
html = req.content

result = json.loads(html)

for i in result['response']['body']['items']['item'] :
    if i['category'] == 'T1H' :
        print('현재 기온은 ', i['fcstValue'],'도 입니다.')
        break
for i in result['response']['body']['items']['item']:
    if i['category'] == 'POP' :
        print('현재 강수 확률은 ', i['fcstValue'],'퍼센트 입니다.')
















