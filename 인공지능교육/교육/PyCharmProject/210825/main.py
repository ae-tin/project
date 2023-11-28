
import pymysql


conn = pymysql.connect(
    user = 'root',
    password='qwer1234',
    host='127.0.0.1',
    db='kti',
    charset='utf8')             # DB에 접속
'''
cursor = conn.cursor()          # 커서 생성,db와 상호작용하는 객체
sql = 'select mid,mpw, mname from member;'  # sql문 작성
cursor.execute(sql)             # 커서로 sql문 실행
result = cursor.fetchall()      # sql의 실행결과를 받아오기
print(result)                   # 출력

sql = "INSERT INTO member (mid, mpw, mname) values ('test02', 'qwer1234', 'tese02');"
cursor.execute(sql)
conn.commit()                   # DB에 변화가 생긴걸 적용,insert,update,delete등

sql = 'select mid,mpw, mname from member;'  # sql문 작성
cursor.execute(sql)             # 커서로 sql문 실행
result = cursor.fetchall()      # sql의 실행결과를 받아오기
print(result)
'''

print('아이디를 입력하세요 : ')
mid = input()
print('패스워드를 입력하세요 : ')
mpw = input()
print('이름를 입력하세요 : ')
mname = input()

cursor = conn.cursor()
sql = "INSERT INTO member (mid, mpw, mname) values ('"+mid+"', '"+mpw+"', '"+mname+"');"
cursor.execute(sql)
conn.commit()



