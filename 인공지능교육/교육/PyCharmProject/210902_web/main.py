from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import session            # 로그인을 유지시켜주는 세션!
import pymysql
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def home():
    conn = pymysql.connect(
	user='admin',
        password='qwer1234',
        host='database-1.cieed6lc0z0o.ap-northeast-2.rds.amazonaws.com',
        db='kti',
        charset='utf8')
    cursor = conn.cursor()
    sql = "SELECT * FROM board ORDER BY bnum DESC;"
    cursor.execute(sql)
    result = cursor.fetchall()

    return render_template('index.html', datas=result)

@app.route('/login')
def login() :
    return render_template('login.html')

# DB에서 사용자 확인후 로그인
@app.route('/dologin', methods = ['POST'])
def dologin():
    userid = request.form.get('userid')
    userpw = request.form.get('userpw')
  
    conn = pymysql.connect(
        user='admin',
        password='qwer1234',
        host='database-1.cieed6lc0z0o.ap-northeast-2.rds.amazonaws.com',
        db='kti',
        charset='utf8')
    cursor = conn.cursor()
    sql = "select mid, mpw, mname from member where mid = '"+userid+"' and mpw = '"+userpw+"';"

    cursor.execute(sql)
    result = cursor.fetchall()
    if len(result) == 0 :
        print('로그인실패')
        return redirect(url_for('login'))
    else :
        print('로그인성공')
        session.clear()
        session['logFlag'] = True
        session['mid'] = result[0][0]
        session['mname'] = result[0][2]
        return redirect(url_for('home'))

    return 'goodluck'

# 로그아웃 기능  - 
########################선생님 어떻게 로그아웃 적용시켰는지 보기
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# 회원가입 페이지 접속
@app.route('/register')
def register() :
    return render_template('register.html')

# DB에 회원 정보 저장
@app.route('/doregister', methods = ['POST'])
def doregister() :
    username = request.form.get('username')
    userid = request.form.get('userid')
    userpw = request.form.get('userpw')
    userpwagain = request.form.get('userpwagain')
    useremail = request.form.get('useremail')

    if userpw == userpwagain :
        conn = pymysql.connect(
            user='admin',
            password='qwer1234',
            host='database-1.cieed6lc0z0o.ap-northeast-2.rds.amazonaws.com',
            db='kti',
            charset='utf8')
        cursor = conn.cursor()
        sql = "INSERT INTO member (mid, mpw, mname, memail) values ('" + userid + "', '" + userpw + "', '" + username + "','"+useremail+"');"
        cursor.execute(sql)
        conn.commit()
        return redirect(url_for('login'))
    else :
        print('잘못된 입력입니다.')
        return redirect(url_for('register'))
    return 'goodluck'


@app.route('/password')
def password() :
    return render_template('password.html')

@app.route('/board')
def board() :

    return render_template('board.html')

#게시판 목록 가기
@app.route('/content')
def content() :

    return render_template('content.html')

# 게시물 작성
@app.route('/docontent', methods = ['POST'] )
def docontent() :

    title = request.form.get('title')
    contents = request.form.get('contents')
    writer = request.form.get('writer')

    conn = pymysql.connect(
        user='admin',
        password='qwer1234',
        host='database-1.cieed6lc0z0o.ap-northeast-2.rds.amazonaws.com',
        db='kti',
        charset='utf8')

    cursor = conn.cursor()
    sql = "INSERT INTO board (btitle, bcontents, bwriter,btime) values ('"+title+"','"+contents+"','"+writer+"',now());"
    cursor.execute(sql)
    conn.commit()

    return render_template('content.html')

@app.route('/doupdate')
def doupdate() :
    bnum =request.form.get('bnum')
    title = request.form.get('title')
    contents = request.form.get('contents')
    writer = request.form.get('writer')

    conn = pymysql.connect(
        user='admin',
        password='qwer1234',
        host='database-1.cieed6lc0z0o.ap-northeast-2.rds.amazonaws.com',
        db='kti',
        charset='utf8')
    cursor = conn.cursor()
    sql = "update board set btitle = '"+title+"' , bcontents = '"+contents+"' where bnum = '"+bnum+"';"
    cursor.execute(sql)
    conn.commit()

    return render_template('password.html')









if __name__ == '__main__' :
    app.run(host = '0.0.0.0', port = 80)


'''
{% if session.get('logFlag')%}
    
{% else %}
{% endif %}
'''









































