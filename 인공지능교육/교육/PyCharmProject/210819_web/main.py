from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import session            # 로그인을 유지시켜주는 세션!
import pymysql
import os

# ctrl+shift+R 은 단어 검색해서 그 단어 모두 바꾸기 가능

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def home() :
    return render_template('index.html')

@app.route('/login')
def login() :
    return render_template('login.html')

@app.route('/dologin', methods = ['POST'])
def dologin():
    userid = request.form.get('userid')
    userpw = request.form.get('userpw')
    print(userid)
    conn = pymysql.connect(
        user='root',
        password='qwer1234',
        host='127.0.0.1',
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



@app.route('/register')
def register() :
    return render_template('register.html')

@app.route('/doregister', methods = ['POST'])
def doregister() :
    username = request.form.get('username')
    userid = request.form.get('userid')
    userpw = request.form.get('userpw')
    userpwagain = request.form.get('userpwagain')

    if userpw == userpwagain :
        conn = pymysql.connect(
            user='root',
            password='qwer1234',
            host='127.0.0.1',
            db='kti',
            charset='utf8')
        cursor = conn.cursor()
        sql = "INSERT INTO member (mid, mpw, mname) values ('" + userid + "', '" + userpw + "', '" + username + "');"
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

@app.route('/content')
def content() :

    return render_template('content.html')

@app.route('/docontent',methods = ['POST'])
def docontent() :

    title = request.form.get('title')
    contents = request.form.get('contents')
    writer = request.form.get('writer')

    conn = pymysql.connect(
        user='root',
        password='qwer1234',
        host='127.0.0.1',
        db='kti',
        charset='utf8')

    cursor = conn.cursor()
    sql = "INSERT INTO board (btitle, bcontents, bwriter,btime) values ('"+title+"','"+contents+"','"+writer+"',now());"
    cursor.execute(sql)
    conn.commit()

    return redirect(url_for('index'))

@app.route('/doupdate')
def doupdate() :
    bnum =request.form.get('bnum')
    title = request.form.get('title')
    contents = request.form.get('contents')
    writer = request.form.get('writer')

    conn = pymysql.connect(
        user='root',
        password='qwer1234',
        host='127.0.0.1',
        db='kti',
        charset='utf8')
    cursor = conn.cursor()
    sql = "update board set btitle = '"+title+"' , bcontents = '"+contents+"' where bnum = '"+bnum+"';"
    cursor.execute(sql)
    conn.commit()

    return render_template('password.html')









if __name__ == '__main__' :
    app.run(host = '0.0.0.0')


'''
{% if session.get('logFlag')%}
    
{% else %}
{% endif %}
'''









































