# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/sum')     # /뒤에 파일이름?
def sub() :
    var1 = request.args.get('num1')
    var2 = request.args.get('num2')
    result = int(var1) + int(var2)
    return '결과 : '+str(result)

if __name__ == '__main__' :
    app.run(debug=True)

@app.route('/sub')     # /뒤에 파일이름?
def sub() :
    var1 = request.args.get('num1')
    var2 = request.args.get('num2')
    result = int(var1) - int(var2)
    return '결과 : '+str(result)












