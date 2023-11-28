from flask import Flask, render_template, request, session
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def register():
    fname = request.files['file']
    print(fname.filename)

    path = os.getcwd()
    path = os.path.join(path,'audio',fname.filename)
    with open(path, "wb") as f:  ## Excel File
        f.write(fname.getbuffer())

    #filename = secure_filename(file.filename)
    #print(filename)
    #path = os.getcwd()
    #file.save(os.path.join(path,'audio',filename))

    return "test"

@app.route('/record')
def ocr():
    return render_template('record.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)