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
import  cv2
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from flask import render_template




app = Flask(__name__)

@app.route('/kmeans')
def kmeans():
    url = request.args.get('imageurl')
    knum = request.args.get('knum')

    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    k = int(knum)
    km = KMeans(n_clusters=k)
    km.fit(image)
    print(km.cluster_centers_)
    return render_template('result.html',data=km.cluster_centers_)



if __name__ == '__main__' :
    app.run(debug=True)









































