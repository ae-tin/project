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

# k-means clustering

# 주어진 데이터셋을 이용하여 몇개의 클러스터를 구성할지 사전에 알 수 있을 때 사용

#channel - 고객채널, region - 고객지역, fresh - 신선제품연간지출
#milk- 유제품연간지출, grocery - 식료품연간지출, frozen- 냉동제품연간지출
#detergents_paper -세제및종이제품연간지출, delicassen- 조제식품연간지출

import pandas as pd

dataset = pd.read_csv('c:/test/sales_data.csv',header=0)

categorical_features = ['Channel','Region']
continuous_features = ['Fresh','Milk','Grocery','Frozen',
                        'Detergents_Paper','Delicassen']

# 자료유형
    #명목, 연속, 수치, 이산, 범주, 순위

for col in categorical_features :
    temp = pd.get_dummies(dataset[col], prefix=col)

    dataset = pd.concat([dataset, temp],axis=1)
    dataset.drop(col, axis=1, inplace = True)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

mms.fit(dataset)
data_transformed = mms.transform(dataset)

from sklearn.cluster import KMeans
distances = []

for k in range(1,15) :
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    distances.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,15), distances, 'bx-')
plt.xlabel('k')
plt.ylabel('distance')
plt.title('Optimal_k')
plt.show()





















































