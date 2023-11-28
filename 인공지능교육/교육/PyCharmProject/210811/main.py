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

import numpy as np

array1 = np.array(
    [
        [1,2],
        [3,4],
        [5,6]
    ]
)

array2 = array1.T
print(array1)
print(array2)


myarray1 = np.array([1,2,3,4,5,6])
myarray2 = np.array(
    [[1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15]])

print(myarray1[2:5])
print(myarray2[1:3, 2:5])

myarray3  = myarray2[1:3, 2:5]
print(myarray3)
print(myarray3.ndim)
print(myarray3.dtype)
print(myarray3.size)
print(myarray3.shape)

myarray4 = myarray3.reshape(1,6)
print(myarray4)

test = np.array(
    [
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25],
    ]
)


#1
print('문제')
print(test[1:4, 2:5])
print(test.T)
print(test.T[0:4, 3:5])
print(test[2:5, 1:3] @ test[0:2, 1:4])

